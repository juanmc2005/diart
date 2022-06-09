import math
from queue import SimpleQueue
from typing import Text, Optional, Callable

import numpy as np
import sounddevice as sd
import torch
from einops import rearrange
from pyannote.core import SlidingWindowFeature, SlidingWindow
from rx.subject import Subject
from tqdm import tqdm

from .audio import FilePath, AudioLoader
from .features import TemporalFeatures


# TODO rename this to something else since the same API is also used to stream features
class AudioSource:
    """Represents a source of audio that can start streaming via the `stream` property.

    Parameters
    ----------
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the audio source.
    """
    def __init__(self, uri: Text, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    @property
    def is_regular(self) -> bool:
        """Whether the stream is regular. Defaults to False.
        A regular stream always yields the same amount of samples per event.
        """
        return False

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream if known. Defaults to None (unknown duration)"""
        return None

    @property
    def length(self) -> Optional[int]:
        """Return the number of audio chunks emitted by this source"""
        return None

    def read(self):
        """Start reading the source and yielding samples through the stream"""
        raise NotImplementedError


class FileAudioSource(AudioSource):
    """Represents an audio source tied to a file.

    Parameters
    ----------
    file: FilePath
        Path to the file to stream.
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the chunks emitted.
    chunk_duration: float
        Duration of each chunk in seconds. Defaults to 5s.
    step_duration: float
        Duration of the step between consecutive chunks in seconds. Defaults to 500ms.
    """
    def __init__(
        self,
        file: FilePath,
        uri: Text,
        sample_rate: int,
        chunk_duration: float = 5,
        step_duration: float = 0.5,
    ):
        super().__init__(uri, sample_rate)
        self.loader = AudioLoader(sample_rate, mono=True)
        self._duration = self.loader.get_duration(file)
        self.file = file
        self.chunk_duration = chunk_duration
        self.step_duration = step_duration
        self.resolution = 1 / sample_rate

    @property
    def is_regular(self) -> bool:
        # An audio file is always a regular source
        return True

    @property
    def duration(self) -> Optional[float]:
        # The duration of a file is known
        return self._duration

    @property
    def length(self) -> Optional[int]:
        return self.loader.get_num_sliding_chunks(
            self.file, self.chunk_duration, self.step_duration
        )

    def read(self):
        """Send each chunk of samples through the stream"""
        chunks = self.loader.load_sliding_chunks(
            self.file, self.chunk_duration, self.step_duration
        )
        for i, waveform in enumerate(chunks):
            window = SlidingWindow(
                start=i * self.step_duration,
                duration=self.resolution,
                step=self.resolution
            )
            chunk = SlidingWindowFeature(waveform.T, window)
            try:
                self.stream.on_next(chunk)
            except Exception as e:
                self.stream.on_error(e)
        self.stream.on_completed()


class PrecalculatedFeaturesAudioSource(FileAudioSource):
    def __init__(
        self,
        file: FilePath,
        uri: Text,
        sample_rate: int,
        segmentation: Callable[[TemporalFeatures], TemporalFeatures],
        embedding: Callable[[TemporalFeatures, TemporalFeatures], TemporalFeatures],
        chunk_duration: float = 5,
        step_duration: float = 0.5,
        batch_size: int = 32,
        progress_msg: Optional[Text] = None,
    ):
        super().__init__(file, uri, sample_rate, chunk_duration, step_duration)
        self.segmentation = segmentation
        self.embedding = embedding
        self.batch_size = batch_size
        self.progress_msg = progress_msg

    def read(self):
        # Split audio into chunks
        chunks = rearrange(
            self.loader.load_sliding_chunks(
                self.file, self.chunk_duration, self.step_duration
            ),
            "chunk channel sample -> chunk sample channel"
        )
        num_chunks = chunks.shape[0]

        # Set progress if needed
        iterator = range(0, num_chunks, self.batch_size)
        if self.progress_msg is not None:
            total = int(math.ceil(num_chunks / self.batch_size))
            iterator = tqdm(iterator, desc=self.progress_msg, total=total, unit="batch", leave=False)

        # Pre-calculate segmentation and embeddings
        segmentation, embeddings = [], []
        for i in iterator:
            i_end = i + self.batch_size
            if i_end > num_chunks:
                i_end = num_chunks
            batch = chunks[i:i_end]
            seg = self.segmentation(batch)
            # Edge case: add batch dimension if i == i_end + 1
            if seg.ndim == 2:
                seg = seg[np.newaxis]
            emb = self.embedding(batch, seg)
            # Edge case: add batch dimension if i == i_end + 1
            if emb.ndim == 2:
                emb = emb.unsqueeze(0)
            segmentation.append(seg)
            embeddings.append(emb)
        segmentation = np.vstack(segmentation)
        embeddings = torch.vstack(embeddings)

        # Stream pre-calculated segmentation, embeddings and chunks
        seg_resolution = self.chunk_duration / segmentation.shape[1]
        for i in range(num_chunks):
            chunk_window = SlidingWindow(
                start=i * self.step_duration,
                duration=self.resolution,
                step=self.resolution,
            )
            seg_window = SlidingWindow(
                start=i * self.step_duration,
                duration=seg_resolution,
                step=seg_resolution,
            )
            try:
                self.stream.on_next((
                    SlidingWindowFeature(chunks[i], chunk_window),
                    SlidingWindowFeature(segmentation[i], seg_window),
                    embeddings[i]
                ))
            except Exception as e:
                self.stream.on_error(e)

        self.stream.on_completed()


class MicrophoneAudioSource(AudioSource):
    """Represents an audio source tied to the default microphone available"""

    def __init__(self, sample_rate: int):
        super().__init__("live_recording", sample_rate)
        self.block_size = 1024
        self.mic_stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            latency=0,
            blocksize=self.block_size,
            callback=self._read_callback
        )
        self.queue = SimpleQueue()

    def _read_callback(self, samples, *args):
        self.queue.put_nowait(samples[:, [0]].T)

    def read(self):
        self.mic_stream.start()
        while self.mic_stream:
            try:
                self.stream.on_next(self.queue.get())
            except Exception as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
