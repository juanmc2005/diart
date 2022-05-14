import random
import time
from queue import SimpleQueue
from typing import Tuple, Text, Optional, Iterable, List

import numpy as np
import sounddevice as sd
from einops import rearrange
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import SlidingWindowFeature, SlidingWindow
from rx.subject import Subject


class ChunkLoader:
    """Loads an audio file and chunks it according to a given window and step size.

    Parameters
    ----------
    sample_rate: int
        Sample rate to load audio.
    window_duration: float
        Duration of the chunk in seconds.
    step_duration: float
        Duration of the step between chunks in seconds.
    """

    def __init__(
        self,
        sample_rate: int,
        window_duration: float,
        step_duration: float,
    ):
        self.audio = Audio(sample_rate, mono=True)
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.window_samples = int(round(window_duration * sample_rate))
        self.step_samples = int(round(step_duration * sample_rate))

    def get_chunks(self, file: AudioFile) -> np.ndarray:
        waveform, _ = self.audio(file)
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.window_samples, self.step_samples),
            "channel chunk frame -> chunk channel frame",
        ).numpy()
        # Add padded last chunk
        if num_samples - self.window_samples % self.step_samples > 0:
            last_chunk = waveform[:, chunks.shape[0] * self.step_samples:].unsqueeze(0).numpy()
            diff_samples = self.window_samples - last_chunk.shape[-1]
            last_chunk = np.concatenate([last_chunk, np.zeros((1, 1, diff_samples))], axis=-1)
            return np.vstack([chunks, last_chunk])
        return chunks

    def num_chunks(self, file: AudioFile) -> int:
        numerator = self.audio.get_duration(file) - self.window_duration + self.step_duration
        return int(np.ceil(numerator / self.step_duration))


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


class AudioFileReader:
    """Represents a method for reading an audio file.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the audio file.
    """
    def __init__(self, sample_rate: int):
        self.audio = Audio(sample_rate=sample_rate, mono=True)
        self.resolution = 1 / sample_rate

    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @property
    def is_regular(self) -> bool:
        """Whether the reading is regular. Defaults to False.
        A regular reading method always yields the same amount of samples."""
        return False

    def get_duration(self, file: AudioFile) -> float:
        return self.audio.get_duration(file)

    def get_num_chunks(self, file: AudioFile) -> Optional[int]:
        return None

    def iterate(self, file: AudioFile) -> Iterable[SlidingWindowFeature]:
        """Return an iterable over the file's samples"""
        raise NotImplementedError


class RegularAudioFileReader(AudioFileReader):
    """Reads a file always yielding the same number of samples with a given step.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the audio file.
    window_duration: float
        Duration of each chunk of samples (window) in seconds.
    step_duration: float
        Step duration between chunks in seconds.
    """
    def __init__(
        self,
        sample_rate: int,
        window_duration: float,
        step_duration: float,
    ):
        super().__init__(sample_rate)
        self.chunk_loader = ChunkLoader(
            sample_rate, window_duration, step_duration
        )

    @property
    def is_regular(self) -> bool:
        return True

    def get_num_chunks(self, file: AudioFile) -> Optional[int]:
        """Return the number of chunks emitted for `file`"""
        return self.chunk_loader.num_chunks(file)

    def iterate(self, file: AudioFile) -> Iterable[SlidingWindowFeature]:
        chunks = self.chunk_loader.get_chunks(file)
        for i, chunk in enumerate(chunks):
            w = SlidingWindow(
                start=i * self.chunk_loader.step_duration,
                duration=self.resolution,
                step=self.resolution
            )
            yield SlidingWindowFeature(chunk.T, w)


class IrregularAudioFileReader(AudioFileReader):
    """Reads an audio file yielding a different number of non-overlapping samples in each event.
    This class is useful to simulate how a system would work in unreliable reading conditions.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the audio file.
    refresh_rate_range: (float, float)
        Duration range within which to determine the number of samples to yield (in seconds).
    simulate_delay: bool
        Whether to simulate that the samples are being read in real time before they are yielded.
        Defaults to False (no delay).
    """
    def __init__(
        self,
        sample_rate: int,
        refresh_rate_range: Tuple[float, float],
        simulate_delay: bool = False,
    ):
        super().__init__(sample_rate)
        self.start, self.end = refresh_rate_range
        self.delay = simulate_delay

    def iterate(self, file: AudioFile) -> Iterable[SlidingWindowFeature]:
        waveform, _ = self.audio(file)
        total_samples = waveform.shape[1]
        i = 0
        while i < total_samples:
            rnd_duration = random.uniform(self.start, self.end)
            if self.delay:
                time.sleep(rnd_duration)
            num_samples = int(round(rnd_duration * self.sample_rate))
            last_i = i
            i += num_samples
            yield waveform[:, last_i:i]


class FileAudioSource(AudioSource):
    """Represents an audio source tied to a file.

    Parameters
    ----------
    file: AudioFile
        The file to stream.
    uri: Text
        Unique identifier of the audio source.
    reader: AudioFileReader
        Determines how the file will be read.
    profile: bool
        If True, prints the average processing time of emitting a chunk. Defaults to False.
    """
    def __init__(
        self,
        file: AudioFile,
        uri: Text,
        reader: AudioFileReader,
        profile: bool = False,
    ):
        super().__init__(uri, reader.sample_rate)
        self.reader = reader
        self._duration = self.reader.get_duration(file)
        self.file = file
        self.profile = profile

    @property
    def is_regular(self) -> bool:
        # The regularity depends on the reader
        return self.reader.is_regular

    @property
    def duration(self) -> Optional[float]:
        # The duration of a file is known
        return self._duration

    @property
    def length(self) -> Optional[int]:
        # Only the reader can know how many chunks are going to be emitted
        return self.reader.get_num_chunks(self.file)

    def _check_print_time(self, times: List[float]):
        if self.profile:
            print(
                f"File {self.uri}: took {np.mean(times).item():.2f} seconds/chunk "
                f"(+/- {np.std(times).item():.2f} seconds/chunk) "
                f"-- based on {len(times)} inputs"
            )

    def read(self):
        """Send each chunk of samples through the stream"""
        times = []
        for waveform in self.reader.iterate(self.file):
            try:
                if self.profile:
                    # Profiling assumes that on_next is blocking
                    start_time = time.monotonic()
                    self.stream.on_next(waveform)
                    times.append(time.monotonic() - start_time)
                else:
                    self.stream.on_next(waveform)
            except Exception as e:
                self._check_print_time(times)
                self.stream.on_error(e)
        self.stream.on_completed()
        self._check_print_time(times)


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
