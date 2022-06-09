from queue import SimpleQueue
from typing import Text, Optional

import sounddevice as sd
from pyannote.core import SlidingWindowFeature, SlidingWindow
from rx.subject import Subject

from .audio import FilePath, AudioLoader


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
