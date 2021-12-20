from rx.subject import Subject
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import SlidingWindowFeature, SlidingWindow
import random
from typing import Tuple
import time
import sounddevice as sd
from einops import rearrange


class AudioSource:
    def __init__(self, uri: str, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()
        self.resolution = 1 / sample_rate

    @property
    def is_regular(self) -> bool:
        return False

    @property
    def duration(self):
        return None

    def read(self):
        raise NotImplementedError


class FileAudioSource(AudioSource):
    def __init__(self, file: AudioFile, uri: str, sample_rate: int):
        super().__init__(uri, sample_rate)
        self.audio = Audio(sample_rate=sample_rate, mono=True)
        self._duration = self.audio.get_duration(file)
        self.file = file

    @property
    def duration(self):
        return self._duration

    def to_iterable(self):
        raise NotImplementedError

    def read(self):
        for waveform in self.to_iterable():
            try:
                self.stream.on_next(waveform)
            except Exception as e:
                self.stream.on_error(e)
        self.stream.on_completed()


class ReliableFileAudioSource(FileAudioSource):
    def __init__(
        self,
        file: AudioFile,
        uri: str,
        sample_rate: int,
        window_duration: float,
        step: float
    ):
        super().__init__(file, uri, sample_rate)
        self.window_duration = window_duration
        self.step = step
        self.window_samples = int(round(self.window_duration * self.sample_rate))
        self.step_samples = int(round(self.step * self.sample_rate))

    @property
    def is_regular(self) -> bool:
        return True

    def to_iterable(self):
        waveform, _ = self.audio(self.file)
        chunks = rearrange(
            waveform.unfold(1, self.window_samples, self.step_samples),
            "channel chunk frame -> chunk channel frame",
        ).numpy()
        for i, chunk in enumerate(chunks):
            w = SlidingWindow(
                start=i * self.step,
                duration=self.resolution,
                step=self.resolution
            )
            yield SlidingWindowFeature(chunk.T, w)


class UnreliableFileAudioSource(FileAudioSource):
    def __init__(
        self,
        file: AudioFile,
        uri: str,
        sample_rate: int,
        refresh_rate_range: Tuple[float, float],
        simulate_delay: bool = False
    ):
        super().__init__(file, uri, sample_rate)
        self.start, self.end = refresh_rate_range
        self.delay = simulate_delay

    def to_iterable(self):
        waveform, _ = self.audio(self.file)
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


class MicrophoneAudioSource(AudioSource):
    def __init__(self, sample_rate: int):
        super().__init__("live_recording", sample_rate)
        self.block_size = 1024
        self.mic_stream = sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            latency=0,
            blocksize=self.block_size,
        )

    def read(self):
        self.mic_stream.start()
        while self.mic_stream:
            try:
                samples = self.mic_stream.read(self.block_size)[0]
            except Exception as e:
                self.stream.on_error(e)
                break
            self.stream.on_next(samples[:, [0]].T)
        self.stream.on_completed()
