from rx.subject import Subject
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import SlidingWindow
import random
from typing import Tuple
import time
import sounddevice as sd


class AudioSource:
    def __init__(self, uri: str, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    def read(self):
        raise NotImplementedError


class FileAudioSource(AudioSource):
    def __init__(self, file: AudioFile, uri: str, sample_rate: int):
        super().__init__(uri, sample_rate)
        self.audio = Audio(sample_rate=sample_rate, mono=True)
        self.file = file

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
        duration: float,
        step: float
    ):
        super().__init__(file, uri, sample_rate)
        self.duration = duration
        self.step = step

    def to_iterable(self):
        duration = self.audio.get_duration(self.file)
        window = SlidingWindow(start=0., duration=self.duration, step=self.step, end=duration)
        for chunk in window:
            waveform, sample_rate = self.audio.crop(self.file, chunk, fixed=self.duration)
            yield waveform


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
        waveform, sample_rate = self.audio(self.file)
        total_samples = waveform.shape[1]
        i = 0
        while i < total_samples:
            rnd_duration = random.uniform(self.start, self.end)
            if self.delay:
                time.sleep(rnd_duration)
            num_samples = int(round(rnd_duration * sample_rate))
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
