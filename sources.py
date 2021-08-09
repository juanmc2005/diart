import rx
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.core import SlidingWindow
import random
from typing import Tuple


class ReliableFileAudioStream:
    def __init__(self, duration: float, step: float, sample_rate: float):
        self.audio = Audio(sample_rate=sample_rate, mono=True)
        self.duration = duration
        self.step = step

    def to_iterable(self, file: AudioFile):
        duration = self.audio.get_duration(file)
        window = SlidingWindow(start=0., duration=self.duration, step=self.step, end=duration)
        for chunk in window:
            waveform, sample_rate = self.audio.crop(file, chunk, fixed=self.duration)
            yield waveform

    def __call__(self, file: AudioFile):
        return rx.from_iterable(self.to_iterable(file))


class UnreliableFileAudioStream:
    def __init__(self, refresh_rate_range: Tuple[float, float], sample_rate: float):
        self.audio = Audio(sample_rate=sample_rate, mono=True)
        self.start, self.end = refresh_rate_range

    def to_iterable(self, file: AudioFile):
        waveform, sample_rate = self.audio(file)
        total_samples = waveform.shape[1]
        i = 0
        while i < total_samples:
            rnd_duration = random.uniform(self.start, self.end)
            num_samples = int(round(rnd_duration * sample_rate))
            last_i = i
            i += num_samples
            yield waveform[:, last_i:i]

    def __call__(self, file: AudioFile):
        return rx.from_iterable(self.to_iterable(file))
