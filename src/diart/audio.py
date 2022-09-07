from pathlib import Path
from typing import Text, Union

import torch
import torchaudio
from torchaudio.functional import resample

torchaudio.set_audio_backend("soundfile")


FilePath = Union[Text, Path]


class AudioLoader:
    def __init__(self, sample_rate: int, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, filepath: FilePath) -> torch.Tensor:
        """Load an audio file into a torch.Tensor.

        Parameters
        ----------
        filepath : FilePath
            Path to an audio file

        Returns
        -------
        waveform : torch.Tensor, shape (channels, samples)
        """
        waveform, sample_rate = torchaudio.load(filepath)
        # Get channel mean if mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if self.sample_rate != sample_rate:
            waveform = resample(waveform, sample_rate, self.sample_rate)
        return waveform

    @staticmethod
    def get_duration(filepath: FilePath) -> float:
        """Get audio file duration in seconds.

        Parameters
        ----------
        filepath : FilePath
            Path to an audio file.

        Returns
        -------
        duration : float
            Duration in seconds.
        """
        info = torchaudio.info(filepath)
        return info.num_frames / info.sample_rate
