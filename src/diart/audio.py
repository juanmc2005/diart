from pathlib import Path
from typing import Text, Union

import torch
from torchcodec.decoders import AudioDecoder

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
        # torchcodec resamples to the target sample rate while decoding
        decoder = AudioDecoder(str(filepath), sample_rate=self.sample_rate)
        waveform = decoder.get_all_samples().data
        # Get channel mean if mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
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
        return AudioDecoder(str(filepath)).metadata.duration_seconds
