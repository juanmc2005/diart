import torch
from torchcodec.decoders import AudioDecoder

from .storage import FilePath, PathLike

__all__ = ["AudioLoader", "FilePath", "PathLike"]


class AudioLoader:
    def __init__(self, sample_rate: int, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, filepath: PathLike) -> torch.Tensor:
        """Load an audio file into a torch.Tensor.

        Parameters
        ----------
        filepath : PathLike
            Path to an audio file (local or remote).

        Returns
        -------
        waveform : torch.Tensor, shape (channels, samples)
        """
        # torchcodec resamples to the target sample rate while decoding.
        # localize() downloads remote files and is a no-op for local ones.
        with FilePath(filepath).localize() as local:
            decoder = AudioDecoder(str(local), sample_rate=self.sample_rate)
            waveform = decoder.get_all_samples().data
        # Get channel mean if mono
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    @staticmethod
    def get_duration(filepath: PathLike) -> float:
        """Get audio file duration in seconds.

        Parameters
        ----------
        filepath : PathLike
            Path to an audio file (local or remote).

        Returns
        -------
        duration : float
            Duration in seconds.
        """
        with FilePath(filepath).localize() as local:
            return AudioDecoder(str(local)).metadata.duration_seconds
