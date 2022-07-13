from pathlib import Path
from typing import Text, Union

import numpy as np
import torch
import torchaudio
from einops import rearrange
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
            Path to an audio file

        Returns
        -------
        duration : float
            Duration in seconds.
        """
        info = torchaudio.info(filepath)
        return info.num_frames / info.sample_rate

    def load_sliding_chunks(self, filepath: FilePath, chunk_duration: float, step_duration: float) -> np.ndarray:
        """Load an audio file and extract sliding chunks of a given duration with a given step duration.

        Parameters
        ----------
        filepath : FilePath
            Path to an audio file
        chunk_duration: float
            Duration of the chunk in seconds.
        step_duration: float
            Duration of the step between chunks in seconds.
        """
        chunk_samples = int(round(chunk_duration * self.sample_rate))
        step_samples = int(round(step_duration * self.sample_rate))
        waveform = self.load(filepath)
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, chunk_samples, step_samples),
            "channel chunk sample -> chunk channel sample",
        ).numpy()
        # Add padded last chunk
        if num_samples - chunk_samples % step_samples > 0:
            last_chunk = waveform[:, chunks.shape[0] * step_samples:].unsqueeze(0).numpy()
            diff_samples = chunk_samples - last_chunk.shape[-1]
            last_chunk = np.concatenate([last_chunk, np.zeros((1, 1, diff_samples))], axis=-1)
            return np.vstack([chunks, last_chunk])
        return chunks

    def get_num_sliding_chunks(self, filepath: FilePath, chunk_duration: float, step_duration: float) -> int:
        """Estimate the number of sliding chunks of a
        given chunk duration and step without loading the audio.

        Parameters
        ----------
        filepath : FilePath
            Path to an audio file
        chunk_duration: float
            Duration of the chunk in seconds.
        step_duration: float
            Duration of the step between chunks in seconds.
        """
        numerator = self.get_duration(filepath) - chunk_duration + step_duration
        return int(np.ceil(numerator / step_duration))
