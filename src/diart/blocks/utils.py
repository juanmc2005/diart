from typing import Text, Optional

import numpy as np
import torch
from pyannote.core import Annotation, Segment, SlidingWindowFeature
import torchaudio.transforms as T

from ..features import TemporalFeatures, TemporalFeatureFormatter


class Binarize:
    """
    Transform a speaker segmentation from the discrete-time domain
    into a continuous-time speaker segmentation.

    Parameters
    ----------
    threshold: float
        Probability threshold to determine if a speaker is active at a given frame.
    uri: Optional[Text]
        Uri of the audio stream. Defaults to no uri.
    """

    def __init__(self, threshold: float, uri: Optional[Text] = None):
        self.uri = uri
        self.threshold = threshold

    def __call__(self, segmentation: SlidingWindowFeature) -> Annotation:
        """
        Return the continuous-time segmentation
        corresponding to the discrete-time input segmentation.

        Parameters
        ----------
        segmentation: SlidingWindowFeature
            Discrete-time speaker segmentation.

        Returns
        -------
        annotation: Annotation
            Continuous-time speaker segmentation.
        """
        num_frames, num_speakers = segmentation.data.shape
        timestamps = segmentation.sliding_window
        is_active = segmentation.data > self.threshold
        # Artificially add last inactive frame to close any remaining speaker turns
        is_active = np.append(is_active, [[False] * num_speakers], axis=0)
        start_times = np.zeros(num_speakers) + timestamps[0].middle
        annotation = Annotation(uri=self.uri, modality="speech")
        for t in range(num_frames):
            # Any (False, True) starts a speaker turn at "True" index
            onsets = np.logical_and(np.logical_not(is_active[t]), is_active[t + 1])
            start_times[onsets] = timestamps[t + 1].middle
            # Any (True, False) ends a speaker turn at "False" index
            offsets = np.logical_and(is_active[t], np.logical_not(is_active[t + 1]))
            for spk in np.where(offsets)[0]:
                region = Segment(start_times[spk], timestamps[t + 1].middle)
                annotation[region, spk] = f"speaker{spk}"
        return annotation


class Resample:
    """Dynamically resample audio chunks.

    Parameters
    ----------
    sample_rate: int
        Original sample rate of the input audio
    resample_rate: int
        Sample rate of the output
    """
    def __init__(self, sample_rate: int, resample_rate: int):
        self.resample = T.Resample(sample_rate, resample_rate)
        self.formatter = TemporalFeatureFormatter()

    def __call__(self, waveform: TemporalFeatures) -> TemporalFeatures:
        wav = self.formatter.cast(waveform)  # shape (batch, samples, 1)
        with torch.no_grad():
            resampled_wav = self.resample(wav.transpose(-1, -2)).transpose(-1, -2)
        return self.formatter.restore_type(resampled_wav)


class AdjustVolume:
    """Change the volume of an audio chunk.

    Notice that the output volume might be different to avoid saturation.

    Parameters
    ----------
    volume_in_db: float
        Target volume in dB.
    """
    def __init__(self, volume_in_db: float):
        self.target_db = volume_in_db
        self.formatter = TemporalFeatureFormatter()

    @staticmethod
    def get_volumes(waveforms: torch.Tensor) -> torch.Tensor:
        """Compute the volumes of a set of audio chunks.

        Parameters
        ----------
        waveforms: torch.Tensor
            Audio chunks. Shape (batch, samples, channels).

        Returns
        -------
        volumes: torch.Tensor
            Audio chunk volumes per channel. Shape (batch, 1, channels)
        """
        return 10 * torch.log10(torch.mean(torch.abs(waveforms) ** 2, dim=1, keepdim=True))

    def __call__(self, waveform: TemporalFeatures) -> TemporalFeatures:
        wav = self.formatter.cast(waveform)  # shape (batch, samples, channels)
        with torch.no_grad():
            # Compute current volume per chunk, shape (batch, 1, channels)
            current_volumes = self.get_volumes(wav)
            # Determine gain to reach the target volume
            gains = 10 ** ((self.target_db - current_volumes) / 20)
            # Apply gain
            wav = gains * wav
            # If maximum value is greater than one, normalize chunk
            maximums = torch.clamp(torch.amax(torch.abs(wav), dim=1, keepdim=True), 1)
            wav = wav / maximums
        return self.formatter.restore_type(wav)
