from typing import Optional, Union, Text

import torch
from einops import rearrange

from ..features import TemporalFeatures, TemporalFeatureFormatter
from ..models import SegmentationModel


class SpeakerSegmentation:
    def __init__(self, model: SegmentationModel, device: Optional[torch.device] = None):
        self.model = model
        self.model.eval()
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.formatter = TemporalFeatureFormatter()

    @staticmethod
    def from_pyannote(
        model,
        use_hf_token: Union[Text, bool, None] = True,
        device: Optional[torch.device] = None
    ) -> 'SpeakerSegmentation':
        seg_model = SegmentationModel.from_pyannote(model, use_hf_token)
        return SpeakerSegmentation(seg_model, device)

    def __call__(self, waveform: TemporalFeatures) -> TemporalFeatures:
        """
        Calculate the speaker segmentation of input audio.

        Parameters
        ----------
        waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)

        Returns
        -------
        speaker_segmentation: TemporalFeatures, shape (batch, frames, speakers)
            The batch dimension is omitted if waveform is a `SlidingWindowFeature`.
        """
        with torch.no_grad():
            wave = rearrange(self.formatter.cast(waveform), "batch sample channel -> batch channel sample")
            output = self.model(wave.to(self.device)).cpu()
        return self.formatter.restore_type(output)
