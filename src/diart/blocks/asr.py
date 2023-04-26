from pathlib import Path
from typing import Optional, Union, List, Text

import torch
from einops import rearrange

from .. import models as m
from ..features import TemporalFeatureFormatter, TemporalFeatures


class SpeechRecognition:
    def __init__(self, model: m.SpeechRecognitionModel, device: Optional[torch.device] = None):
        self.model = model
        self.model.eval()
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.formatter = TemporalFeatureFormatter()

    @staticmethod
    def from_whisper(
        name: Text,
        download_path: Optional[Union[Text, Path]] = None,
        in_memory: bool = False,
        fp16: bool = False,
        no_speech_threshold: float = 0.6,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1,
        decode_with_fallback: bool = False,
        device: Optional[Union[Text, torch.device]] = None,
    ) -> 'SpeechRecognition':
        asr_model = m.SpeechRecognitionModel.from_whisper(
            name,
            download_path,
            in_memory,
            fp16,
            no_speech_threshold,
            compression_ratio_threshold,
            logprob_threshold,
            decode_with_fallback,
        )
        return SpeechRecognition(asr_model, device)

    def __call__(self, waveform: TemporalFeatures) -> List[m.TranscriptionResult]:
        """
        Compute the transcription of input audio.

        Parameters
        ----------
        waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)
            Audio to transcribe

        Returns
        -------
        transcriptions: List[Transcription]
            A list of timestamped transcriptions
        """
        with torch.no_grad():
            wave = rearrange(
                self.formatter.cast(waveform),
                "batch sample channel -> batch channel sample"
            )
            # output = self.model(wave.to(self.device)).cpu()
            output = self.model(wave.to(self.device))
        return output
