from pathlib import Path
from typing import Sequence, Optional, Any, Union, List, Text, Tuple

import numpy as np
import torch
from pyannote.core import SlidingWindowFeature
from rx.core import Observer

from . import base
from .hparams import HyperParameter, TauActive
from .. import blocks
from .. import models as m
from .. import sinks
from .. import utils
from ..metrics import Metric, WordErrorRate


class TranscriptionConfig(base.PipelineConfig):
    def __init__(
        self,
        asr: Optional[m.SpeechRecognitionModel] = None,
        segmentation: Optional[m.SegmentationModel] = None,
        tau_active: float = 0.5,
        duration: Optional[float] = 3,
        language: Optional[Text] = None,
        beam_size: int = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.language = language
        self.beam_size = beam_size

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default ASR model is Whisper small (244M parameters)
        self.asr = asr
        if self.asr is None:
            self.asr = m.SpeechRecognitionModel.from_whisper("small")
        self.asr.set_language(self.language)
        self.asr.set_beam_size(self.beam_size)

        self.segmentation = segmentation
        self.tau_active = tau_active

        self._duration = duration
        self._sample_rate: Optional[int] = None

    @property
    def duration(self) -> float:
        if self._duration is None:
            self._duration = self.asr.duration
        return self._duration

    @property
    def step(self) -> float:
        return self.duration

    @property
    def latency(self) -> float:
        return self.duration

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            self._sample_rate = self.asr.sample_rate
        return self._sample_rate

    @staticmethod
    def from_dict(data: Any) -> 'TranscriptionConfig':
        # Check for explicit device, otherwise check for 'cpu' bool, otherwise pass None
        device = utils.get(data, "device", None)
        if device is None:
            device = torch.device("cpu") if utils.get(data, "cpu", False) else None

        # Default ASR model is Whisper small (244M parameters)
        whisper_size = utils.get(data, "whisper", "small")
        asr = m.SpeechRecognitionModel.from_whisper(whisper_size)

        # No VAD segmentation by default
        segmentation = utils.get(data, "segmentation", None)
        if segmentation is not None:
            hf_token = utils.parse_hf_token_arg(utils.get(data, "hf_token", True))
            segmentation = m.SegmentationModel.from_pyannote(segmentation, hf_token)

        # Tau hyper-parameter and its alias
        tau = utils.get(data, "tau_active", None)
        if tau is None:
            tau = utils.get(data, "tau", 0.5)

        return TranscriptionConfig(
            asr=asr,
            segmentation=segmentation,
            tau_active=tau,
            duration=utils.get(data, "duration", 3),
            language=utils.get(data, "language", None),
            beam_size=utils.get(data, "beam_size", None),
            device=device,
        )


class Transcription(base.Pipeline):
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self._config = TranscriptionConfig() if config is None else config
        self.asr = blocks.SpeechRecognition(self.config.asr, self.config.device)
        self.segmentation = None
        if self.config.segmentation is not None:
            self.segmentation = blocks.SpeakerSegmentation(self.config.segmentation, self.config.device)

    @staticmethod
    def get_config_class() -> type:
        return TranscriptionConfig

    @staticmethod
    def suggest_metric() -> Metric:
        return WordErrorRate()

    @staticmethod
    def hyper_parameters() -> Sequence[HyperParameter]:
        return [TauActive]

    @property
    def config(self) -> TranscriptionConfig:
        return self._config

    def reset(self):
        # No internal state. Nothing to do
        pass

    def set_timestamp_shift(self, shift: float):
        # No timestamped output. Nothing to do
        pass

    def join_predictions(self, predictions: List[Text]) -> Text:
        return "\n".join(predictions)

    def write_prediction(self, uri: Text, prediction: Text, dir_path: Union[Text, Path]):
        with open(Path(dir_path) / f"{uri}.txt", "w") as out_file:
            out_file.write(prediction)

    def suggest_writer(self, uri: Text, output_dir: Union[Text, Path]) -> Observer:
        return sinks.TextWriter(Path(output_dir) / f"{uri}.txt")

    def suggest_display(self) -> Observer:
        return sinks.RichScreen()

    def __call__(
        self,
        waveforms: Sequence[SlidingWindowFeature],
    ) -> Sequence[Tuple[Text, SlidingWindowFeature]]:
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(np.rint(self.config.duration * self.config.sample_rate))
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Run voice detection if required
        if self.segmentation is None:
            has_voice = torch.arange(0, batch_size)
        else:
            segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
            has_voice = torch.max(segmentations, dim=-1)[0]  # shape (batch, frames)
            has_voice = torch.any(has_voice >= self.config.tau_active, dim=-1)  # shape (batch,)
            has_voice = torch.where(has_voice)[0]

        # Return empty strings if no speech in the entire batch
        if len(has_voice) == 0:
            return [("", wav) for wav in waveforms]

        # Transcribe batch
        outputs = self.asr(batch[has_voice])
        mapping = {i_voice.item(): i_output for i_output, i_voice in enumerate(has_voice)}

        # No-speech outputs are empty strings
        return [
            (outputs[mapping[i]].text if i in has_voice else "", waveforms[i])
            for i in range(batch_size)
        ]
