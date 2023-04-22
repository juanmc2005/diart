from pathlib import Path
from typing import Sequence, Optional, Any, Union, List, Text, Tuple

import numpy as np
import torch
from einops import rearrange
from pyannote.core import SlidingWindowFeature

from . import base
from .. import models as m
from .. import utils
from ..blocks.base import HyperParameter
from ..features import TemporalFeatureFormatter, TemporalFeatures
from ..metrics import Metric, WordErrorRate

BeamSize = HyperParameter("beam_size", low=1, high=20)


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
        device: Optional[Union[Text, torch.device]] = None,
    ) -> 'SpeechRecognition':
        asr_model = m.SpeechRecognitionModel.from_whisper(
            name, download_path, in_memory, fp16
        )
        return SpeechRecognition(asr_model, device)

    def __call__(self, waveform: TemporalFeatures) -> List[m.Transcription]:
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


class TranscriptionConfig(base.StreamingConfig):
    def __init__(
        self,
        asr: Optional[m.SpeechRecognitionModel] = None,
        duration: Optional[float] = None,
        language: Optional[Text] = None,
        beam_size: int = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default ASR model is Whisper small (244M parameters)
        self.asr = asr
        if self.asr is None:
            self.asr = m.SpeechRecognitionModel.from_whisper("small")
        self.asr.set_language(language)
        self.asr.set_beam_size(beam_size)

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

        name = utils.get(data, "whisper", "small")
        asr = m.SpeechRecognitionModel.from_whisper(name)

        return TranscriptionConfig(
            asr=asr,
            duration=utils.get(data, "duration", None),
            language=utils.get(data, "language", None),
            beam_size=utils.get(data, "beam_size", None),
            device=device,
        )


class Transcription(base.StreamingPipeline):
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self._config = TranscriptionConfig() if config is None else config
        self.asr = SpeechRecognition(self.config.asr, self.config.device)

    @staticmethod
    def get_config_class() -> type:
        return TranscriptionConfig

    @staticmethod
    def suggest_metric() -> Metric:
        return WordErrorRate()

    @staticmethod
    def hyper_parameters() -> Sequence[HyperParameter]:
        return [BeamSize]

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

        # Transcribe batch
        # TODO only transcribe if there's speech
        outputs = self.asr(batch)

        return [(out.text, wav) for out, wav in zip(outputs, waveforms)]

        # TODO align text with speakers if diarization is not None

        # diarization = diarization[0]
        #
        # # Align transcription with diarization to determine speakers
        # full_transcription = []
        # buffer_shift = waveform.sliding_window.start
        # for text, timestamp in zip(outputs.chunks, outputs.timestamps):
        #     target_region = Segment(
        #         buffer_shift + timestamp.start,
        #         buffer_shift + timestamp.end
        #     )
        #     dia = diarization.crop(target_region)
        #     speakers = dia.labels()
        #     num_speakers = len(speakers)
        #     if num_speakers == 0:
        #         # Include transcription but don't assign a speaker
        #         full_transcription.append(text)
        #     elif num_speakers == 1:
        #         # Typical case, annotate text with the only speaker
        #         full_transcription.append(f"[{speakers[0]}]{text}")
        #     else:
        #         # Multiple speakers for the same text block, choose the most active one
        #         max_spk = np.argmax([dia.label_duration(spk) for spk in speakers])
        #         full_transcription.append(f"[{speakers[max_spk]}]{text}")
        #
        # return [(" ".join(full_transcription).strip(), waveform)]
