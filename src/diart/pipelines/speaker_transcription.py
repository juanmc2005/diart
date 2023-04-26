from pathlib import Path
from typing import Any, Optional, Union, Sequence, Tuple, Text, List

import numpy as np
import torch
from diart.metrics import Metric
from pyannote.core import SlidingWindowFeature, SlidingWindow, Annotation, Segment
from rx.core import Observer
from typing_extensions import Literal

from .base import Pipeline, PipelineConfig
from .diarization import SpeakerDiarization, SpeakerDiarizationConfig
from .hparams import HyperParameter, TauActive, RhoUpdate, DeltaNew
from .. import models as m
from .. import sinks
from .. import blocks
from .. import utils
from ..metrics import WordErrorRate


class SpeakerAwareTranscriptionConfig(PipelineConfig):
    def __init__(
        self,
        asr: Optional[m.SpeechRecognitionModel] = None,
        segmentation: Optional[m.SegmentationModel] = None,
        embedding: Optional[m.EmbeddingModel] = None,
        duration: Optional[float] = None,
        asr_duration: float = 3,
        step: float = 0.5,
        latency: Optional[Union[float, Literal["max", "min"]]] = None,
        tau_active: float = 0.5,
        rho_update: float = 0.3,
        delta_new: float = 1,
        language: Optional[Text] = None,
        beam_size: Optional[int] = None,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        merge_collar: float = 0.05,
        diarization_device: Optional[torch.device] = None,
        asr_device: Optional[torch.device] = None,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation
        if self.segmentation is None:
            self.segmentation = m.SegmentationModel.from_pyannote("pyannote/segmentation")

        self._duration = duration
        self._sample_rate: Optional[int] = None

        # Default embedding model is pyannote/embedding
        self.embedding = embedding
        if self.embedding is None:
            self.embedding = m.EmbeddingModel.from_pyannote("pyannote/embedding")

        # Latency defaults to the step duration
        self._step = step
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self.duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.merge_collar = merge_collar
        self.asr_duration = asr_duration

        self.diarization_device = diarization_device
        if self.diarization_device is None:
            self.diarization_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.language = language
        self.beam_size = beam_size

        self.asr_device = asr_device
        if self.asr_device is None:
            self.asr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default ASR model is Whisper small (244M parameters)
        self.asr = asr
        if self.asr is None:
            self.asr = m.SpeechRecognitionModel.from_whisper("small")
        self.asr.set_language(self.language)
        self.asr.set_beam_size(self.beam_size)

    def to_diarization_config(self) -> SpeakerDiarizationConfig:
        return SpeakerDiarizationConfig(
            segmentation=self.segmentation,
            embedding=self.embedding,
            duration=self.duration,
            step=self.step,
            latency=self.latency,
            tau_active=self.tau_active,
            rho_update=self.rho_update,
            delta_new=self.delta_new,
            gamma=self.gamma,
            beta=self.beta,
            max_speakers=self.max_speakers,
            merge_collar=self.merge_collar,
            device=self.diarization_device,
        )

    @property
    def duration(self) -> float:
        # Default duration is the one given by the segmentation model
        if self._duration is None:
            self._duration = self.segmentation.duration
        return self._duration

    @property
    def step(self) -> float:
        return self._step

    @property
    def latency(self) -> float:
        return self._latency

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            dia_sample_rate = self.segmentation.sample_rate
            asr_sample_rate = self.asr.sample_rate
            msg = "Sample rates for speech recognition and speaker segmentation models must match"
            assert dia_sample_rate == asr_sample_rate, msg
            self._sample_rate = dia_sample_rate
        return self._sample_rate

    @staticmethod
    def from_dict(data: Any) -> 'SpeakerAwareTranscriptionConfig':
        # Resolve arguments exactly like diarization
        dia_config = SpeakerDiarizationConfig.from_dict(data)

        # Default ASR model is Whisper small (244M parameters)
        whisper_size = utils.get(data, "whisper", "small")
        asr = m.SpeechRecognitionModel.from_whisper(whisper_size)

        return SpeakerAwareTranscriptionConfig(
            asr=asr,
            segmentation=dia_config.segmentation,
            embedding=dia_config.embedding,
            duration=dia_config.duration,
            asr_duration=utils.get(data, "asr_duration", 3),
            step=dia_config.step,
            latency=dia_config.latency,
            tau_active=dia_config.tau_active,
            rho_update=dia_config.rho_update,
            delta_new=dia_config.delta_new,
            language=utils.get(data, "language", None),
            beam_size=utils.get(data, "beam_size", None),
            gamma=dia_config.gamma,
            beta=dia_config.beta,
            max_speakers=dia_config.max_speakers,
            merge_collar=dia_config.merge_collar,
            diarization_device=dia_config.device,
            # TODO handle different devices
            asr_device=dia_config.device,
        )


class SpeakerAwareTranscription(Pipeline):
    def __init__(self, config: Optional[SpeakerAwareTranscriptionConfig] = None):
        self._config = SpeakerAwareTranscriptionConfig() if config is None else config
        self.diarization = SpeakerDiarization(self.config.to_diarization_config())
        self.asr = blocks.SpeechRecognition(self.config.asr, self.config.asr_device)

        # Internal state, handle with care
        self.audio_buffer, self.dia_buffer = None, None

    @staticmethod
    def get_config_class() -> type:
        return SpeakerAwareTranscriptionConfig

    @staticmethod
    def suggest_metric() -> Metric:
        # TODO per-speaker WER?
        return WordErrorRate()

    @staticmethod
    def hyper_parameters() -> Sequence[HyperParameter]:
        return [TauActive, RhoUpdate, DeltaNew]

    @property
    def config(self) -> SpeakerAwareTranscriptionConfig:
        return self._config

    def reset(self):
        self.diarization.reset()
        self.audio_buffer, self.dia_buffer = None, None

    def set_timestamp_shift(self, shift: float):
        self.diarization.set_timestamp_shift(shift)

    def join_predictions(self, predictions: List[Text]) -> Text:
        return "\n".join(predictions)

    def write_prediction(self, uri: Text, prediction: Text, dir_path: Union[Text, Path]):
        with open(Path(dir_path) / f"{uri}.txt", "w") as out_file:
            out_file.write(prediction)

    def suggest_display(self) -> Observer:
        return sinks.RichScreen()

    def suggest_writer(self, uri: Text, output_dir: Union[Text, Path]) -> Observer:
        return sinks.TextWriter(Path(output_dir) / f"{uri}.txt")

    def _update_buffers(self, diarization_output: Sequence[Tuple[Annotation, SlidingWindowFeature]]):
        # Separate diarization and aligned audio chunks
        first_chunk = diarization_output[0][1]
        output_start = first_chunk.extent.start
        resolution = first_chunk.sliding_window.duration
        diarization, chunk_data = Annotation(), []
        for dia, chunk in diarization_output:
            diarization = diarization.update(dia)
            chunk_data.append(chunk.data)

        # Update diarization output buffer
        if self.dia_buffer is None:
            self.dia_buffer = diarization
        else:
            self.dia_buffer = self.dia_buffer.update(diarization)
        self.dia_buffer = self.dia_buffer.support(self.config.merge_collar)

        # Update audio buffer
        if self.audio_buffer is None:
            window = SlidingWindow(resolution, resolution, output_start)
            self.audio_buffer = SlidingWindowFeature(np.concatenate(chunk_data, axis=0), window)
        else:
            chunk_data.insert(0, self.audio_buffer.data)
            self.audio_buffer = SlidingWindowFeature(
                np.concatenate(chunk_data, axis=0),
                self.audio_buffer.sliding_window,
            )

    def _extract_asr_inputs(self) -> Tuple[List[SlidingWindowFeature], List[Annotation]]:
        chunk_duration = self.config.asr_duration
        buffer_duration = self.audio_buffer.extent.duration
        batch_size = int(buffer_duration / chunk_duration)
        buffer_start = self.audio_buffer.extent.start
        resolution = self.audio_buffer.sliding_window.duration

        # Extract audio chunks with their diarization
        asr_inputs, input_dia, last_end_time = [], [], None
        for i in range(batch_size):
            start = buffer_start + i * chunk_duration
            last_end_time = start + chunk_duration
            region = Segment(start, last_end_time)
            chunk = self.audio_buffer.crop(region, fixed=chunk_duration)
            window = SlidingWindow(resolution, resolution, start)
            asr_inputs.append(SlidingWindowFeature(chunk, window))
            input_dia.append(self.dia_buffer.crop(region))

        # Remove extracted chunks from buffers
        if asr_inputs:
            new_buffer_bounds = Segment(last_end_time, self.audio_buffer.extent.end)
            new_buffer = self.audio_buffer.crop(new_buffer_bounds, fixed=new_buffer_bounds.duration)
            window = SlidingWindow(resolution, resolution, last_end_time)
            self.audio_buffer = SlidingWindowFeature(new_buffer, window)
            self.dia_buffer = self.dia_buffer.extrude(Segment(0, last_end_time))

        return asr_inputs, input_dia

    def _get_speaker_transcriptions(
        self,
        input_diarization: List[Annotation],
        asr_inputs: List[SlidingWindowFeature],
        asr_outputs: List[m.TranscriptionResult],
    ) -> Text:
        transcriptions = []
        for i, waveform in enumerate(asr_inputs):
            if waveform is None:
                continue
            buffer_shift = waveform.sliding_window.start
            for text, timestamp in zip(asr_outputs[i].chunks, asr_outputs[i].timestamps):
                if not text.strip():
                    continue
                target_region = Segment(
                    buffer_shift + timestamp.start,
                    buffer_shift + timestamp.end,
                )
                dia = input_diarization[i].crop(target_region)
                speakers = dia.labels()
                num_speakers = len(speakers)
                if num_speakers == 0:
                    # Include transcription but don't assign a speaker
                    transcriptions.append(text)
                elif num_speakers == 1:
                    # Typical case, annotate text with the only speaker
                    transcriptions.append(f"[{speakers[0]}]{text}")
                else:
                    # Multiple speakers for the same text block, choose the most active one
                    max_spk = np.argmax([dia.label_duration(spk) for spk in speakers])
                    transcriptions.append(f"[{speakers[max_spk]}]{text}")
        return " ".join(transcriptions).strip()

    def __call__(
        self,
        waveforms: Sequence[SlidingWindowFeature],
    ) -> Sequence[Text]:
        # Compute diarization output
        diarization_output = self.diarization(waveforms)
        self._update_buffers(diarization_output)

        # Extract audio to transcribe from the buffer
        asr_inputs, asr_input_dia = self._extract_asr_inputs()
        if not asr_inputs:
            return ["" for _ in waveforms]

        # Detect non-speech chunks
        has_voice = torch.tensor([dia.get_timeline().duration() > 0 for dia in asr_input_dia])
        has_voice = torch.where(has_voice)[0]
        # Return empty strings if no speech in the entire batch
        if len(has_voice) == 0:
            return ["" for _ in waveforms]

        # Create ASR batch, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in asr_inputs])

        # Transcribe batch
        asr_outputs = self.asr(batch[has_voice])
        asr_outputs = [
            asr_outputs[i] if i in has_voice else None
            for i in range(batch.shape[0])
        ]

        # Attach speaker labels to ASR output and concatenate
        transcription = self._get_speaker_transcriptions(
            asr_input_dia, asr_inputs, asr_outputs
        )

        # Fill output sequence with empty strings
        batch_size = len(waveforms)
        output = [transcription]
        if batch_size > 1:
            output += [""] * (batch_size - 1)

        return output
