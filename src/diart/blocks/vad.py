from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from pyannote.core import (
    Annotation,
    Timeline,
    SlidingWindowFeature,
    SlidingWindow,
    Segment,
)
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.detection import DetectionErrorRate
from typing_extensions import Literal

from . import base
from .aggregation import DelayedAggregation
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .. import models as m
from .. import utils


class VoiceActivityDetectionConfig(base.PipelineConfig):
    def __init__(
        self,
        segmentation: m.SegmentationModel | None = None,
        duration: float | None = None,
        step: float = 0.5,
        latency: float | Literal["max", "min"] | None = None,
        tau_active: float = 0.6,
        device: torch.device | None = None,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation or m.SegmentationModel.from_pyannote(
            "pyannote/segmentation"
        )

        self._duration = duration
        self._step = step
        self._sample_rate: int | None = None

        # Latency defaults to the step duration
        self._latency = latency
        if self._latency is None or self._latency == "min":
            self._latency = self._step
        elif self._latency == "max":
            self._latency = self._duration

        self.tau_active = tau_active
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
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
        # Expected sample rate is given by the segmentation model
        if self._sample_rate is None:
            self._sample_rate = self.segmentation.sample_rate
        return self._sample_rate


class VoiceActivityDetection(base.Pipeline):
    def __init__(self, config: VoiceActivityDetectionConfig | None = None):
        self._config = VoiceActivityDetectionConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = SpeakerSegmentation(
            self._config.segmentation, self._config.device
        )
        self.pred_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = Binarize(self._config.tau_active)

        # Internal state, handle with care
        self.timestamp_shift = 0
        self.chunk_buffer, self.pred_buffer = [], []

    @staticmethod
    def get_config_class() -> type:
        return VoiceActivityDetectionConfig

    @staticmethod
    def suggest_metric() -> BaseMetric:
        return DetectionErrorRate(collar=0, skip_overlap=False)

    @staticmethod
    def hyper_parameters() -> Sequence[base.HyperParameter]:
        return [base.TauActive]

    @property
    def config(self) -> base.PipelineConfig:
        return self._config

    def reset(self):
        self.set_timestamp_shift(0)
        self.chunk_buffer, self.pred_buffer = [], []

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def __call__(
        self,
        waveforms: Sequence[SlidingWindowFeature],
    ) -> Sequence[tuple[Annotation, SlidingWindowFeature]]:
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(
            np.rint(self.config.duration * self.config.sample_rate)
        )
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Extract segmentation
        segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
        voice_detection = torch.max(segmentations, dim=-1, keepdim=True)[
            0
        ]  # shape (batch, frames, 1)

        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]

        outputs = []
        for wav, vad in zip(waveforms, voice_detection):
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
            vad = SlidingWindowFeature(vad.cpu().numpy(), sw)

            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(vad)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            agg_prediction = self.binarize(agg_prediction).get_timeline(copy=False)

            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Timeline(uri=agg_prediction.uri)
                for segment in agg_prediction:
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction.add(new_segment)
                agg_prediction = shifted_agg_prediction

            # Convert timeline into annotation with single speaker "speech"
            agg_prediction = agg_prediction.to_annotation(utils.repeat_label("speech"))
            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        return outputs
