from typing import Optional, Tuple, Sequence

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment

from .aggregation import DelayedAggregation
from .clustering import OnlineSpeakerClustering
from .embedding import OverlapAwareSpeakerEmbedding
from .segmentation import SpeakerSegmentation
from .utils import Binarize
from .config import BasePipelineConfig, PipelineConfig


class BasePipeline:
    @staticmethod
    def get_config_class() -> type:
        raise NotImplementedError

    @property
    def config(self) -> BasePipelineConfig:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def set_timestamp_shift(self, shift: float):
        raise NotImplementedError

    def __call__(
        self,
        waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[Tuple[Annotation, SlidingWindowFeature]]:
        raise NotImplementedError


class OnlineSpeakerDiarization(BasePipeline):
    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = PipelineConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = SpeakerSegmentation(self._config.segmentation, self._config.device)
        self.embedding = OverlapAwareSpeakerEmbedding(
            self._config.embedding, self._config.gamma, self._config.beta, norm=1, device=self._config.device
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
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []
        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return PipelineConfig

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def reset(self):
        self.set_timestamp_shift(0)
        self.clustering = OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        self.chunk_buffer, self.pred_buffer = [], []

    def __call__(
        self,
        waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[Tuple[Annotation, SlidingWindowFeature]]:
        batch_size = len(waveforms)
        msg = "Pipeline expected at least 1 input"
        assert batch_size >= 1, msg

        # Create batch from chunk sequence, shape (batch, samples, channels)
        batch = torch.stack([torch.from_numpy(w.data) for w in waveforms])

        expected_num_samples = int(np.rint(self.config.duration * self.config.sample_rate))
        msg = f"Expected {expected_num_samples} samples per chunk, but got {batch.shape[1]}"
        assert batch.shape[1] == expected_num_samples, msg

        # Extract segmentation and embeddings
        segmentations = self.segmentation(batch)  # shape (batch, frames, speakers)
        embeddings = self.embedding(batch, segmentations)  # shape (batch, speakers, emb_dim)

        seg_resolution = waveforms[0].extent.duration / segmentations.shape[1]

        outputs = []
        for wav, seg, emb in zip(waveforms, segmentations, embeddings):
            # Add timestamps to segmentation
            sw = SlidingWindow(
                start=wav.extent.start,
                duration=seg_resolution,
                step=seg_resolution,
            )
            seg = SlidingWindowFeature(seg.cpu().numpy(), sw)

            # Update clustering state and permute segmentation
            permuted_seg = self.clustering(seg, emb)

            # Update sliding buffer
            self.chunk_buffer.append(wav)
            self.pred_buffer.append(permuted_seg)

            # Aggregate buffer outputs for this time step
            agg_waveform = self.audio_aggregation(self.chunk_buffer)
            agg_prediction = self.pred_aggregation(self.pred_buffer)
            agg_prediction = self.binarize(agg_prediction)

            # Shift prediction timestamps if required
            if self.timestamp_shift != 0:
                shifted_agg_prediction = Annotation(agg_prediction.uri)
                for segment, track, speaker in agg_prediction.itertracks(yield_label=True):
                    new_segment = Segment(
                        segment.start + self.timestamp_shift,
                        segment.end + self.timestamp_shift,
                    )
                    shifted_agg_prediction[new_segment, track] = speaker
                agg_prediction = shifted_agg_prediction

            outputs.append((agg_prediction, agg_waveform))

            # Make place for new chunks in buffer if required
            if len(self.chunk_buffer) == self.pred_aggregation.num_overlapping_windows:
                self.chunk_buffer = self.chunk_buffer[1:]
                self.pred_buffer = self.pred_buffer[1:]

        return outputs
