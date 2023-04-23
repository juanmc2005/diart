from pathlib import Path
from typing import Optional, Tuple, Sequence, Union, Any, Text, List

import numpy as np
import torch
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow, Segment
from rx.core import Observer
from typing_extensions import Literal

from . import base
from .hparams import HyperParameter, TauActive, RhoUpdate, DeltaNew
from .. import blocks
from .. import models as m
from .. import sinks
from .. import utils
from ..metrics import Metric, DiarizationErrorRate


class SpeakerDiarizationConfig(base.StreamingConfig):
    def __init__(
        self,
        segmentation: Optional[m.SegmentationModel] = None,
        embedding: Optional[m.EmbeddingModel] = None,
        duration: Optional[float] = None,
        step: float = 0.5,
        latency: Optional[Union[float, Literal["max", "min"]]] = None,
        tau_active: float = 0.5,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        merge_collar: float = 0.05,
        device: Optional[torch.device] = None,
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
            self._latency = self._duration

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.merge_collar = merge_collar

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def from_dict(data: Any) -> 'SpeakerDiarizationConfig':
        # Check for explicit device, otherwise check for 'cpu' bool, otherwise pass None
        device = utils.get(data, "device", None)
        if device is None:
            device = torch.device("cpu") if utils.get(data, "cpu", False) else None

        # Instantiate models
        hf_token = utils.parse_hf_token_arg(utils.get(data, "hf_token", True))
        segmentation = utils.get(data, "segmentation", "pyannote/segmentation")
        segmentation = m.SegmentationModel.from_pyannote(segmentation, hf_token)
        embedding = utils.get(data, "embedding", "pyannote/embedding")
        embedding = m.EmbeddingModel.from_pyannote(embedding, hf_token)

        # Hyper-parameters and their aliases
        tau = utils.get(data, "tau_active", None)
        if tau is None:
            tau = utils.get(data, "tau", 0.5)
        rho = utils.get(data, "rho_update", None)
        if rho is None:
            rho = utils.get(data, "rho", 0.3)
        delta = utils.get(data, "delta_new", None)
        if delta is None:
            delta = utils.get(data, "delta", 1)

        return SpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            duration=utils.get(data, "duration", None),
            step=utils.get(data, "step", 0.5),
            latency=utils.get(data, "latency", None),
            tau_active=tau,
            rho_update=rho,
            delta_new=delta,
            gamma=utils.get(data, "gamma", 3),
            beta=utils.get(data, "beta", 10),
            max_speakers=utils.get(data, "max_speakers", 20),
            merge_collar=utils.get(data, "merge_collar", 0.05),
            device=device,
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


class SpeakerDiarization(base.StreamingPipeline):
    def __init__(self, config: Optional[SpeakerDiarizationConfig] = None):
        self._config = SpeakerDiarizationConfig() if config is None else config

        msg = f"Latency should be in the range [{self._config.step}, {self._config.duration}]"
        assert self._config.step <= self._config.latency <= self._config.duration, msg

        self.segmentation = blocks.SpeakerSegmentation(self._config.segmentation, self._config.device)
        self.embedding = blocks.OverlapAwareSpeakerEmbedding(
            self._config.embedding, self._config.gamma, self._config.beta, norm=1, device=self._config.device
        )
        self.pred_aggregation = blocks.DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="hamming",
            cropping_mode="loose",
        )
        self.audio_aggregation = blocks.DelayedAggregation(
            self._config.step,
            self._config.latency,
            strategy="first",
            cropping_mode="center",
        )
        self.binarize = blocks.Binarize(self._config.tau_active)

        # Internal state, handle with care
        self.timestamp_shift = 0
        self.clustering = None
        self.chunk_buffer, self.pred_buffer = [], []
        self.reset()

    @staticmethod
    def get_config_class() -> type:
        return SpeakerDiarizationConfig

    @staticmethod
    def hyper_parameters() -> Sequence[HyperParameter]:
        return [TauActive, RhoUpdate, DeltaNew]

    @property
    def config(self) -> SpeakerDiarizationConfig:
        return self._config

    def set_timestamp_shift(self, shift: float):
        self.timestamp_shift = shift

    def join_predictions(self, predictions: List[Annotation]) -> Annotation:
        result = Annotation(uri=predictions[0].uri)
        for pred in predictions:
            result.update(pred)
        return result.support(self.config.merge_collar)

    def write_prediction(self, uri: Text, prediction: Annotation, dir_path: Union[Text, Path]):
        with open(Path(dir_path) / f"{uri}.rttm", "w") as out_file:
            prediction.write_rttm(out_file)

    def suggest_metric(self) -> Metric:
        return DiarizationErrorRate(collar=0, skip_overlap=False)

    def suggest_writer(self, uri: Text, output_dir: Union[Text, Path]) -> Observer:
        return sinks.RTTMWriter(uri, Path(output_dir) / f"{uri}.rttm")

    def suggest_display(self) -> Observer:
        return sinks.StreamingPlot(
            self.config.duration,
            self.config.step,
            self.config.latency,
            self.config.sample_rate
        )

    def reset(self):
        self.set_timestamp_shift(0)
        self.clustering = blocks.IncrementalSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        self.chunk_buffer, self.pred_buffer = [], []

    def __call__(
        self,
        waveforms: Sequence[SlidingWindowFeature],
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
