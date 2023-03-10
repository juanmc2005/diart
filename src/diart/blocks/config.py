from typing import Any, Optional, Union

import torch
from typing_extensions import Literal

from .. import models as m
from .. import utils


class BasePipelineConfig:
    @property
    def duration(self) -> float:
        raise NotImplementedError

    @property
    def step(self) -> float:
        raise NotImplementedError

    @property
    def latency(self) -> float:
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Any) -> 'BasePipelineConfig':
        raise NotImplementedError


class PipelineConfig(BasePipelineConfig):
    def __init__(
        self,
        segmentation: Optional[m.SegmentationModel] = None,
        embedding: Optional[m.EmbeddingModel] = None,
        duration: Optional[float] = None,
        step: float = 0.5,
        latency: Optional[Union[float, Literal["max", "min"]]] = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation
        if self.segmentation is None:
            self.segmentation = m.SegmentationModel.from_pyannote("pyannote/segmentation")

        # Default duration is the one given by the segmentation model
        self._duration = duration

        # Expected sample rate is given by the segmentation model
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

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def from_dict(data: Any) -> 'PipelineConfig':
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
            tau = utils.get(data, "tau", 0.6)
        rho = utils.get(data, "rho_update", None)
        if rho is None:
            rho = utils.get(data, "rho", 0.3)
        delta = utils.get(data, "delta_new", None)
        if delta is None:
            delta = utils.get(data, "delta", 1)

        return PipelineConfig(
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
            device=device,
        )

    @property
    def duration(self) -> float:
        if self._duration is None:
            self._duration = self.segmentation.get_duration()
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
            self._sample_rate = self.segmentation.get_sample_rate()
        return self._sample_rate
