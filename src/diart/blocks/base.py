from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Sequence, Text

from pyannote.core import SlidingWindowFeature
from pyannote.metrics.base import BaseMetric

from .. import utils
from ..audio import FilePath, AudioLoader


@dataclass
class HyperParameter:
    name: Text
    low: float
    high: float

    @staticmethod
    def from_name(name: Text) -> "HyperParameter":
        if name == "tau_active":
            return TauActive
        if name == "rho_update":
            return RhoUpdate
        if name == "delta_new":
            return DeltaNew
        raise ValueError(f"Hyper-parameter '{name}' not recognized")


TauActive = HyperParameter("tau_active", low=0, high=1)
RhoUpdate = HyperParameter("rho_update", low=0, high=1)
DeltaNew = HyperParameter("delta_new", low=0, high=2)


class PipelineConfig(ABC):
    @property
    @abstractmethod
    def duration(self) -> float:
        pass

    @property
    @abstractmethod
    def step(self) -> float:
        pass

    @property
    @abstractmethod
    def latency(self) -> float:
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    def get_file_padding(self, filepath: FilePath) -> Tuple[float, float]:
        file_duration = AudioLoader(self.sample_rate, mono=True).get_duration(filepath)
        right = utils.get_padding_right(self.latency, self.step)
        left = utils.get_padding_left(file_duration + right, self.duration)
        return left, right


class Pipeline(ABC):
    @staticmethod
    @abstractmethod
    def get_config_class() -> type:
        pass

    @staticmethod
    @abstractmethod
    def suggest_metric() -> BaseMetric:
        pass

    @staticmethod
    @abstractmethod
    def hyper_parameters() -> Sequence[HyperParameter]:
        pass

    @property
    @abstractmethod
    def config(self) -> PipelineConfig:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def set_timestamp_shift(self, shift: float):
        pass

    @abstractmethod
    def __call__(
        self, waveforms: Sequence[SlidingWindowFeature]
    ) -> Sequence[Tuple[Any, SlidingWindowFeature]]:
        pass
