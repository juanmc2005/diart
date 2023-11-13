from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, Sequence, Text

from pyannote.core import SlidingWindowFeature
from pyannote.metrics.base import BaseMetric

from .. import utils
from ..audio import FilePath, AudioLoader


@dataclass
class HyperParameter:
    """Represents a pipeline hyper-parameter that can be tuned by diart"""

    name: Text
    """Name of the hyper-parameter (e.g. tau_active)"""
    low: float
    """Lowest value that this parameter can take"""
    high: float
    """Highest value that this parameter can take"""

    @staticmethod
    def from_name(name: Text) -> "HyperParameter":
        """Create a HyperParameter object given its name.

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        Returns
        -------
        HyperParameter
        """
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
    """Configuration containing the required
    parameters to build and run a pipeline"""

    @property
    @abstractmethod
    def duration(self) -> float:
        """The duration of an input audio chunk (in seconds)"""
        pass

    @property
    @abstractmethod
    def step(self) -> float:
        """The step between two consecutive input audio chunks (in seconds)"""
        pass

    @property
    @abstractmethod
    def latency(self) -> float:
        """The algorithmic latency of the pipeline (in seconds).
        At time `t` of the audio stream, the pipeline will
        output predictions for time `t - latency`.
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """The sample rate of the input audio stream"""
        pass

    def get_file_padding(self, filepath: FilePath) -> Tuple[float, float]:
        file_duration = AudioLoader(self.sample_rate, mono=True).get_duration(filepath)
        right = utils.get_padding_right(self.latency, self.step)
        left = utils.get_padding_left(file_duration + right, self.duration)
        return left, right


class Pipeline(ABC):
    """Represents a streaming audio pipeline"""

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
        """Runs the next steps of the pipeline
        given a list of consecutive audio chunks.

        Parameters
        ----------
        waveforms: Sequence[SlidingWindowFeature]
            Consecutive chunk waveforms for the pipeline to ingest

        Returns
        -------
        Sequence[Tuple[Any, SlidingWindowFeature]]
            For each input waveform, a tuple containing
            the pipeline output and its respective audio
        """
        pass
