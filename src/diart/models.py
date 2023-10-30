from abc import ABC, abstractmethod
from typing import Optional, Text, Union, Callable, TYPE_CHECKING

import torch
import torch.nn as nn
from requests import HTTPError

try:
    from pyannote.audio import Inference, Model
    from pyannote.audio.pipelines.speaker_verification import (
        PretrainedSpeakerEmbedding,
    )

    _has_pyannote = True
except ImportError:
    _has_pyannote = False

if TYPE_CHECKING:
    from pyannote.audio import Model
    from pyannote.audio.pipelines.speaker_verification import (
        PretrainedSpeakerEmbedding,
    )


class PyannoteLoader:
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__()
        self.model_info = model_info
        self.hf_token = hf_token

    def __call__(self) -> Union[Model, PretrainedSpeakerEmbedding]:
        try:
            return Model.from_pretrained(self.model_info, use_auth_token=self.hf_token)
        except HTTPError:
            return PretrainedSpeakerEmbedding(
                self.model_info, use_auth_token=self.hf_token
            )


class LazyModel(ABC):
    def __init__(self, loader: Callable[[], Callable]):
        super().__init__()
        self.get_model = loader
        self.model: Optional[nn.Module] = None

    def is_in_memory(self) -> bool:
        """Return whether the model has been loaded into memory"""
        return self.model is not None

    def load(self):
        if not self.is_in_memory():
            self.model = self.get_model()

    def to(self, *args, **kwargs) -> nn.Module:
        self.load()
        return self.model.to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.load()
        return self.model(*args, **kwargs)

    def eval(self) -> "LazyModel":
        self.load()
        if isinstance(self.model, nn.Module):
            self.model.eval()
        return self


class SegmentationModel(LazyModel):
    """
    Minimal interface for a segmentation model.
    """

    @staticmethod
    def from_pyannote(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "SegmentationModel":
        """
        Returns a `SegmentationModel` wrapping a pyannote model.

        Parameters
        ----------
        model: pyannote.PipelineModel
            The pyannote.audio model to fetch.
        use_hf_token: str | bool, optional
            The Huggingface access token to use when downloading the model.
            If True, use huggingface-cli login token.
            Defaults to None.

        Returns
        -------
        wrapper: SegmentationModel
        """
        assert _has_pyannote, "No pyannote.audio installation found"
        return PyannoteSegmentationModel(model, use_hf_token)

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        pass


class PyannoteSegmentationModel(SegmentationModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__(PyannoteLoader(model_info, hf_token))

    @property
    def sample_rate(self) -> int:
        self.load()
        return self.model.audio.sample_rate

    @property
    def duration(self) -> float:
        self.load()
        return self.model.specifications.duration

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return super().__call__(waveform)


class EmbeddingModel(LazyModel):
    """Minimal interface for an embedding model."""

    @staticmethod
    def from_pyannote(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "EmbeddingModel":
        """
        Returns an `EmbeddingModel` wrapping a pyannote model.

        Parameters
        ----------
        model: pyannote.PipelineModel
            The pyannote.audio model to fetch.
        use_hf_token: str | bool, optional
            The Huggingface access token to use when downloading the model.
            If True, use huggingface-cli login token.
            Defaults to None.

        Returns
        -------
        wrapper: EmbeddingModel
        """
        assert _has_pyannote, "No pyannote.audio installation found"
        return PyannoteEmbeddingModel(model, use_hf_token)


class PyannoteEmbeddingModel(EmbeddingModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__(PyannoteLoader(model_info, hf_token))

    def __call__(
        self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Normalize weights
        if weights is not None:
            min_values = weights.min(dim=1, keepdim=True).values
            max_values = weights.max(dim=1, keepdim=True).values
            weights = (weights - min_values) / (max_values - min_values)
            weights.nan_to_num_(0.0)

        if isinstance(self.model, nn.Module):
            return super().__call__(waveform, weights)

        else:
            if weights is not None:
                # Move to cpu for numpy conversion
                weights = weights.to("cpu")
            # Move to cpu for numpy conversion
            waveform = waveform.to("cpu")
            return torch.from_numpy(super().__call__(waveform, weights))
