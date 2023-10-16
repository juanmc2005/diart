from abc import ABC, abstractmethod
from typing import Optional, Text, Union, Callable, Mapping, TYPE_CHECKING

import torch
import torch.nn as nn

try:
    import pyannote.audio.pipelines.utils as pyannote_loader
    from pyannote.audio import Inference, Model
    from pyannote.audio.pipelines.speaker_verification import WeSpeakerPretrainedSpeakerEmbedding

    _has_pyannote = True
except ImportError:
    _has_pyannote = False

if TYPE_CHECKING:
    PipelineInference = Union[WeSpeakerPretrainedSpeakerEmbedding, Model, Text, Mapping]


class PyannoteLoader:
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__()
        self.model_info = model_info
        self.hf_token = hf_token

    def __call__(self) -> nn.Module:
        return pyannote_loader.get_model(self.model_info, self.hf_token)


class PyannoteWeSpeakerSpeakerEmbeddingLoader:
    def __init__(self, inference_info):
        super().__init__()
        self.inference_info = inference_info

    def __call__(self) -> WeSpeakerPretrainedSpeakerEmbedding:
        return WeSpeakerPretrainedSpeakerEmbedding(self.inference_info)


class LazyModel(nn.Module, ABC):
    def __init__(self, loader: Callable[[], nn.Module]):
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
        return super().to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.load()
        return super().__call__(*args, **kwargs)


class LazyWeSpeakerSpeakerEmbedding(WeSpeakerPretrainedSpeakerEmbedding, ABC):
    def __init__(self, loader: Callable[[], WeSpeakerPretrainedSpeakerEmbedding]):
        self.get_inference = loader
        self.inference: Optional[WeSpeakerPretrainedSpeakerEmbedding] = None
        # __init__ at end because in there we call .to() which requires .load()
        super().__init__()

    def is_in_memory(self) -> bool:
        """Return whether the model has been loaded into memory"""
        return self.inference is not None

    def load(self):
        if not self.is_in_memory():
            self.inference = self.get_inference()

    def to(self, *args, **kwargs) -> WeSpeakerPretrainedSpeakerEmbedding:
        self.load()
        return super().to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.load()
        return super().__call__(*args, **kwargs)


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

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the segmentation model.

        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)

        Returns
        -------
        speaker_segmentation: torch.Tensor, shape (batch, frames, speakers)
        """
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

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform)


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

    @abstractmethod
    def forward(
            self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of an embedding model with optional weights.

        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        weights: Optional[torch.Tensor], shape (batch, frames)
            Temporal weights for each sample in the batch. Defaults to no weights.

        Returns
        -------
        speaker_embeddings: torch.Tensor, shape (batch, embedding_dim)
        """
        pass


class PyannoteEmbeddingModel(EmbeddingModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__(PyannoteLoader(model_info, hf_token))

    def forward(
            self,
            waveform: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(waveform, weights=weights)


class WeSpeakerSpeakerEmbeddingInference(LazyWeSpeakerSpeakerEmbedding):
    """Minimal interface for a we speaker embedding inference."""

    def __init__(self, loader: Callable[[], WeSpeakerPretrainedSpeakerEmbedding]):
        super().__init__(loader)

    @staticmethod
    def from_pyannote(inference,
                      ) -> "WeSpeakerSpeakerEmbeddingInference":
        """
        Returns an `EmbeddingModel` wrapping a pyannote model.

        Parameters
        ----------
        inference: pyannote.audio.pipelines.speaker_verification.WeSpeakerPretrainedSpeakerEmbedding
            The pyannote.audio inference to fetch.

        Returns
        -------
        wrapper: EmbeddingModel
        """
        assert _has_pyannote, "No pyannote.audio installation found"
        return PyannoteWeSpeakerSpeakerEmbeddingInference(inference)

    @abstractmethod
    def forward(
            self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of an embedding model with optional weights.

        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        weights: Optional[torch.Tensor], shape (batch, frames)
            Temporal weights for each sample in the batch. Defaults to no weights.

        Returns
        -------
        speaker_embeddings: torch.Tensor, shape (batch, embedding_dim)
        """
        pass


class PyannoteWeSpeakerSpeakerEmbeddingInference(WeSpeakerSpeakerEmbeddingInference):
    def __init__(self, wespeaker_info):
        super().__init__(PyannoteWeSpeakerSpeakerEmbeddingLoader(wespeaker_info))

    def forward(
            self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.load()
        return torch.from_numpy(self.inference(waveform, weights))
