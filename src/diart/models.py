from typing import Optional, Text, Union

import torch
import torch.nn as nn

try:
    import pyannote.audio.pipelines.utils as pyannote_loader
    _has_pyannote = True
except ImportError:
    _has_pyannote = False


class LazyModel(nn.Module):
    @property
    def model(self) -> Optional[nn.Module]:
        raise NotImplementedError

    def load(self):
        """Load model to memory"""
        raise NotImplementedError

    def is_in_memory(self) -> bool:
        """Return whether the model has been loaded into memory"""
        return self.model is not None

    def to(self, *args, **kwargs) -> nn.Module:
        if not self.in_memory():
            self.load_model()
        return super().to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if not self.is_in_memory():
            self.load()
        return super().__call__(*args, **kwargs)


class PyannoteModel(LazyModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__()
        self.model_info = model_info
        self.hf_token = hf_token
        self._model: Optional[nn.Module] = None

    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    def load(self):
        """Load model to memory"""
        if not self.is_in_memory():
            self._model = pyannote_loader.get_model(self.model_info, self.hf_token)


class PyannoteSegmentationModel(PyannoteModel):
    def get_sample_rate(self) -> int:
        self.load()
        return self.model.audio.sample_rate

    def get_duration(self) -> float:
        self.load()
        return self.model.specifications.duration

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform)


class PyannoteEmbeddingModel(PyannoteModel):
    def forward(
        self,
        waveform: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(waveform, weights=weights)


class SegmentationModel(nn.Module):
    """
    Minimal interface for a segmentation model.
    """
    def __init__(self, model: LazyModel):
        super().__init__()
        self.model = model

    @staticmethod
    def from_pyannote(model, use_hf_token: Union[Text, bool, None] = True) -> 'SegmentationModel':
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
        return SegmentationModel(PyannoteSegmentationModel(model, use_hf_token))

    def get_sample_rate(self) -> int:
        return self.model.get_sample_rate()

    def get_duration(self) -> float:
        return self.model.get_duration()

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
        return self.model(waveform)

    def to(self, *args, **kwargs) -> nn.Module:
        self.model.load()
        return super().to(*args, **kwargs)


class EmbeddingModel(nn.Module):
    """Minimal interface for an embedding model."""
    def __init__(self, model: LazyModel):
        super().__init__()
        self.model = model

    @staticmethod
    def from_pyannote(model, use_hf_token: Union[Text, bool, None] = True) -> 'EmbeddingModel':
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
        return EmbeddingModel(PyannoteEmbeddingModel(model, use_hf_token))

    def forward(
        self,
        waveform: torch.Tensor,
        weights: Optional[torch.Tensor] = None
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
        return self.model(waveform, weights)

    def to(self, *args, **kwargs) -> nn.Module:
        self.model.load()
        return super().to(*args, **kwargs)
