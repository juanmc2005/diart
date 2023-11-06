from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Text, Union, Callable, List, Tuple

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import yaml
from requests import HTTPError

try:
    from pyannote.audio import Model
    from pyannote.audio.pipelines.speaker_verification import (
        PretrainedSpeakerEmbedding,
    )
    from pyannote.audio.utils.powerset import Powerset

    _has_pyannote = True
except ImportError:
    _has_pyannote = False


class PowersetAdapter(nn.Module):
    def __init__(self, segmentation_model: nn.Module):
        super().__init__()
        self.model = segmentation_model
        specs = self.model.specifications
        max_speakers_per_frame = specs.powerset_max_classes
        max_speakers_per_chunk = len(specs.classes)
        self.powerset = Powerset(max_speakers_per_chunk, max_speakers_per_frame)

    @property
    def specifications(self):
        return self.model.specifications

    @property
    def audio(self):
        return self.model.audio

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.powerset.to_multilabel(self.model(waveform), soft=False)


class PyannoteLoader:
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__()
        self.model_info = model_info
        self.hf_token = hf_token

    def __call__(self) -> Callable:
        try:
            model = Model.from_pretrained(self.model_info, use_auth_token=self.hf_token)
            specs = getattr(model, "specifications", None)
            if specs is not None and specs.powerset:
                model = PowersetAdapter(model)
            return model
        except HTTPError:
            return PretrainedSpeakerEmbedding(
                self.model_info, use_auth_token=self.hf_token
            )


class ONNXLoader:
    def __init__(self, path: Path, input_names: List[str], output_names: List[str]):
        super().__init__()
        self.path = path
        self.input_names = input_names
        self.output_names = output_names

    def __call__(self) -> ONNXModel:
        return ONNXModel(self.path, self.input_names, self.output_names)


class ONNXModel:
    def __init__(self, path: Path, input_names: List[str], output_names: List[str]):
        super().__init__()
        self.path = path
        self.input_names = input_names
        self.output_names = output_names
        self.device = torch.device("cpu")
        self.session = None
        self.recreate_session()

    @property
    def execution_provider(self) -> str:
        device = "CUDA" if self.device.type == "cuda" else "CPU"
        return f"{device}ExecutionProvider"

    def recreate_session(self):
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = onnxruntime.InferenceSession(
            self.path,
            sess_options=options,
            providers=[self.execution_provider],
        )

    def to(self, device: torch.device) -> ONNXModel:
        if device.type != self.device.type:
            self.device = device
            self.recreate_session()
        return self

    def __call__(self, *args) -> Tuple[torch.Tensor, ...]:
        inputs = {
            name: arg.cpu().numpy().astype(np.float32)
            for name, arg in zip(self.input_names, args)
        }
        outputs = self.session.run(self.output_names, inputs)
        return tuple(
            torch.from_numpy(out).float().to(args[0].device) for out in outputs
        )


class LazyModel(ABC):
    def __init__(self, loader: Callable[[], Callable]):
        super().__init__()
        self.get_model = loader
        self.model: Optional[Callable] = None

    def is_in_memory(self) -> bool:
        """Return whether the model has been loaded into memory"""
        return self.model is not None

    def load(self):
        if not self.is_in_memory():
            self.model = self.get_model()

    def to(self, device: torch.device) -> LazyModel:
        self.load()
        self.model = self.model.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.load()
        return self.model(*args, **kwargs)

    def eval(self) -> LazyModel:
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

    @staticmethod
    def from_onnx(model_path: Union[str, Path]) -> "SegmentationModel":
        return ONNXSegmentationModel(model_path)

    @staticmethod
    def from_pretrained(model, use_hf_token: Union[Text, bool, None] = True) -> "SegmentationModel":
        if isinstance(model, str) or isinstance(model, Path):
            if Path(model).name.endswith(".onnx"):
                return SegmentationModel.from_onnx(model)
        return SegmentationModel.from_pyannote(model, use_hf_token)

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        pass

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Call the forward pass of the segmentation model.
        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        Returns
        -------
        speaker_segmentation: torch.Tensor, shape (batch, frames, speakers)
        """
        return super().__call__(waveform)


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


class ONNXSegmentationModel(SegmentationModel):
    def __init__(self, model_path: Union[str, Path]):
        model_path = Path(model_path)
        loader = ONNXLoader(model_path, input_names=["waveform"], output_names=["segmentation"])
        super().__init__(loader)
        with open(model_path.parent / f"{model_path.stem}.yml", "r") as metadata_file:
            metadata = yaml.load(metadata_file, yaml.SafeLoader)
        self.metadata = metadata

    @property
    def sample_rate(self) -> int:
        return self.metadata["sample_rate"]

    @property
    def duration(self) -> float:
        return self.metadata["duration"]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # ONNX models may have multiple return values, so this will be a list of size 1
        return super().__call__(waveform)[0]


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

    @staticmethod
    def from_onnx(model_path: Union[str, Path]) -> "EmbeddingModel":
        return ONNXEmbeddingModel(model_path)

    @staticmethod
    def from_pretrained(model, use_hf_token: Union[Text, bool, None] = True) -> "EmbeddingModel":
        if isinstance(model, str) or isinstance(model, Path):
            if Path(model).name.endswith(".onnx"):
                return EmbeddingModel.from_onnx(model)
        return EmbeddingModel.from_pyannote(model, use_hf_token)

    def __call__(
        self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Call the forward pass of an embedding model with optional weights.
        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        weights: Optional[torch.Tensor], shape (batch, frames)
            Temporal weights for each sample in the batch. Defaults to no weights.
        Returns
        -------
        speaker_embeddings: torch.Tensor, shape (batch, embedding_dim)
        """
        embeddings = super().__call__(waveform, weights)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        return embeddings


class PyannoteEmbeddingModel(EmbeddingModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__(PyannoteLoader(model_info, hf_token))


class ONNXEmbeddingModel(EmbeddingModel):
    def __init__(self, model_path: Union[str, Path]):
        loader = ONNXLoader(Path(model_path), input_names=["waveform", "weights"], output_names=["embedding"])
        super().__init__(loader)

    def __call__(
        self, waveform: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.load()
        # ONNX models may have multiple return values, so this will be a list of size 1
        embeddings = self.model(waveform, weights)[0]
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        return embeddings
