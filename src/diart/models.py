from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Optional, Text, Union, Callable, List

import numpy as np
import torch
import torch.nn as nn
from requests import HTTPError

try:
    from pyannote.audio import Model
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    from pyannote.audio.utils.powerset import Powerset

    IS_PYANNOTE_AVAILABLE = True
except ImportError:
    IS_PYANNOTE_AVAILABLE = False

try:
    import onnxruntime as ort

    IS_ONNX_AVAILABLE = True
except ImportError:
    IS_ONNX_AVAILABLE = False


class PowersetAdapter(nn.Module):
    def __init__(self, segmentation_model: nn.Module):
        super().__init__()
        self.model = segmentation_model
        specs = self.model.specifications
        max_speakers_per_frame = specs.powerset_max_classes
        max_speakers_per_chunk = len(specs.classes)
        self.powerset = Powerset(max_speakers_per_chunk, max_speakers_per_frame)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.powerset.to_multilabel(self.model(waveform))


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
            pass
        except ModuleNotFoundError:
            pass
        return PretrainedSpeakerEmbedding(self.model_info, use_auth_token=self.hf_token)


class ONNXLoader:
    def __init__(self, path: str | Path, input_names: List[str], output_name: str):
        super().__init__()
        self.path = Path(path)
        self.input_names = input_names
        self.output_name = output_name

    def __call__(self) -> ONNXModel:
        return ONNXModel(self.path, self.input_names, self.output_name)


class ONNXModel:
    def __init__(self, path: Path, input_names: List[str], output_name: str):
        super().__init__()
        self.path = path
        self.input_names = input_names
        self.output_name = output_name
        self.device = torch.device("cpu")
        self.session = None
        self.recreate_session()

    @property
    def execution_provider(self) -> str:
        device = "CUDA" if self.device.type == "cuda" else "CPU"
        return f"{device}ExecutionProvider"

    def recreate_session(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.path,
            sess_options=options,
            providers=[self.execution_provider],
        )

    def to(self, device: torch.device) -> ONNXModel:
        if device.type != self.device.type:
            self.device = device
            self.recreate_session()
        return self

    def __call__(self, *args) -> torch.Tensor:
        inputs = {
            name: arg.cpu().numpy().astype(np.float32)
            for name, arg in zip(self.input_names, args)
        }
        output = self.session.run([self.output_name], inputs)[0]
        return torch.from_numpy(output).float().to(args[0].device)


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
        assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
        return SegmentationModel(PyannoteLoader(model, use_hf_token))

    @staticmethod
    def from_onnx(
        model_path: Union[str, Path],
        input_name: str = "waveform",
        output_name: str = "segmentation",
    ) -> "SegmentationModel":
        assert IS_ONNX_AVAILABLE, "No ONNX installation found"
        return SegmentationModel(ONNXLoader(model_path, [input_name], output_name))

    @staticmethod
    def from_pretrained(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "SegmentationModel":
        if isinstance(model, str) or isinstance(model, Path):
            if Path(model).name.endswith(".onnx"):
                return SegmentationModel.from_onnx(model)
        return SegmentationModel.from_pyannote(model, use_hf_token)

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
        assert IS_PYANNOTE_AVAILABLE, "No pyannote.audio installation found"
        loader = PyannoteLoader(model, use_hf_token)
        return EmbeddingModel(loader)

    @staticmethod
    def from_onnx(
        model_path: Union[str, Path],
        input_names: List[str] | None = None,
        output_name: str = "embedding",
    ) -> "EmbeddingModel":
        assert IS_ONNX_AVAILABLE, "No ONNX installation found"
        input_names = input_names or ["waveform", "weights"]
        loader = ONNXLoader(model_path, input_names, output_name)
        return EmbeddingModel(loader)

    @staticmethod
    def from_pretrained(
        model, use_hf_token: Union[Text, bool, None] = True
    ) -> "EmbeddingModel":
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
