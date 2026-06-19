import random
from pathlib import Path

import pytest
import torch

from diart import SpeakerDiarizationConfig
from diart.models import EmbeddingModel, SegmentationModel

MODEL_DIR = Path(__file__).parent.parent / "assets" / "models"
DATA_DIR = Path(__file__).parent / "data"


class DummySegmentationModel:
    def to(self, device):
        pass

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 3

        batch_size, num_channels, num_samples = waveform.shape
        num_frames = random.randint(250, 500)
        num_speakers = random.randint(3, 5)

        return torch.rand(batch_size, num_frames, num_speakers)


class DummyEmbeddingModel:
    def to(self, device):
        pass

    def __call__(self, waveform: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 3
        assert weights.ndim == 2

        batch_size, num_channels, num_samples = waveform.shape
        batch_size_weights, num_frames = weights.shape

        assert batch_size == batch_size_weights

        embedding_dim = random.randint(128, 512)

        return torch.randn(batch_size, embedding_dim)


@pytest.fixture(scope="session")
def segmentation_model() -> SegmentationModel:
    return SegmentationModel(DummySegmentationModel)


@pytest.fixture(scope="session")
def embedding_model() -> EmbeddingModel:
    return EmbeddingModel(DummyEmbeddingModel)


@pytest.fixture(scope="session")
def segmentation() -> SegmentationModel:
    return SegmentationModel.from_pretrained(MODEL_DIR / "segmentation_uint8.onnx")


@pytest.fixture(scope="session")
def embedding() -> EmbeddingModel:
    return EmbeddingModel.from_pretrained(MODEL_DIR / "embedding_uint8.onnx")


@pytest.fixture(scope="session")
def make_config(segmentation, embedding):
    def _config(latency):
        return SpeakerDiarizationConfig(
            segmentation=segmentation,
            embedding=embedding,
            step=0.5,
            latency=latency,
            tau_active=0.507,
            rho_update=0.006,
            delta_new=1.057,
        )

    return _config
