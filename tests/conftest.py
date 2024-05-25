import random

import pytest
import torch

from diart.models import SegmentationModel, EmbeddingModel


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
