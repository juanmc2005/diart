from typing import Union, Optional

import numpy as np
import torch
from pyannote.core import SlidingWindow, SlidingWindowFeature

TemporalFeatures = Union[SlidingWindowFeature, np.ndarray, torch.Tensor]


class TemporalFeatureFormatterState:
    """
    Represents the recorded type of a temporal feature formatter.
    Its job is to transform temporal features into tensors and
    recover the original format on other features.
    """
    def to_tensor(self, features: TemporalFeatures) -> torch.Tensor:
        raise NotImplementedError

    def to_internal_type(self, features: torch.Tensor) -> TemporalFeatures:
        """
        Cast `features` to the representing type and remove batch dimension if required.

        Parameters
        ----------
        features: torch.Tensor, shape (batch, frames, dim)
            Batched temporal features.
        Returns
        -------
        new_features: SlidingWindowFeature or numpy.ndarray or torch.Tensor, shape (batch, frames, dim)
        """
        raise NotImplementedError


class SlidingWindowFeatureFormatterState(TemporalFeatureFormatterState):
    def __init__(self, duration: float):
        self.duration = duration
        self._cur_start_time = 0

    def to_tensor(self, features: SlidingWindowFeature) -> torch.Tensor:
        msg = "Features sliding window duration and step must be equal"
        assert features.sliding_window.duration == features.sliding_window.step, msg
        self._cur_start_time = features.sliding_window.start
        return torch.from_numpy(features.data)

    def to_internal_type(self, features: torch.Tensor) -> TemporalFeatures:
        batch_size, num_frames, _ = features.shape
        assert batch_size == 1, "Batched SlidingWindowFeature objects are not supported"
        # Calculate resolution
        resolution = self.duration / num_frames
        # Temporal shift to keep track of current start time
        resolution = SlidingWindow(start=self._cur_start_time, duration=resolution, step=resolution)
        return SlidingWindowFeature(features.squeeze(dim=0).cpu().numpy(), resolution)


class NumpyArrayFormatterState(TemporalFeatureFormatterState):
    def to_tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(features)

    def to_internal_type(self, features: torch.Tensor) -> TemporalFeatures:
        return features.cpu().numpy()


class PytorchTensorFormatterState(TemporalFeatureFormatterState):
    def to_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def to_internal_type(self, features: torch.Tensor) -> TemporalFeatures:
        return features


class TemporalFeatureFormatter:
    """
    Manages the typing and format of temporal features.
    When casting temporal features as torch.Tensor, it remembers its
    type and format so it can lately restore it on other temporal features.
    """
    def __init__(self):
        self.state: Optional[TemporalFeatureFormatterState] = None

    def set_state(self, features: TemporalFeatures):
        if isinstance(features, SlidingWindowFeature):
            msg = "Features sliding window duration and step must be equal"
            assert features.sliding_window.duration == features.sliding_window.step, msg
            self.state = SlidingWindowFeatureFormatterState(
                features.data.shape[0] * features.sliding_window.duration,
            )
        elif isinstance(features, np.ndarray):
            self.state = NumpyArrayFormatterState()
        elif isinstance(features, torch.Tensor):
            self.state = PytorchTensorFormatterState()
        else:
            msg = "Unknown format. Provide one of SlidingWindowFeature, numpy.ndarray, torch.Tensor"
            raise ValueError(msg)

    def cast(self, features: TemporalFeatures) -> torch.Tensor:
        """
        Transform features into a `torch.Tensor` and add batch dimension if missing.

        Parameters
        ----------
        features: SlidingWindowFeature or numpy.ndarray or torch.Tensor
            Shape (frames, dim) or (batch, frames, dim)

        Returns
        -------
        features: torch.Tensor, shape (batch, frames, dim)
        """
        # Set state if not initialized
        self.set_state(features)
        # Convert features to tensor
        data = self.state.to_tensor(features)
        # Make sure there's a batch dimension
        msg = "Temporal features must be 2D or 3D"
        assert data.ndim in (2, 3), msg
        if data.ndim == 2:
            data = data.unsqueeze(0)
        return data.float()

    def restore_type(self, features: torch.Tensor) -> TemporalFeatures:
        """
        Cast `features` to the internal type and remove batch dimension if required.

        Parameters
        ----------
        features: torch.Tensor, shape (batch, frames, dim)
            Batched temporal features.
        Returns
        -------
        new_features: SlidingWindowFeature or numpy.ndarray or torch.Tensor, shape (batch, frames, dim)
        """
        return self.state.to_internal_type(features)
