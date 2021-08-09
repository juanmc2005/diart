import torch
import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio import Model
from typing import Union, Text, Tuple, Callable, Optional, List
from pathlib import Path


def MakePair(function: Callable) -> Callable:
    def apply(arg) -> Tuple:
        return arg, function(arg)
    return apply


def MapArgument(argpos: int, function: Callable) -> Callable:
    def apply(*args) -> Tuple:
        args = list(args)
        args[argpos] = function(args[argpos])
        return tuple(args)
    return apply


def MapArgumentAppend(argpos: int, function: Callable) -> Callable:
    def apply(*args) -> Tuple:
        args = list(args)
        args.append(function(args[argpos]))
        return tuple(args)
    return apply


def MapArguments(argpos: List[int], function: Callable) -> Callable:
    def apply(*args) -> Tuple:
        fargs = [arg for i, arg in enumerate(args) if i in argpos]
        remaining = [arg for i, arg in enumerate(args) if i not in argpos]
        return tuple(remaining + [function(*fargs)])
    return apply


def MapArgumentsAppend(argpos: List[int], function: Callable) -> Callable:
    def apply(*args) -> Tuple:
        fargs = [arg for i, arg in enumerate(args) if i in argpos]
        return tuple(list(args) + [function(*fargs)])
    return apply


def KeepArguments(argpos: List[int]) -> Callable:
    def apply(*args):
        kept = [arg for i, arg in enumerate(args) if i in argpos]
        if len(kept) == 1:
            return kept[0]
        return tuple(kept)
    return apply


class FrameWiseModel:
    def __init__(self, model: Union[Path, Text]):
        self.model = Model.from_pretrained(model)
        self.model.eval()

    def __call__(self, waveform: SlidingWindowFeature) -> SlidingWindowFeature:
        with torch.no_grad():
            wave = waveform.data.T[np.newaxis]
            output = self.model(torch.from_numpy(wave)).numpy()[0]
        # Temporal resolution of the output
        resolution = self.model.introspection.frames
        # Temporal shift to keep track of current start time
        resolution = SlidingWindow(start=waveform.sliding_window.start,
                                   duration=resolution.duration,
                                   step=resolution.step)
        return SlidingWindowFeature(output, resolution)


def OverlappedSpeechPenalty(gamma: float = 3, beta: float = 10) -> Callable:
    """
    :param gamma: float, optional
        Exponent to sharpen per-frame speaker probability scores and distributions.
        Defaults to 3.
    :param beta: float, optional
        Softmax's temperature parameter (actually 1/beta) to sharpen per-frame speaker probability distributions.
        Defaults to 10.
    """
    def apply(segmentation: SlidingWindowFeature) -> SlidingWindowFeature:
        weights = torch.from_numpy(segmentation.data).float().T
        with torch.no_grad():
            probs = torch.softmax(beta * weights, dim=0)
            weights = torch.pow(weights, gamma) * torch.pow(probs, gamma)
            weights[weights < 1e-8] = 1e-8
        return SlidingWindowFeature(weights.T.numpy(), segmentation.sliding_window)

    return apply


def Embedding(model: Union[Path, Text]) -> Callable:
    model = Model.from_pretrained(model)
    model.eval()

    def apply(waveform: SlidingWindowFeature, weights: Optional[SlidingWindowFeature]) -> torch.Tensor:
        with torch.no_grad():
            chunk = torch.from_numpy(waveform.data.T).float()
            inputs = chunk.unsqueeze(0).to(model.device)
            if weights is not None:
                # weights has shape (num_local_speakers, num_frames)
                weights = torch.from_numpy(weights.data.T).float()
                inputs = inputs.repeat(weights.shape[0], 1, 1)
            # Shape (num_speakers, emb_dimension)
            embeddings = model(inputs, weights=weights)
        return embeddings

    return apply


def EmbeddingNormalization(norm: Union[float, torch.Tensor] = 1) -> Callable:

    def apply(embeddings: torch.Tensor):
        if isinstance(norm, torch.Tensor):
            assert norm.shape[0] == embeddings.shape[0]
        with torch.no_grad():
            norm_embs = norm * embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return norm_embs

    return apply


def ActiveSpeakers(threshold: float = 0.5) -> Callable:
    def apply(segmentation: SlidingWindowFeature) -> np.ndarray:
        return np.where(np.max(segmentation.data, axis=0) >= threshold)[0]
    return apply


def LongSpeechSpeakers(threshold: float = 0.5) -> Callable:
    def apply(segmentation: SlidingWindowFeature) -> np.ndarray:
        return np.where(np.mean(segmentation.data, axis=0) >= threshold)[0]
    return apply
