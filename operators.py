import rx
from rx import operators as ops
from rx.core import Observable
from dataclasses import dataclass
from typing import Callable, Optional, List
import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.audio.core.online import SpeakerMap
import functional as fn
from clustering import OnlineSpeakerClustering


Operator = Callable[[Observable], Observable]


def make_pair(function: Callable) -> Operator:
    return ops.map(fn.MakePair(function))


def map_arg(argpos: int, function: Callable) -> Operator:
    return ops.starmap(fn.MapArgument(argpos, function))


def map_arg_append(argpos: int, function: Callable) -> Operator:
    return ops.starmap(fn.MapArgumentAppend(argpos, function))


def map_many_args(argpos: List[int], function: Callable) -> Operator:
    return ops.starmap(fn.MapArguments(argpos, function))


def map_many_args_append(argpos: List[int], function: Callable) -> Operator:
    return ops.starmap(fn.MapArgumentsAppend(argpos, function))


def keep_args(argpos: List[int]) -> Operator:
    return ops.starmap(fn.KeepArguments(argpos))


@dataclass
class AudioBufferState:
    chunk: Optional[np.ndarray]
    buffer: Optional[np.ndarray]
    start_time: float
    changed: bool

    @staticmethod
    def initial():
        return AudioBufferState(None, None, 0, False)

    @staticmethod
    def has_samples(num_samples: int):
        def call_fn(state) -> bool:
            return state.chunk is not None and state.chunk.shape[1] == num_samples
        return call_fn

    @staticmethod
    def to_sliding_window(sample_rate: int):
        def call_fn(state) -> SlidingWindowFeature:
            resolution = SlidingWindow(start=state.start_time, duration=1. / sample_rate, step=1. / sample_rate)
            return SlidingWindowFeature(state.chunk.T, resolution)
        return call_fn


def RegularizeStream(
    duration: float = 5,
    step: float = 0.5,
    sample_rate: int = 16000
) -> Callable[[Observable], Observable]:
    chunk_samples = int(round(sample_rate * duration))
    step_samples = int(round(sample_rate * step))

    def accumulate(state: AudioBufferState, value: np.ndarray):
        # state contains the last emitted chunk, the current step buffer and the last starting time
        assert value.ndim == 2 and value.shape[0] == 1, "Waveform must have shape (1, samples)"

        start_time = state.start_time
        buffer = value if state.buffer is None else np.concatenate([state.buffer, value], axis=1)
        if buffer.shape[1] >= step_samples:
            if buffer.shape[1] == step_samples:
                new_chunk, new_buffer = buffer, None
            else:
                new_chunk = buffer[:, :step_samples]
                new_buffer = buffer[:, step_samples:]

            if state.chunk is not None:
                new_chunk = np.concatenate([state.chunk, new_chunk], axis=1)

            if new_chunk.shape[1] > chunk_samples:
                new_chunk = new_chunk[:, -chunk_samples:]
                start_time += step

            return AudioBufferState(new_chunk, new_buffer, start_time, True)

        return AudioBufferState(state.chunk, buffer, start_time, False)

    return rx.pipe(
        # Accumulate last <=5s waveform as an AudioBufferState
        ops.scan(accumulate, AudioBufferState.initial()),
        # Take only states that have 5s duration and whose chunk has changed since last time
        ops.filter(AudioBufferState.has_samples(chunk_samples)),
        ops.filter(lambda state: state.changed),
        # Transform state into a SlidingWindowFeature containing the new 5s chunk
        ops.map(AudioBufferState.to_sliding_window(sample_rate))
    )


def ClusterSpeakers(
    tau_active: float,
    rho_update: float,
    delta_new: float,
    k_max_speakers: int,
    metric: Optional[str] = "cosine",
    max_speakers: int = 20
) -> Callable[[Observable], Observable]:
    model = OnlineSpeakerClustering(delta_new, k_max_speakers, metric, max_speakers)

    def apply_mapping(segmentation: SlidingWindowFeature, mapping: SpeakerMap) -> SlidingWindowFeature:
        return SlidingWindowFeature(mapping.apply(segmentation.data), segmentation.sliding_window)

    return rx.pipe(
        map_arg_append(0, fn.ActiveSpeakers(threshold=tau_active)),  # (segmentation, embeddings, active)
        map_arg_append(0, fn.LongSpeechSpeakers(threshold=rho_update)),  # (segmentation, embeddings, active, long)
        map_many_args_append(list(range(1, 4)), model.identify),  # (segmentation, embeddings, active, long, mapping)
        map_many_args([0, 4], apply_mapping),
        # (embeddings, active, long, permutation)
        keep_args([3])
    )
