import rx
from rx import operators as ops
from rx.core import Observable
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature


Operator = Callable[[Observable], Observable]


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
) -> Operator:
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
