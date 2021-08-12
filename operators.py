import rx
from rx import operators as ops
from rx.core import Observable
from dataclasses import dataclass
from typing import Callable, Optional, List, Literal
import numpy as np
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
import warnings


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


def regularize_stream(
    duration: float = 5,
    step: float = 0.5,
    sample_rate: int = 16000
) -> Operator:
    chunk_samples = int(round(sample_rate * duration))
    step_samples = int(round(sample_rate * step))

    def accumulate(state: AudioBufferState, value: np.ndarray):
        # State contains the last emitted chunk, the current step buffer and the last starting time
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"Waveform must have shape (1, samples) but {value.shape} was found")
        start_time = state.start_time

        # Add new samples to the buffer
        buffer = value if state.buffer is None else np.concatenate([state.buffer, value], axis=1)

        # Check for buffer overflow
        if buffer.shape[1] >= step_samples:
            # Pop samples from buffer
            if buffer.shape[1] == step_samples:
                new_chunk, new_buffer = buffer, None
            else:
                new_chunk = buffer[:, :step_samples]
                new_buffer = buffer[:, step_samples:]

            # Add samples to next chunk
            if state.chunk is not None:
                new_chunk = np.concatenate([state.chunk, new_chunk], axis=1)

            # Truncate chunk to ensure a fixed duration
            if new_chunk.shape[1] > chunk_samples:
                new_chunk = new_chunk[:, -chunk_samples:]
                start_time += step

            # Chunk has changed because of buffer overflow
            return AudioBufferState(new_chunk, new_buffer, start_time, changed=True)

        # Chunk has not changed
        return AudioBufferState(state.chunk, buffer, start_time, changed=False)

    return rx.pipe(
        # Accumulate last <=duration seconds of waveform as an AudioBufferState
        ops.scan(accumulate, AudioBufferState.initial()),
        # Take only states that have the desired duration and whose chunk has changed since last time
        ops.filter(AudioBufferState.has_samples(chunk_samples)),
        ops.filter(lambda state: state.changed),
        # Transform state into a SlidingWindowFeature containing the new chunk
        ops.map(AudioBufferState.to_sliding_window(sample_rate))
    )


def aggregate(
    duration: float,
    step: float,
    latency: Optional[float] = None,
    strategy: Literal["mean", "hamming", "any"] = "mean",
):
    if latency is None:
        latency = step
    assert duration >= latency >= step
    assert strategy in ["mean", "hamming", "any"]
    if strategy == "hamming":
        warnings.warn("'hamming' aggregation is not supported yet, defaulting to 'mean'")
    num_overlapping = int(round(latency / step))

    def apply(buffers: List[SlidingWindowFeature]) -> SlidingWindowFeature:
        # Determine overlapping region to aggregate
        real_time = buffers[-1].extent.end
        start_time = 0
        if buffers[0].extent.start > 0:
            start_time = real_time - latency
        required = Segment(start_time, real_time - latency + step)
        # Stack all overlapping regions
        intersection = np.stack([
            buffer.crop(required, fixed=required.duration)
            for buffer in buffers
        ])
        # Aggregate according to strategy
        if strategy in ("mean", "hamming"):
            aggregation = np.mean(intersection, axis=0)
        else:
            aggregation = intersection[0]
        # Determine resolution
        resolution = buffers[-1].sliding_window
        resolution = SlidingWindow(start=required.start, duration=resolution.duration, step=resolution.step)
        return SlidingWindowFeature(aggregation, resolution)

    return ops.pipe(
        # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
        ops.buffer_with_count(num_overlapping, 1),
        # Aggregate buffered chunks
        ops.map(apply)
    )
