from dataclasses import dataclass
from typing import Callable, Optional, List, Any, Tuple

import numpy as np
import rx
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from rx import operators as ops
from rx.core import Observable

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


def buffer_slide(n: int):
    def accumulate(state: List[Any], value: Any) -> List[Any]:
        new_state = [*state, value]
        if len(new_state) > n:
            return new_state[1:]
        return new_state
    return rx.pipe(ops.scan(accumulate, []))


@dataclass
class PredictionWithAudio:
    prediction: Annotation
    waveform: Optional[SlidingWindowFeature] = None

    @property
    def has_audio(self) -> bool:
        return self.waveform is not None


@dataclass
class OutputAccumulationState:
    annotation: Optional[Annotation]
    waveform: Optional[SlidingWindowFeature]
    real_time: float
    next_sample: Optional[int]

    @staticmethod
    def initial() -> 'OutputAccumulationState':
        return OutputAccumulationState(None, None, 0, 0)

    @property
    def cropped_waveform(self) -> SlidingWindowFeature:
        return SlidingWindowFeature(
            self.waveform[:self.next_sample],
            self.waveform.sliding_window,
        )

    def to_tuple(self) -> Tuple[Optional[Annotation], Optional[SlidingWindowFeature], float]:
        return self.annotation, self.cropped_waveform, self.real_time


def accumulate_output(
    duration: float,
    step: float,
    merge_collar: float = 0.05,
) -> Operator:
    def accumulate(
        state: OutputAccumulationState,
        value: Tuple[Annotation, Optional[SlidingWindowFeature]]
    ) -> OutputAccumulationState:
        value = PredictionWithAudio(*value)
        annotation, waveform, real_time = None, None, 0

        if state.annotation is None:
            annotation = value.prediction
            real_time = duration
        else:
            annotation = state.annotation.update(value.prediction).support(merge_collar)
            real_time = state.real_time + step

        new_next_sample = 0
        if value.has_audio:
            num_new_samples = value.waveform.data.shape[0]
            new_next_sample = state.next_sample + num_new_samples
            sw_holder = state
            if state.waveform is None:
                waveform, sw_holder = np.zeros((10 * num_new_samples, 1)), value
            elif new_next_sample < state.waveform.data.shape[0]:
                waveform = state.waveform.data
            else:
                waveform = np.concatenate(
                    (state.waveform.data, np.zeros_like(state.waveform.data)), axis=0
                )
            waveform[state.next_sample:new_next_sample] = value.waveform.data
            waveform = SlidingWindowFeature(waveform, sw_holder.waveform.sliding_window)

        return OutputAccumulationState(annotation, waveform, real_time, new_next_sample)

    return rx.pipe(
        ops.scan(accumulate, OutputAccumulationState.initial()),
        ops.map(OutputAccumulationState.to_tuple),
    )
