from dataclasses import dataclass
from typing import Callable, Optional, List, Any, Tuple, Text

import numpy as np
import rx
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Segment
from rx import operators as ops
from rx.core import Observable
from tqdm import tqdm

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
    patch_collar: float = 0.05,
) -> Operator:
    """Accumulate predictions and audio to infinity: O(N) space complexity.
    Uses a pre-allocated buffer that doubles its size once full: O(logN) concat operations.

    Parameters
    ----------
    duration: float
        Buffer duration in seconds.
    step: float
        Duration of the chunks at each event in seconds.
        The first chunk may be bigger given the latency.
    patch_collar: float, optional
        Collar to merge speaker turns of the same speaker, in seconds.
        Defaults to 0.05 (i.e. 50ms).
    Returns
    -------
    A reactive x operator implementing this behavior.
    """
    def accumulate(
        state: OutputAccumulationState,
        value: Tuple[Annotation, Optional[SlidingWindowFeature]]
    ) -> OutputAccumulationState:
        value = PredictionWithAudio(*value)
        annotation, waveform = None, None

        # Determine the real time of the stream
        real_time = duration if state.annotation is None else state.real_time + step

        # Update total annotation with current predictions
        if state.annotation is None:
            annotation = value.prediction
        else:
            annotation = state.annotation.update(value.prediction).support(patch_collar)

        # Update total waveform if there's audio in the input
        new_next_sample = 0
        if value.has_audio:
            num_new_samples = value.waveform.data.shape[0]
            new_next_sample = state.next_sample + num_new_samples
            sw_holder = state
            if state.waveform is None:
                # Initialize the audio buffer with 10 times the size of the first chunk
                waveform, sw_holder = np.zeros((10 * num_new_samples, 1)), value
            elif new_next_sample < state.waveform.data.shape[0]:
                # The buffer still has enough space to accommodate the chunk
                waveform = state.waveform.data
            else:
                # The buffer is full, double its size
                waveform = np.concatenate(
                    (state.waveform.data, np.zeros_like(state.waveform.data)), axis=0
                )
            # Copy chunk into buffer
            waveform[state.next_sample:new_next_sample] = value.waveform.data
            waveform = SlidingWindowFeature(waveform, sw_holder.waveform.sliding_window)

        return OutputAccumulationState(annotation, waveform, real_time, new_next_sample)

    return rx.pipe(
        ops.scan(accumulate, OutputAccumulationState.initial()),
        ops.map(OutputAccumulationState.to_tuple),
    )


def buffer_output(
    duration: float,
    step: float,
    latency: float,
    sample_rate: int,
    patch_collar: float = 0.05,
) -> Operator:
    """Store last predictions and audio inside a fixed buffer.
    Provides the best time/space complexity trade-off if the past data is not needed.

    Parameters
    ----------
    duration: float
        Buffer duration in seconds.
    step: float
        Duration of the chunks at each event in seconds.
        The first chunk may be bigger given the latency.
    latency: float
        Latency of the system in seconds.
    sample_rate: int
        Sample rate of the audio source.
    patch_collar: float, optional
        Collar to merge speaker turns of the same speaker, in seconds.
        Defaults to 0.05 (i.e. 50ms).

    Returns
    -------
    A reactive x operator implementing this behavior.
    """
    # Define some useful constants
    num_samples = int(round(duration * sample_rate))
    num_step_samples = int(round(step * sample_rate))
    resolution = 1 / sample_rate

    def accumulate(
        state: OutputAccumulationState,
        value: Tuple[Annotation, Optional[SlidingWindowFeature]]
    ) -> OutputAccumulationState:
        value = PredictionWithAudio(*value)
        annotation, waveform = None, None

        # Determine the real time of the stream and the start time of the buffer
        real_time = duration if state.annotation is None else state.real_time + step
        start_time = max(0., real_time - latency - duration)

        # Update annotation and constrain its bounds to the buffer
        if state.annotation is None:
            annotation = value.prediction
        else:
            annotation = state.annotation.update(value.prediction).support(patch_collar)
            if start_time > 0:
                annotation = annotation.extrude(Segment(0, start_time))

        # Update the audio buffer if there's audio in the input
        new_next_sample = state.next_sample + num_step_samples
        if value.has_audio:
            if state.waveform is None:
                # Determine the size of the first chunk
                expected_duration = duration + step - latency
                expected_samples = int(round(expected_duration * sample_rate))
                # Shift indicator to start copying new audio in the buffer
                new_next_sample = state.next_sample + expected_samples
                # Buffer size is duration + step
                waveform = np.zeros((num_samples + num_step_samples, 1))
                # Copy first chunk into buffer (slicing because of rounding errors)
                waveform[:expected_samples] = value.waveform.data[:expected_samples]
            elif state.next_sample <= num_samples:
                # The buffer isn't full, copy into next free buffer chunk
                waveform = state.waveform.data
                waveform[state.next_sample:new_next_sample] = value.waveform.data
            else:
                # The buffer is full, shift values to the left and copy into last buffer chunk
                waveform = np.roll(state.waveform.data, -num_step_samples, axis=0)
                # If running on a file, the online prediction may be shorter depending on the latency
                # The remaining audio at the end is appended, so value.waveform may be longer than num_step_samples
                # In that case, we simply ignore the appended samples.
                waveform[-num_step_samples:] = value.waveform.data[:num_step_samples]

            # Wrap waveform in a sliding window feature to include timestamps
            window = SlidingWindow(start=start_time, duration=resolution, step=resolution)
            waveform = SlidingWindowFeature(waveform, window)

        return OutputAccumulationState(annotation, waveform, real_time, new_next_sample)

    return rx.pipe(
        ops.scan(accumulate, OutputAccumulationState.initial()),
        ops.map(OutputAccumulationState.to_tuple),
    )


def progress(
    desc: Optional[Text] = None,
    total: Optional[int] = None,
    unit: Text = "it",
    leave: bool = True
) -> Operator:
    pbar = tqdm(desc=desc, total=total, unit=unit, leave=leave)
    return rx.pipe(
        ops.do_action(
            on_next=lambda _: pbar.update(),
            on_error=lambda _: pbar.close(),
            on_completed=lambda: pbar.close(),
        )
    )
