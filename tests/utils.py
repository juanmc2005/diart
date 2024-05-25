from __future__ import annotations
import random
import numpy as np
from pyannote.core import SlidingWindowFeature, SlidingWindow


def build_waveform_swf(
    duration: float, sample_rate: int, start_time: float | None = None
) -> SlidingWindowFeature:
    start_time = round(random.uniform(0, 600), 1) if start_time is None else start_time
    chunk_size = int(duration * sample_rate)
    resolution = duration / chunk_size
    samples = np.random.randn(chunk_size, 1)
    sliding_window = SlidingWindow(
        start=start_time, step=resolution, duration=resolution
    )
    return SlidingWindowFeature(samples, sliding_window)
