import numpy as np
import pytest
from pyannote.core import SlidingWindow, SlidingWindowFeature

from diart.blocks.aggregation import (
    AggregationStrategy,
    HammingWeightedAverageStrategy,
    FirstOnlyStrategy,
    AverageStrategy,
    DelayedAggregation,
)


def test_strategy_build():
    strategy = AggregationStrategy.build("mean")
    assert isinstance(strategy, AverageStrategy)

    strategy = AggregationStrategy.build("hamming")
    assert isinstance(strategy, HammingWeightedAverageStrategy)

    strategy = AggregationStrategy.build("first")
    assert isinstance(strategy, FirstOnlyStrategy)

    with pytest.raises(Exception):
        AggregationStrategy.build("invalid")


def test_aggregation():
    duration = 5
    frames = 500
    step = 0.5
    speakers = 2
    start_time = 10
    resolution = duration / frames

    dagg1 = DelayedAggregation(step=step, latency=2, strategy="mean")
    dagg2 = DelayedAggregation(step=step, latency=2, strategy="hamming")
    dagg3 = DelayedAggregation(step=step, latency=2, strategy="first")

    for dagg in [dagg1, dagg2, dagg3]:
        assert dagg.num_overlapping_windows == 4

    buffers = [
        SlidingWindowFeature(
            np.random.rand(frames, speakers),
            SlidingWindow(
                start=(i + start_time) * step, duration=resolution, step=resolution
            ),
        )
        for i in range(dagg1.num_overlapping_windows)
    ]

    for dagg in [dagg1, dagg2, dagg3]:
        assert dagg(buffers).data.shape == (51, 2)
