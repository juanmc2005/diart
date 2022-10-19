from typing import Optional, List

import numpy as np
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from typing_extensions import Literal


class AggregationStrategy:
    """Abstract class representing a strategy to aggregate overlapping buffers

    Parameters
    ----------
    cropping_mode: ("strict", "loose", "center"), optional
        Defines the mode to crop buffer chunks as in pyannote.core.
        See https://pyannote.github.io/pyannote-core/reference.html#pyannote.core.SlidingWindowFeature.crop
        Defaults to "loose".
    """

    def __init__(self, cropping_mode: Literal["strict", "loose", "center"] = "loose"):
        assert cropping_mode in ["strict", "loose", "center"], f"Invalid cropping mode `{cropping_mode}`"
        self.cropping_mode = cropping_mode

    @staticmethod
    def build(
        name: Literal["mean", "hamming", "first"],
        cropping_mode: Literal["strict", "loose", "center"] = "loose"
    ) -> 'AggregationStrategy':
        """Build an AggregationStrategy instance based on its name"""
        assert name in ("mean", "hamming", "first")
        if name == "mean":
            return AverageStrategy(cropping_mode)
        elif name == "hamming":
            return HammingWeightedAverageStrategy(cropping_mode)
        else:
            return FirstOnlyStrategy(cropping_mode)

    def __call__(self, buffers: List[SlidingWindowFeature], focus: Segment) -> SlidingWindowFeature:
        """Aggregate chunks over a specific region.

        Parameters
        ----------
        buffers: list of SlidingWindowFeature, shapes (frames, speakers)
            Buffers to aggregate
        focus: Segment
            Region to aggregate that is shared among the buffers

        Returns
        -------
        aggregation: SlidingWindowFeature, shape (cropped_frames, speakers)
            Aggregated values over the focus region
        """
        aggregation = self.aggregate(buffers, focus)
        resolution = focus.duration / aggregation.shape[0]
        resolution = SlidingWindow(
            start=focus.start,
            duration=resolution,
            step=resolution
        )
        return SlidingWindowFeature(aggregation, resolution)

    def aggregate(self, buffers: List[SlidingWindowFeature], focus: Segment) -> np.ndarray:
        raise NotImplementedError


class HammingWeightedAverageStrategy(AggregationStrategy):
    """Compute the average weighted by the corresponding Hamming-window aligned to each buffer"""

    def aggregate(self, buffers: List[SlidingWindowFeature], focus: Segment) -> np.ndarray:
        num_frames, num_speakers = buffers[0].data.shape
        hamming, intersection = [], []
        for buffer in buffers:
            # Crop buffer to focus region
            b = buffer.crop(focus, mode=self.cropping_mode, fixed=focus.duration)
            # Crop Hamming window to focus region
            h = np.expand_dims(np.hamming(num_frames), axis=-1)
            h = SlidingWindowFeature(h, buffer.sliding_window)
            h = h.crop(focus, mode=self.cropping_mode, fixed=focus.duration)
            hamming.append(h.data)
            intersection.append(b.data)
        hamming, intersection = np.stack(hamming), np.stack(intersection)
        # Calculate weighted mean
        return np.sum(hamming * intersection, axis=0) / np.sum(hamming, axis=0)


class AverageStrategy(AggregationStrategy):
    """Compute a simple average over the focus region"""

    def aggregate(self, buffers: List[SlidingWindowFeature], focus: Segment) -> np.ndarray:
        # Stack all overlapping regions
        intersection = np.stack([
            buffer.crop(focus, mode=self.cropping_mode, fixed=focus.duration)
            for buffer in buffers
        ])
        return np.mean(intersection, axis=0)


class FirstOnlyStrategy(AggregationStrategy):
    """Instead of aggregating, keep the first focus region in the buffer list"""

    def aggregate(self, buffers: List[SlidingWindowFeature], focus: Segment) -> np.ndarray:
        return buffers[0].crop(focus, mode=self.cropping_mode, fixed=focus.duration)


class DelayedAggregation:
    """Aggregate aligned overlapping windows of the same duration
    across sliding buffers with a specific step and latency.

    Parameters
    ----------
    step: float
        Shift between two consecutive buffers, in seconds.
    latency: float, optional
        Desired latency, in seconds. Defaults to step.
        The higher the latency, the more overlapping windows to aggregate.
    strategy: ("mean", "hamming", "first"), optional
        Specifies how to aggregate overlapping windows. Defaults to "hamming".
        "mean": simple average
        "hamming": average weighted by the Hamming window values (aligned to the buffer)
        "first": no aggregation, pick the first overlapping window
    cropping_mode: ("strict", "loose", "center"), optional
        Defines the mode to crop buffer chunks as in pyannote.core.
        See https://pyannote.github.io/pyannote-core/reference.html#pyannote.core.SlidingWindowFeature.crop
        Defaults to "loose".

    Example
    --------
    >>> duration = 5
    >>> frames = 500
    >>> step = 0.5
    >>> speakers = 2
    >>> start_time = 10
    >>> resolution = duration / frames
    >>> dagg = DelayedAggregation(step=step, latency=2, strategy="mean")
    >>> buffers = [
    >>>     SlidingWindowFeature(
    >>>         np.random.rand(frames, speakers),
    >>>         SlidingWindow(start=(i + start_time) * step, duration=resolution, step=resolution)
    >>>     )
    >>>     for i in range(dagg.num_overlapping_windows)
    >>> ]
    >>> dagg.num_overlapping_windows
    ... 4
    >>> dagg(buffers).data.shape
    ... (51, 2)  # Rounding errors are possible when cropping the buffers
    """

    def __init__(
        self,
        step: float,
        latency: Optional[float] = None,
        strategy: Literal["mean", "hamming", "first"] = "hamming",
        cropping_mode: Literal["strict", "loose", "center"] = "loose"
    ):
        self.step = step
        self.latency = latency
        self.strategy = strategy
        assert cropping_mode in ["strict", "loose", "center"], f"Invalid cropping mode `{cropping_mode}`"
        self.cropping_mode = cropping_mode

        if self.latency is None:
            self.latency = self.step

        assert self.step <= self.latency, "Invalid latency requested"

        self.num_overlapping_windows = int(round(self.latency / self.step))
        self.aggregate = AggregationStrategy.build(self.strategy, self.cropping_mode)

    def _prepend(
        self,
        output_window: SlidingWindowFeature,
        output_region: Segment,
        buffers: List[SlidingWindowFeature]
    ):
        # FIXME instead of prepending the output of the first chunk,
        #  add padding of `chunk_duration - latency` seconds at the
        #  beginning of the stream so scores can be aggregated accordingly.
        #  Remember to shift predictions by the padding.
        last_buffer = buffers[-1].extent
        # Prepend prediction until we match the latency in case of first buffer
        if len(buffers) == 1 and last_buffer.start == 0:
            num_frames = output_window.data.shape[0]
            first_region = Segment(0, output_region.end)
            first_output = buffers[0].crop(
                first_region, mode=self.cropping_mode, fixed=first_region.duration
            )
            first_output[-num_frames:] = output_window.data
            resolution = output_region.end / first_output.shape[0]
            output_window = SlidingWindowFeature(
                first_output,
                SlidingWindow(start=0, duration=resolution, step=resolution)
            )
        return output_window

    def __call__(self, buffers: List[SlidingWindowFeature]) -> SlidingWindowFeature:
        # Determine overlapping region to aggregate
        start = buffers[-1].extent.end - self.latency
        region = Segment(start, start + self.step)
        return self._prepend(self.aggregate(buffers, region), region, buffers)
