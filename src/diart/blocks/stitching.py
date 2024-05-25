import numpy as np
from pyannote.core import SlidingWindowFeature
from pyannote.audio.utils.permutation import permutate


class OnlineStitching:
    """Implements online stitching of local speaker segmentation

    Parameters
    ----------
    """

    def __init__(self):
        self.previous_segmentation: SlidingWindowFeature | None = None

    def stitch(self, segmentation: SlidingWindowFeature) -> SlidingWindowFeature:
        """

        Parameters
        ----------
        segmentation: np.ndarray, shape (frames, local_speakers)
            Matrix of segmentation outputs

        Returns
        -------
        permutated_segmentation: np.ndarray, shape (frames, pseudo_speakers)
        """

        if self.previous_segmentation is None:
            self.previous_segmentation = segmentation
            return segmentation

        num_frames, num_speakers = segmentation.data.shape
        duration = segmentation.extent.duration
        step = (
            segmentation.sliding_window.start
            - self.previous_segmentation.sliding_window.start
        )
        num_overlapping_frames = round(num_frames * (1 - step / duration))

        permutation = permutate(
            self.previous_segmentation[-num_overlapping_frames:][None],
            segmentation[:num_overlapping_frames],
        )[1][0]
        permutated_segmentation = SlidingWindowFeature(
            segmentation[:, permutation], segmentation.sliding_window
        )
        self.previous_segmentation = permutated_segmentation

        return permutated_segmentation

    def __call__(self, segmentation: SlidingWindowFeature) -> SlidingWindowFeature:
        return self.stitch(segmentation)
