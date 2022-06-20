from typing import Text

import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature


class Binarize:
    """
    Transform a speaker segmentation from the discrete-time domain
    into a continuous-time speaker segmentation.

    Parameters
    ----------
    uri: Text
        Uri of the audio stream.
    threshold: float
        Probability threshold to determine if a speaker is active at a given frame.
    """

    def __init__(self, uri: Text, threshold: float):
        self.uri = uri
        self.threshold = threshold

    def __call__(self, segmentation: SlidingWindowFeature) -> Annotation:
        """
        Return the continuous-time segmentation
        corresponding to the discrete-time input segmentation.

        Parameters
        ----------
        segmentation: SlidingWindowFeature
            Discrete-time speaker segmentation.

        Returns
        -------
        annotation: Annotation
            Continuous-time speaker segmentation.
        """
        num_frames, num_speakers = segmentation.data.shape
        timestamps = segmentation.sliding_window
        is_active = segmentation.data > self.threshold
        # Artificially add last inactive frame to close any remaining speaker turns
        is_active = np.append(is_active, [[False] * num_speakers], axis=0)
        start_times = np.zeros(num_speakers) + timestamps[0].middle
        annotation = Annotation(uri=self.uri, modality="speech")
        for t in range(num_frames):
            # Any (False, True) starts a speaker turn at "True" index
            onsets = np.logical_and(np.logical_not(is_active[t]), is_active[t + 1])
            start_times[onsets] = timestamps[t + 1].middle
            # Any (True, False) ends a speaker turn at "False" index
            offsets = np.logical_and(is_active[t], np.logical_not(is_active[t + 1]))
            for spk in np.where(offsets)[0]:
                region = Segment(start_times[spk], timestamps[t + 1].middle)
                annotation[region, spk] = f"speaker{spk}"
        return annotation
