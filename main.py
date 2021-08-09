import numpy as np
import rx.operators as ops
import torch

import sources as src
import operators as pops
import functional as fn
from pyannote.core import SlidingWindowFeature
from traceback import print_exc
import utils

duration = 5
step = 0.5
sample_rate = 16000

# Simulate an unreliable recording protocol yielding new audio with a varying refresh rates
stream = src.UnreliableFileAudioStream(refresh_rate_range=(0.1, 0.9), sample_rate=sample_rate)
# regular_stream = stream("/home/coria/DH_DEV_0001.flac").pipe(
#     pops.RegularizeStream(duration, step, sample_rate)
# )
# segmented = regular_stream.pipe(
#     ops.map(fn.FrameWiseModel("pyannote/segmentation"))
# )

pipeline = stream("/home/coria/DH_DEV_0001.flac").pipe(
    pops.RegularizeStream(duration, step, sample_rate),  # waveform
    pops.make_pair(fn.FrameWiseModel("pyannote/segmentation")),  # (waveform, segmentation)
    pops.map_arg_append(1, fn.OverlappedSpeechPenalty(gamma=3, beta=10)),  # (waveform, segmentation, weights)
    pops.map_many_args([0, 2], fn.Embedding("pyannote/embedding")),  # (segmentation, embeddings)
    pops.map_arg(1, fn.EmbeddingNormalization(norm=1)),  # (segmentation, embeddings)
    pops.ClusterSpeakers(tau_active=0.6, rho_update=0.3, delta_new=1, k_max_speakers=4),  # (permutation)
    # TODO latency
    # TODO binarize
    ops.take(5)
)


def show(feature: SlidingWindowFeature):
    print(feature.data.shape, feature.sliding_window.start)


def print_shape(array: np.ndarray):
    print(array.shape)


def print_norm(embs: torch.Tensor):
    print(torch.norm(embs, p=2, dim=1, keepdim=True))


pipeline.subscribe(
    on_next=utils.visualize,
    on_error=lambda e: print_exc()
)
