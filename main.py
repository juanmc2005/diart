import rx
import rx.operators as ops
import sources as src
import operators as my_ops
import functional as fn
from traceback import print_exc
import utils

duration = 5
step = 0.5
latency = 5
sample_rate = 16000

# Simulate an unreliable recording protocol yielding new audio with a varying refresh rates
unreliable_source = src.UnreliableFileAudioSource(refresh_rate_range=(0.1, 0.9), sample_rate=sample_rate)
# Regularize the stream to a specific chunk duration and step
regular_stream = unreliable_source.stream.pipe(
    my_ops.regularize_stream(duration, step, sample_rate)
)
# Branch the stream to calculate chunk segmentation
segmentation_stream = regular_stream.pipe(
    ops.map(fn.FrameWiseModel("pyannote/segmentation"))
)
# Join audio and segmentation stream to calculate speaker embeddings
osp = fn.OverlappedSpeechPenalty(gamma=3, beta=10)
embedding_stream = rx.zip(regular_stream, segmentation_stream).pipe(
    ops.starmap(lambda wave, seg: (wave, osp(seg))),
    ops.starmap(fn.ChunkWiseModel("pyannote/embedding")),
    ops.map(fn.EmbeddingNormalization(norm=1))
)
# Join segmentation and embedding streams to update a background clustering model
clustering = fn.OnlineSpeakerClustering(
    tau_active=0.6, rho_update=0.3, delta_new=1, k_max_speakers=4
)
pipeline = rx.zip(segmentation_stream, embedding_stream).pipe(
    ops.starmap(clustering),
    my_ops.aggregate(duration, step, latency),
    # TODO binarize
    ops.take(5)
)


pipeline.subscribe(
    on_next=utils.visualize(duration),
    on_error=lambda e: print_exc(),
    on_completed=lambda: print("Done")
)

unreliable_source.read("/home/coria/DH_DEV_0001.flac")
