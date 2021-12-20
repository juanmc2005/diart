import rx
import rx.operators as ops
from typing import Optional
from pyannote.audio.pipelines.utils import PipelineModel

from .sources import AudioSource
from . import operators as my_ops
from . import functional as fn


class OnlineSpeakerDiarization:
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        embedding: PipelineModel = "pyannote/embedding",
        duration: Optional[float] = None,
        step: float = 0.5,
        latency: Optional[float] = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
    ):
        self.segmentation = fn.FrameWiseModel(segmentation)
        self.duration = duration
        if self.duration is None:
            self.duration = self.segmentation.model.specifications.duration
        self.embedding = fn.ChunkWiseModel(embedding)
        self.step = step
        self.latency = latency
        if self.latency is None:
            self.latency = self.step
        assert self.step <= self.latency <= self.duration, "Invalid latency requested"
        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers

    def from_source(self, source: AudioSource, output_waveform: bool = False) -> rx.Observable:
        # Regularize the stream to a specific chunk duration and step
        regular_stream = source.stream
        if not source.is_regular:
            regular_stream = source.stream.pipe(
                my_ops.regularize_stream(self.duration, self.step, source.sample_rate)
            )
        # Branch the stream to calculate chunk segmentation
        segmentation_stream = regular_stream.pipe(
            ops.map(self.segmentation)
        )
        # Join audio and segmentation stream to calculate speaker embeddings
        osp = fn.OverlappedSpeechPenalty(gamma=self.gamma, beta=self.beta)
        embedding_stream = rx.zip(regular_stream, segmentation_stream).pipe(
            ops.starmap(lambda wave, seg: (wave, osp(seg))),
            ops.starmap(self.embedding),
            ops.map(fn.EmbeddingNormalization(norm=1))
        )
        # Join segmentation and embedding streams to update a background clustering model
        #  while regulating latency and binarizing the output
        clustering = fn.OnlineSpeakerClustering(
            self.tau_active, self.rho_update, self.delta_new, "cosine", self.max_speakers
        )
        aggregation = fn.DelayedAggregation(self.step, self.latency, strategy="hamming")
        pipeline = rx.zip(segmentation_stream, embedding_stream).pipe(
            ops.starmap(clustering),
            # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
            my_ops.buffer_slide(aggregation.num_overlapping_windows),
            # Aggregate overlapping output windows
            ops.map(aggregation),
            # Binarize output
            ops.map(fn.Binarize(source.uri, self.tau_active)),
        )
        if output_waveform:
            window_selector = fn.DelayedAggregation(self.step, self.latency, strategy="first")
            pipeline = pipeline.pipe(
                ops.zip(regular_stream.pipe(
                    my_ops.buffer_slide(window_selector.num_overlapping_windows),
                    ops.map(window_selector),
                ))
            )
        return pipeline
