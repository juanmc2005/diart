from typing import Optional

import rx
import rx.operators as ops
from pyannote.audio.pipelines.utils import PipelineModel

from . import functional as fn
from . import operators as dops
from . import sources as src


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

    @property
    def sample_rate(self) -> int:
        return self.segmentation.model.audio.sample_rate

    def get_end_time(self, source: src.AudioSource) -> Optional[float]:
        if source.duration is not None:
            return source.duration - source.duration % self.step
        return None

    def from_source(self, source: src.AudioSource, output_waveform: bool = True) -> rx.Observable:
        msg = f"Audio source has sample rate {source.sample_rate}, expected {self.sample_rate}"
        assert source.sample_rate == self.sample_rate, msg
        # Regularize the stream to a specific chunk duration and step
        regular_stream = source.stream
        if not source.is_regular:
            regular_stream = source.stream.pipe(
                dops.regularize_stream(self.duration, self.step, source.sample_rate)
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
        end_time = self.get_end_time(source)
        aggregation = fn.DelayedAggregation(
            self.step, self.latency, strategy="hamming", stream_end=end_time
        )
        pipeline = rx.zip(segmentation_stream, embedding_stream).pipe(
            ops.starmap(clustering),
            # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
            dops.buffer_slide(aggregation.num_overlapping_windows),
            # Aggregate overlapping output windows
            ops.map(aggregation),
            # Binarize output
            ops.map(fn.Binarize(source.uri, self.tau_active)),
        )
        # Add corresponding waveform to the output
        if output_waveform:
            window_selector = fn.DelayedAggregation(
                self.step, self.latency, strategy="first", stream_end=end_time
            )
            waveform_stream = regular_stream.pipe(
                dops.buffer_slide(window_selector.num_overlapping_windows),
                ops.map(window_selector),
            )
            return rx.zip(pipeline, waveform_stream)
        # No waveform needed, add None for consistency
        return pipeline.pipe(ops.map(lambda ann: (ann, None)))
