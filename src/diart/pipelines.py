import math
from pathlib import Path
from typing import Optional, Union, Text

import numpy as np
import rx
import rx.operators as ops
import torch
from einops import rearrange
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import SlidingWindowFeature, SlidingWindow
from tqdm import tqdm

from . import blocks
from . import operators as dops
from . import sources as src


class PipelineConfig:
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
        device: Optional[torch.device] = None,
    ):
        self.segmentation = segmentation
        self.embedding = embedding
        self.requested_duration = duration
        self.step = step
        self.latency = latency
        if self.latency is None:
            self.latency = self.step
        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

    def last_chunk_end_time(self, conv_duration: float) -> Optional[float]:
        """
        Return the end time of the last chunk for a given conversation duration.

        Parameters
        ----------
        conv_duration: float
            Duration of a conversation in seconds.
        """
        return conv_duration - conv_duration % self.step


class OnlineSpeakerTracking:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def from_model_streams(
        self,
        uri: Text,
        source_duration: Optional[float],
        segmentation_stream: rx.Observable,
        embedding_stream: rx.Observable,
        audio_chunk_stream: Optional[rx.Observable] = None,
    ) -> rx.Observable:
        end_time = None
        if source_duration is not None:
            end_time = self.config.last_chunk_end_time(source_duration)
        # Initialize clustering and aggregation modules
        clustering = blocks.OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        aggregation = blocks.DelayedAggregation(
            self.config.step, self.config.latency, strategy="hamming", stream_end=end_time
        )
        binarize = blocks.Binarize(uri, self.config.tau_active)

        # Join segmentation and embedding streams to update a background clustering model
        #  while regulating latency and binarizing the output
        pipeline = rx.zip(segmentation_stream, embedding_stream).pipe(
            ops.starmap(clustering),
            # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
            dops.buffer_slide(aggregation.num_overlapping_windows),
            # Aggregate overlapping output windows
            ops.map(aggregation),
            # Binarize output
            ops.map(binarize),
        )
        # Add corresponding waveform to the output
        if audio_chunk_stream is not None:
            window_selector = blocks.DelayedAggregation(
                self.config.step, self.config.latency, strategy="first", stream_end=end_time
            )
            waveform_stream = audio_chunk_stream.pipe(
                dops.buffer_slide(window_selector.num_overlapping_windows),
                ops.map(window_selector),
            )
            return rx.zip(pipeline, waveform_stream)
        # No waveform needed, add None for consistency
        return pipeline.pipe(ops.map(lambda ann: (ann, None)))


class OnlineSpeakerDiarization:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.segmentation = blocks.FramewiseModel(config.segmentation, config.device)
        self.embedding = blocks.OverlapAwareSpeakerEmbedding(
            config.embedding, config.gamma, config.beta, norm=1, device=config.device
        )
        self.speaker_tracking = OnlineSpeakerTracking(config)
        msg = "Invalid latency requested"
        assert config.step <= config.latency <= self.duration, msg

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by the segmentation model"""
        return self.segmentation.sample_rate

    @property
    def duration(self) -> float:
        """Chunk duration (in seconds). Defaults to segmentation model duration"""
        duration = self.config.requested_duration
        if duration is None:
            duration = self.segmentation.duration
        return duration

    def from_source(
        self,
        source: src.AudioSource,
        output_waveform: bool = True
    ) -> rx.Observable:
        msg = f"Audio source has sample rate {source.sample_rate}, expected {self.sample_rate}"
        assert source.sample_rate == self.sample_rate, msg
        # Regularize the stream to a specific chunk duration and step
        regular_stream = source.stream
        if not source.is_regular:
            regular_stream = source.stream.pipe(
                dops.regularize_stream(self.duration, self.config.step, source.sample_rate)
            )
        # Branch the stream to calculate chunk segmentation
        seg_stream = regular_stream.pipe(ops.map(self.segmentation))
        # Join audio and segmentation stream to calculate overlap-aware speaker embeddings
        emb_stream = rx.zip(regular_stream, seg_stream).pipe(ops.starmap(self.embedding))
        chunk_stream = regular_stream if output_waveform else None
        return self.speaker_tracking.from_model_streams(
            source.uri, source.duration, seg_stream, emb_stream, chunk_stream
        )

    def from_file(
        self,
        file: Union[Text, Path],
        output_waveform: bool = False,
        batch_size: int = 32,
        desc: Optional[Text] = None,
    ) -> rx.Observable:
        # Audio file information
        file = Path(file)
        chunk_loader = src.ChunkLoader(
            self.sample_rate, self.duration, self.config.step
        )

        # Split audio into chunks
        chunks = rearrange(
            chunk_loader.get_chunks(file),
            "chunk channel sample -> chunk sample channel"
        )
        num_chunks = chunks.shape[0]

        # Set progress if needed
        iterator = range(0, num_chunks, batch_size)
        if desc is not None:
            total = int(math.ceil(num_chunks / batch_size))
            iterator = tqdm(iterator, desc=desc, total=total, unit="batch", leave=False)

        # Pre-calculate segmentation and embeddings
        segmentation, embeddings = [], []
        for i in iterator:
            i_end = i + batch_size
            if i_end > num_chunks:
                i_end = num_chunks
            batch = chunks[i:i_end]
            seg = self.segmentation(batch)
            # Edge case: add batch dimension if i == i_end + 1
            if seg.ndim == 2:
                seg = seg[np.newaxis]
            emb = self.embedding(batch, seg)
            # Edge case: add batch dimension if i == i_end + 1
            if emb.ndim == 2:
                emb = emb.unsqueeze(0)
            segmentation.append(seg)
            embeddings.append(emb)
        segmentation = np.vstack(segmentation)
        embeddings = torch.vstack(embeddings)

        # Stream pre-calculated segmentation, embeddings and chunks
        resolution = self.duration / segmentation.shape[1]
        seg_stream = rx.range(0, num_chunks).pipe(
            ops.map(lambda i: SlidingWindowFeature(
                segmentation[i], SlidingWindow(resolution, resolution, i * self.config.step)
            ))
        )
        emb_stream = rx.range(0, num_chunks).pipe(ops.map(lambda i: embeddings[i]))
        wav_resolution = 1 / self.sample_rate
        chunk_stream = None
        if output_waveform:
            chunk_stream = rx.range(0, num_chunks).pipe(
                ops.map(lambda i: SlidingWindowFeature(
                    chunks[i], SlidingWindow(wav_resolution, wav_resolution, i * self.config.step)
                ))
            )

        # Build speaker tracking pipeline
        duration = chunk_loader.audio.get_duration(file)
        return self.speaker_tracking.from_model_streams(
            file.stem, duration, seg_stream, emb_stream, chunk_stream
        )
