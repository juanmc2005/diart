import math
from pathlib import Path
from typing import Optional, Text, List, Tuple

import numpy as np
import rx
import rx.operators as ops
import torch
from einops import rearrange
from pyannote.core import SlidingWindowFeature, SlidingWindow
from tqdm import tqdm

from . import blocks
from . import models as m
from . import operators as dops
from . import sources as src
from . import utils


class PipelineConfig:
    def __init__(
        self,
        segmentation: Optional[m.SegmentationModel] = None,
        embedding: Optional[m.EmbeddingModel] = None,
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
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation
        if self.segmentation is None:
            self.segmentation = m.SegmentationModel.from_pyannote("pyannote/segmentation")

        # Default duration is the one given by the segmentation model
        self.duration = duration
        if self.duration is None:
            self.duration = self.segmentation.get_duration()

        # Expected sample rate is given by the segmentation model
        self.sample_rate = self.segmentation.get_sample_rate()

        # Default embedding model is pyannote/embedding
        self.embedding = embedding
        if self.embedding is None:
            self.embedding = m.EmbeddingModel.from_pyannote("pyannote/embedding")

        # Latency defaults to the step duration
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def get_end_time(self, duration: Optional[float]) -> Optional[float]:
        return None if duration is None else self.config.last_chunk_end_time(duration)

    def get_operators(
        self,
        source_uri: Text,
        source_duration: Optional[float],
        output_waveform: bool = True
    ) -> List[dops.Operator]:
        clustering = blocks.OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        end_time = self.get_end_time(source_duration)
        pred_aggregation = blocks.DelayedAggregation(
            self.config.step, self.config.latency, strategy="hamming", stream_end=end_time
        )
        audio_aggregation = blocks.DelayedAggregation(
            self.config.step, self.config.latency, strategy="first", stream_end=end_time
        )
        binarize = blocks.Binarize(source_uri, self.config.tau_active)
        return [
            # Identify global speakers with online clustering
            ops.starmap(lambda wav, seg, emb: (wav, clustering(seg, emb))),
            # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
            dops.buffer_slide(pred_aggregation.num_overlapping_windows),
            # Aggregate overlapping output windows
            ops.map(lambda buffers: utils.unzip(buffers)),
            ops.starmap(lambda wav_buffer, pred_buffer: (
                audio_aggregation(wav_buffer), pred_aggregation(pred_buffer)
            )),
            # Binarize output
            ops.starmap(lambda wav, pred: (binarize(pred), wav if output_waveform else None)),
        ]


class OnlineSpeakerDiarization:
    def __init__(self, config: PipelineConfig, profile: bool = False):
        self.config = config
        self.profile = profile
        self.segmentation = blocks.SpeakerSegmentation(config.segmentation, config.device)
        self.embedding = blocks.OverlapAwareSpeakerEmbedding(
            config.embedding, config.gamma, config.beta, norm=1, device=config.device
        )
        self.speaker_tracking = OnlineSpeakerTracking(config)
        msg = f"Latency should be in the range [{config.step}, {config.duration}]"
        assert config.step <= config.latency <= config.duration, msg

    def from_source(self, source: src.AudioSource, output_waveform: bool = True) -> rx.Observable:
        msg = f"Audio source has sample rate {source.sample_rate}, expected {self.config.sample_rate}"
        assert source.sample_rate == self.config.sample_rate, msg
        operators = []
        # Regularize the stream to a specific chunk duration and step
        if not source.is_regular:
            operators.append(dops.regularize_stream(
                self.config.duration, self.config.step, source.sample_rate
            ))
        operators += [
            # Extract segmentation and keep audio
            ops.map(lambda wav: (wav, self.segmentation(wav))),
            # Extract embeddings and keep segmentation
            ops.starmap(lambda wav, seg: (wav, seg, self.embedding(wav, seg))),
        ]
        # Add speaker tracking
        operators += self.speaker_tracking.get_operators(source.uri, source.duration, output_waveform)
        if self.profile:
            return dops.profile(source.stream, operators)
        return source.stream.pipe(*operators)

    def from_file(
        self,
        filepath: src.FilePath,
        output_waveform: bool = False,
        batch_size: int = 32,
        desc: Optional[Text] = None,
    ) -> rx.Observable:
        loader = src.AudioLoader(self.config.sample_rate, mono=True)

        # Split audio into chunks
        chunks = rearrange(
            loader.load_sliding_chunks(filepath, self.config.duration, self.config.step),
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
        resolution = self.config.duration / segmentation.shape[1]
        seg_stream = rx.range(0, num_chunks).pipe(
            ops.map(lambda i: SlidingWindowFeature(
                segmentation[i], SlidingWindow(resolution, resolution, i * self.config.step)
            ))
        )
        emb_stream = rx.range(0, num_chunks).pipe(ops.map(lambda i: embeddings[i]))
        wav_resolution = 1 / self.config.sample_rate
        chunk_stream = None
        if output_waveform:
            chunk_stream = rx.range(0, num_chunks).pipe(
                ops.map(lambda i: SlidingWindowFeature(
                    chunks[i], SlidingWindow(wav_resolution, wav_resolution, i * self.config.step)
                ))
            )

        # Build speaker tracking pipeline
        return rx.zip(chunk_stream, seg_stream, emb_stream).pipe(
            *self.speaker_tracking.get_operators(
                Path(filepath).stem,
                loader.get_duration(filepath),
                output_waveform,
            )
        )
