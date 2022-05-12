from typing import Union, Optional, List, Iterable, Tuple
from typing_extensions import Literal

import numpy as np
import torch
from pyannote.audio.pipelines.utils import PipelineModel, get_model, get_devices
from pyannote.audio.utils.signal import Binarize as PyanBinarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from einops import rearrange

from .mapping import SpeakerMap, SpeakerMapBuilder


TemporalFeatures = Union[SlidingWindowFeature, np.ndarray, torch.Tensor]


def resolve_features(features: TemporalFeatures) -> torch.Tensor:
    """
    Transform features into a `torch.Tensor` and add batch dimension if missing.

    Parameters
    ----------
    features: Union[SlidingWindowFeature, np.ndarray, torch.Tensor]
        Shape (frames, channels) or (batch, frames, channels)

    Returns
    -------
    transformed_features: torch.Tensor, shape (batch, frames, channels)
    """
    # As torch.Tensor with shape (..., channels, frames)
    if isinstance(features, SlidingWindowFeature):
        data = torch.from_numpy(features.data)
    elif isinstance(features, np.ndarray):
        data = torch.from_numpy(features)
    else:
        data = features
    # Make sure there's a batch dimension
    msg = "Temporal features must be 2D or 3D"
    assert data.ndim in (2, 3), msg
    if data.ndim == 2:
        data = data.unsqueeze(0)
    return data.float()


class FramewiseModel:
    def __init__(self, model: PipelineModel, device: Optional[torch.device] = None):
        self.model = get_model(model)
        self.model.eval()
        if device is None:
            device = get_devices(needs=1)[0]
        self.model.to(device)

    @property
    def sample_rate(self) -> int:
        return self.model.audio.sample_rate

    @property
    def duration(self) -> float:
        return self.model.specifications.duration

    def __call__(self, waveform: TemporalFeatures) -> TemporalFeatures:
        with torch.no_grad():
            wave = rearrange(resolve_features(waveform), "batch sample channel -> batch channel sample")
            output = self.model(wave.to(self.model.device)).cpu()

        batch_size, num_frames, _ = output.shape

        # Remove batch dimension if batch size is 1
        if output.shape[0] == 1:
            output = output[0]

        # Wrap if a SlidingWindowFeature was given as input
        if isinstance(waveform, SlidingWindowFeature):
            # Temporal resolution of the output
            duration = wave.shape[-1] / self.sample_rate
            resolution = duration / num_frames
            # Temporal shift to keep track of current start time
            resolution = SlidingWindow(
                start=waveform.sliding_window.start,
                duration=resolution,
                step=resolution
            )
            return SlidingWindowFeature(output.numpy(), resolution)

        if isinstance(waveform, np.ndarray):
            return output.numpy()

        return output


class ChunkwiseModel:
    def __init__(self, model: PipelineModel, device: Optional[torch.device] = None):
        self.model = get_model(model)
        self.model.eval()
        if device is None:
            device = get_devices(needs=1)[0]
        self.model.to(device)

    def __call__(self, waveform: TemporalFeatures, weights: Optional[TemporalFeatures]) -> torch.Tensor:
        with torch.no_grad():
            inputs = resolve_features(waveform).to(self.model.device)
            inputs = rearrange(inputs, "batch sample channel -> batch channel sample")
            if weights is not None:
                weights = resolve_features(weights).to(self.model.device)
                batch_size, _, num_speakers = weights.shape
                inputs = inputs.repeat(1, num_speakers, 1)
                weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
                inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
                output = rearrange(
                    self.model(inputs, weights=weights),
                    "(batch spk) feat -> batch spk feat",
                    batch=batch_size,
                    spk=num_speakers
                )
            else:
                output = self.model(inputs)
            return output.squeeze().cpu()


class OverlappedSpeechPenalty:
    """
    Parameters
    ----------
    gamma: float, optional
        Exponent to lower low-confidence predictions.
        Defaults to 3.
    beta: float, optional
        Softmax's temperature parameter (actually 1/beta) to lower joint speaker activations.
        Defaults to 10.
    """
    def __init__(self, gamma: float = 3, beta: float = 10):
        self.gamma = gamma
        self.beta = beta

    def __call__(self, segmentation: TemporalFeatures) -> TemporalFeatures:
        weights = resolve_features(segmentation)  # shape (batch, frames, speakers)
        with torch.no_grad():
            probs = torch.softmax(self.beta * weights, dim=-1)
            weights = torch.pow(weights, self.gamma) * torch.pow(probs, self.gamma)
            weights[weights < 1e-8] = 1e-8
        if isinstance(segmentation, SlidingWindowFeature):
            return SlidingWindowFeature(weights.cpu().numpy(), segmentation.sliding_window)
        if isinstance(segmentation, np.ndarray):
            return weights.cpu().numpy()
        return weights


class EmbeddingNormalization:
    def __init__(self, norm: Union[float, torch.Tensor] = 1):
        self.norm = norm
        # Add batch dimension if missing
        if isinstance(self.norm, torch.Tensor) and self.norm.ndim == 2:
            self.norm = self.norm.unsqueeze(0)

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Add batch dimension if missing
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if isinstance(self.norm, torch.Tensor):
            batch_size1, num_speakers1, _ = self.norm.shape
            batch_size2, num_speakers2, _ = embeddings.shape
            assert batch_size1 == batch_size2 and num_speakers1 == num_speakers2
        with torch.no_grad():
            norm_embs = self.norm * embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return norm_embs.squeeze()


class OverlapAwareSpeakerEmbedding:
    """
    Extract overlap-aware speaker embeddings given an audio chunk and its segmentation.

    Parameters
    ----------
    model: pyannote.audio.Model, Text or Dict
        The embedding model. It must take a waveform and weights as input.
    gamma: float, optional
        Exponent to lower low-confidence predictions.
        Defaults to 3.
    beta: float, optional
        Softmax's temperature parameter (actually 1/beta) to lower joint speaker activations.
        Defaults to 10.
    norm: float or torch.Tensor of shape (batch, speakers, 1) where batch is optional
        The target norm for the embeddings. It can be different for each speaker.
        Defaults to 1.
    device: Optional[torch.device]
        The device on which to run the embedding model.
        Defaults to GPU if available or CPU if not.
    """
    def __init__(
        self,
        model: PipelineModel,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        device: Optional[torch.device] = None,
    ):
        self.embedding = ChunkwiseModel(model, device)
        self.osp = OverlappedSpeechPenalty(gamma, beta)
        self.normalize = EmbeddingNormalization(norm)

    def __call__(self, waveform: TemporalFeatures, segmentation: TemporalFeatures) -> torch.Tensor:
        return self.normalize(self.embedding(waveform, self.osp(segmentation)))


class AggregationStrategy:
    """Abstract class representing a strategy to aggregate overlapping buffers"""

    @staticmethod
    def build(name: Literal["mean", "hamming", "first"]) -> 'AggregationStrategy':
        """Build an AggregationStrategy instance based on its name"""
        assert name in ("mean", "hamming", "first")
        if name == "mean":
            return AverageStrategy()
        elif name == "hamming":
            return HammingWeightedAverageStrategy()
        else:
            return FirstOnlyStrategy()

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
            b = buffer.crop(focus, fixed=focus.duration)
            # Crop Hamming window to focus region
            h = np.expand_dims(np.hamming(num_frames), axis=-1)
            h = SlidingWindowFeature(h, buffer.sliding_window)
            h = h.crop(focus, fixed=focus.duration)
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
            buffer.crop(focus, fixed=focus.duration)
            for buffer in buffers
        ])
        return np.mean(intersection, axis=0)


class FirstOnlyStrategy(AggregationStrategy):
    """Instead of aggregating, keep the first focus region in the buffer list"""

    def aggregate(self, buffers: List[SlidingWindowFeature], focus: Segment) -> np.ndarray:
        return buffers[0].crop(focus, fixed=focus.duration)


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
    strategy: ("mean", "hamming", "any"), optional
        Specifies how to aggregate overlapping windows. Defaults to "hamming".
        "mean": simple average
        "hamming": average weighted by the Hamming window values (aligned to the buffer)
        "any": no aggregation, pick the first overlapping window
    stream_end: float, optional
        Stream end time (in seconds). Defaults to None.
        If the stream end time is known, then append remaining outputs at the end,
        otherwise the last `latency - step` seconds are ignored.

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
        stream_end: Optional[float] = None
    ):
        self.step = step
        self.latency = latency
        self.strategy = strategy
        self.stream_end = stream_end

        if self.latency is None:
            self.latency = self.step

        assert self.step <= self.latency, "Invalid latency requested"

        self.num_overlapping_windows = int(round(self.latency / self.step))
        self.aggregate = AggregationStrategy.build(self.strategy)

    def _prepend_or_append(
        self,
        output_window: SlidingWindowFeature,
        output_region: Segment,
        buffers: List[SlidingWindowFeature]
    ):
        last_buffer = buffers[-1].extent
        # Prepend prediction until we match the latency in case of first buffer
        if len(buffers) == 1 and last_buffer.start == 0:
            num_frames = output_window.data.shape[0]
            first_region = Segment(0, output_region.end)
            first_output = buffers[0].crop(
                first_region, fixed=first_region.duration
            )
            first_output[-num_frames:] = output_window.data
            resolution = output_region.end / first_output.shape[0]
            output_window = SlidingWindowFeature(
                first_output,
                SlidingWindow(start=0, duration=resolution, step=resolution)
            )
        # Append rest of the outputs
        elif self.stream_end is not None and last_buffer.end == self.stream_end:
            # FIXME instead of appending a larger chunk than expected when latency > step,
            #  keep emitting windows until the signal ends.
            #  This should be fixed at the observable level and not within the aggregation block.
            num_frames = output_window.data.shape[0]
            last_region = Segment(output_region.start, last_buffer.end)
            last_output = buffers[-1].crop(
                last_region, fixed=last_region.duration
            )
            last_output[:num_frames] = output_window.data
            resolution = self.latency / last_output.shape[0]
            output_window = SlidingWindowFeature(
                last_output,
                SlidingWindow(
                    start=output_region.start,
                    duration=resolution,
                    step=resolution
                )
            )
        return output_window

    def __call__(self, buffers: List[SlidingWindowFeature]) -> SlidingWindowFeature:
        # Determine overlapping region to aggregate
        start = buffers[-1].extent.end - self.latency
        region = Segment(start, start + self.step)
        return self._prepend_or_append(self.aggregate(buffers, region), region, buffers)


class OnlineSpeakerClustering:
    def __init__(
        self,
        tau_active: float,
        rho_update: float,
        delta_new: float,
        metric: Optional[str] = "cosine",
        max_speakers: int = 20
    ):
        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.metric = metric
        self.max_speakers = max_speakers
        self.centers: Optional[np.ndarray] = None
        self.active_centers = set()
        self.blocked_centers = set()

    @property
    def num_free_centers(self) -> int:
        return self.max_speakers - self.num_known_speakers - self.num_blocked_speakers

    @property
    def num_known_speakers(self) -> int:
        return len(self.active_centers)

    @property
    def num_blocked_speakers(self) -> int:
        return len(self.blocked_centers)

    @property
    def inactive_centers(self) -> List[int]:
        return [
            c
            for c in range(self.max_speakers)
            if c not in self.active_centers or c in self.blocked_centers
        ]

    def get_next_center_position(self) -> Optional[int]:
        for center in range(self.max_speakers):
            if center not in self.active_centers and center not in self.blocked_centers:
                return center

    def init_centers(self, dimension: int):
        self.centers = np.zeros((self.max_speakers, dimension))
        self.active_centers = set()
        self.blocked_centers = set()

    def update(self, assignments: Iterable[Tuple[int, int]], embeddings: np.ndarray):
        if self.centers is not None:
            for l_spk, g_spk in assignments:
                assert g_spk in self.active_centers, "Cannot update unknown centers"
                self.centers[g_spk] += embeddings[l_spk]

    def add_center(self, embedding: np.ndarray) -> int:
        center = self.get_next_center_position()
        self.centers[center] = embedding
        self.active_centers.add(center)
        return center

    def identify(
        self,
        segmentation: SlidingWindowFeature,
        embeddings: torch.Tensor
    ) -> SpeakerMap:
        embeddings = embeddings.detach().cpu().numpy()
        active_speakers = np.where(np.max(segmentation.data, axis=0) >= self.tau_active)[0]
        long_speakers = np.where(np.mean(segmentation.data, axis=0) >= self.rho_update)[0]
        num_local_speakers = segmentation.data.shape[1]

        if self.centers is None:
            self.init_centers(embeddings.shape[1])
            assignments = [
                (spk, self.add_center(embeddings[spk]))
                for spk in active_speakers
            ]
            return SpeakerMapBuilder.hard_map(
                shape=(num_local_speakers, self.max_speakers),
                assignments=assignments,
                maximize=False,
            )

        # Obtain a mapping based on distances between embeddings and centers
        dist_map = SpeakerMapBuilder.dist(embeddings, self.centers, self.metric)
        # Remove any assignments containing invalid speakers
        inactive_speakers = np.array([
            spk for spk in range(num_local_speakers)
            if spk not in active_speakers
        ])
        dist_map = dist_map.unmap_speakers(inactive_speakers, self.inactive_centers)
        # Keep assignments under the distance threshold
        valid_map = dist_map.unmap_threshold(self.delta_new)

        # Some speakers might be unidentified
        missed_speakers = [
            s for s in active_speakers
            if not valid_map.is_source_speaker_mapped(s)
        ]

        # Add assignments to new centers if possible
        new_center_speakers = []
        for spk in missed_speakers:
            has_space = len(new_center_speakers) < self.num_free_centers
            if has_space and spk in long_speakers:
                # Flag as a new center
                new_center_speakers.append(spk)
            else:
                # Cannot create a new center
                # Get global speakers in order of preference
                preferences = np.argsort(dist_map.mapping_matrix[spk, :])
                preferences = [
                    g_spk for g_spk in preferences if g_spk in self.active_centers
                ]
                # Get the free global speakers among the preferences
                _, g_assigned = valid_map.valid_assignments()
                free = [g_spk for g_spk in preferences if g_spk not in g_assigned]
                if free:
                    # The best global speaker is the closest free one
                    valid_map = valid_map.set_source_speaker(spk, free[0])

        # Update known centers
        to_update = [
            (ls, gs)
            for ls, gs in zip(*valid_map.valid_assignments())
            if ls not in missed_speakers and ls in long_speakers
        ]
        self.update(to_update, embeddings)

        # Add new centers
        for spk in new_center_speakers:
            valid_map = valid_map.set_source_speaker(
                spk, self.add_center(embeddings[spk])
            )

        return valid_map

    def __call__(self, segmentation: SlidingWindowFeature, embeddings: torch.Tensor) -> SlidingWindowFeature:
        return SlidingWindowFeature(
            self.identify(segmentation, embeddings).apply(segmentation.data),
            segmentation.sliding_window
        )


class Binarize:
    def __init__(self, uri: str, tau_active: float):
        self.uri = uri
        self._binarize = PyanBinarize(
            onset=tau_active,
            offset=tau_active,
            min_duration_on=0,
            min_duration_off=0,
        )

    def _select(
        self, scores: SlidingWindowFeature, speaker: int
    ) -> SlidingWindowFeature:
        return SlidingWindowFeature(
            scores[:, speaker].reshape(-1, 1), scores.sliding_window
        )

    def __call__(self, segmentation: SlidingWindowFeature) -> Annotation:
        annotation = Annotation(uri=self.uri, modality="speech")
        for speaker in range(segmentation.data.shape[1]):
            turns = self._binarize(self._select(segmentation, speaker))
            for speaker_turn in turns.itersegments():
                annotation[speaker_turn, speaker] = f"speaker{speaker}"
        return annotation
