import torch
import numpy as np
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.audio.utils.signal import Binarize as PyanBinarize
from pyannote.audio.pipelines.utils import PipelineModel, get_model, get_devices
from mapping import SpeakerMap, SpeakerMapBuilder
from typing import Union, Optional, List, Iterable, Tuple


class FrameWiseModel:
    def __init__(self, model: PipelineModel, device: Optional[torch.device] = None):
        self.model = get_model(model)
        self.model.eval()
        if device is None:
            device = get_devices(needs=1)[0]
        self.model.to(device)

    def __call__(self, waveform: SlidingWindowFeature) -> SlidingWindowFeature:
        with torch.no_grad():
            wave = torch.from_numpy(waveform.data.T[np.newaxis])
            output = self.model(wave.to(self.model.device)).cpu().numpy()[0]
        # Temporal resolution of the output
        resolution = self.model.introspection.frames
        # Temporal shift to keep track of current start time
        resolution = SlidingWindow(start=waveform.sliding_window.start,
                                   duration=resolution.duration,
                                   step=resolution.step)
        return SlidingWindowFeature(output, resolution)


class ChunkWiseModel:
    def __init__(self, model: PipelineModel, device: Optional[torch.device] = None):
        self.model = get_model(model)
        self.model.eval()
        if device is None:
            device = get_devices(needs=1)[0]
        self.model.to(device)

    def __call__(self, waveform: SlidingWindowFeature, weights: Optional[SlidingWindowFeature]) -> torch.Tensor:
        with torch.no_grad():
            chunk = torch.from_numpy(waveform.data.T).float()
            inputs = chunk.unsqueeze(0).to(self.model.device)
            if weights is not None:
                # weights has shape (num_local_speakers, num_frames)
                weights = torch.from_numpy(weights.data.T).float().to(self.model.device)
                inputs = inputs.repeat(weights.shape[0], 1, 1)
            # Shape (num_speakers, emb_dimension)
            output = self.model(inputs, weights=weights).cpu()
        return output


class OverlappedSpeechPenalty:
    """
    :param gamma: float, optional
        Exponent to sharpen per-frame speaker probability scores and distributions.
        Defaults to 3.
    :param beta: float, optional
        Softmax's temperature parameter (actually 1/beta) to sharpen per-frame speaker probability distributions.
        Defaults to 10.
    """
    def __init__(self, gamma: float = 3, beta: float = 10):
        self.gamma = gamma
        self.beta = beta

    def __call__(self, segmentation: SlidingWindowFeature) -> SlidingWindowFeature:
        weights = torch.from_numpy(segmentation.data).float().T
        with torch.no_grad():
            probs = torch.softmax(self.beta * weights, dim=0)
            weights = torch.pow(weights, self.gamma) * torch.pow(probs, self.gamma)
            weights[weights < 1e-8] = 1e-8
        return SlidingWindowFeature(weights.T.numpy(), segmentation.sliding_window)


class EmbeddingNormalization:
    def __init__(self, norm: Union[float, torch.Tensor] = 1):
        self.norm = norm

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        if isinstance(self.norm, torch.Tensor):
            assert self.norm.shape[0] == embeddings.shape[0]
        with torch.no_grad():
            norm_embs = self.norm * embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return norm_embs


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
