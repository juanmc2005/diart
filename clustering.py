import torch
from typing import Optional, List, Iterable, Tuple
import numpy as np
from pyannote.audio.core.online import SpeakerMap, SpeakerMapBuilder


class OnlineSpeakerClustering:
    def __init__(
        self,
        max_distance: float,
        num_local_speakers: int,
        metric: Optional[str] = "cosine",
        max_speakers: int = 20
    ):
        self.max_distance = max_distance
        self.num_local_speakers = num_local_speakers
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
        embeddings: torch.Tensor,
        active_speakers: np.ndarray,
        long_speakers: np.ndarray
    ) -> SpeakerMap:
        embeddings = embeddings.detach().cpu().numpy()
        embedding_dim = embeddings.shape[1]

        if self.centers is None:
            self.init_centers(embedding_dim)
            assignments = [
                (spk, self.add_center(embeddings[spk]))
                for spk in active_speakers
            ]
            return SpeakerMapBuilder.hard_map(
                shape=(self.num_local_speakers, self.max_speakers),
                assignments=assignments,
                maximize=False,
            )

        # Obtain a mapping based on distances between embeddings and centers
        dist_map = SpeakerMapBuilder.dist(embeddings, self.centers, self.metric)
        # Remove any assignments containing invalid speakers
        inactive_speakers = np.array([
            spk for spk in range(self.num_local_speakers)
            if spk not in active_speakers
        ])
        dist_map = dist_map.unmap_speakers(inactive_speakers, self.inactive_centers)
        # Keep assignments under the distance threshold
        valid_map = dist_map.unmap_threshold(self.max_distance)

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
