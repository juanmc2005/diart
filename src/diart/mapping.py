from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Text, Tuple, Union

import numpy as np
from pyannote.core.utils.distance import cdist
from scipy.optimize import linear_sum_assignment


class MappingMatrixObjective:
    def invalid_tensor(self, shape: Union[Tuple, int]) -> np.ndarray:
        return np.ones(shape) * self.invalid_value

    def optimal_assignments(self, matrix: np.ndarray) -> List[int]:
        return list(linear_sum_assignment(matrix, self.maximize)[1])

    def mapped_indices(self, matrix: np.ndarray, axis: int) -> List[int]:
        # Entries full of invalid_value are not mapped
        best_values = self.best_value_fn(matrix, axis=axis)
        return list(np.where(best_values != self.invalid_value)[0])

    def hard_speaker_map(
        self, num_src: int, num_tgt: int, assignments: Iterable[Tuple[int, int]]
    ) -> SpeakerMap:
        mapping_matrix = self.invalid_tensor(shape=(num_src, num_tgt))
        for src, tgt in assignments:
            mapping_matrix[src, tgt] = self.best_possible_value
        return SpeakerMap(mapping_matrix, self)

    @property
    def invalid_value(self) -> float:
        # linear_sum_assignment cannot deal with np.inf,
        # which would be ideal. Using a big number instead.
        return -1e10 if self.maximize else 1e10

    @property
    def maximize(self) -> bool:
        raise NotImplementedError()

    @property
    def best_possible_value(self) -> float:
        raise NotImplementedError()

    @property
    def best_value_fn(self) -> Callable:
        raise NotImplementedError()


class MinimizationObjective(MappingMatrixObjective):
    @property
    def maximize(self) -> bool:
        return False

    @property
    def best_possible_value(self) -> float:
        return 0

    @property
    def best_value_fn(self) -> Callable:
        return np.min


class MaximizationObjective(MappingMatrixObjective):
    def __init__(self, max_value: float = 1):
        self.max_value = max_value

    @property
    def maximize(self) -> bool:
        return True

    @property
    def best_possible_value(self) -> float:
        return self.max_value

    @property
    def best_value_fn(self) -> Callable:
        return np.max


class SpeakerMapBuilder:
    @staticmethod
    def hard_map(
        shape: Tuple[int, int], assignments: Iterable[Tuple[int, int]], maximize: bool
    ) -> SpeakerMap:
        num_src, num_tgt = shape
        objective = MaximizationObjective if maximize else MinimizationObjective
        return objective().hard_speaker_map(num_src, num_tgt, assignments)

    @staticmethod
    def correlation(scores1: np.ndarray, scores2: np.ndarray) -> SpeakerMap:
        score_matrix_per_frame = (
            np.stack(  # (local_speakers, num_frames, global_speakers)
                [
                    scores1[:, speaker : speaker + 1] * scores2
                    for speaker in range(scores1.shape[1])
                ],
                axis=0,
            )
        )
        # Calculate total speech "activations" per local speaker
        local_speech_scores = np.sum(scores1, axis=0).reshape(-1, 1)
        # Calculate speaker mapping matrix
        # Cost matrix is the correlation divided by sum of local activations
        score_matrix = np.sum(score_matrix_per_frame, axis=1) / local_speech_scores
        # We want to maximize the correlation to calculate optimal speaker alignments
        return SpeakerMap(score_matrix, MaximizationObjective(max_value=1))

    @staticmethod
    def mse(scores1: np.ndarray, scores2: np.ndarray) -> SpeakerMap:
        cost_matrix = np.stack(  # (local_speakers, local_speakers)
            [
                np.square(scores1[:, speaker : speaker + 1] - scores2).mean(axis=0)
                for speaker in range(scores1.shape[1])
            ],
            axis=0,
        )
        # We want to minimize the MSE to calculate optimal speaker alignments
        return SpeakerMap(cost_matrix, MinimizationObjective())

    @staticmethod
    def mae(scores1: np.ndarray, scores2: np.ndarray) -> SpeakerMap:
        cost_matrix = np.stack(  # (local_speakers, local_speakers)
            [
                np.abs(scores1[:, speaker : speaker + 1] - scores2).mean(axis=0)
                for speaker in range(scores1.shape[1])
            ],
            axis=0,
        )
        # We want to minimize the MSE to calculate optimal speaker alignments
        return SpeakerMap(cost_matrix, MinimizationObjective())

    @staticmethod
    def dist(
        embeddings1: np.ndarray, embeddings2: np.ndarray, metric: Text = "cosine"
    ) -> SpeakerMap:
        # We want to minimize the distance to calculate optimal speaker alignments
        dist_matrix = cdist(embeddings1, embeddings2, metric=metric)
        return SpeakerMap(dist_matrix, MinimizationObjective())

    @staticmethod
    def clf_output(predictions: np.ndarray, pad_to: Optional[int] = None) -> SpeakerMap:
        """
        Parameters
        ----------
        predictions : np.ndarray, (num_local_speakers, num_global_speakers)
            Probability outputs of a speaker embedding classifier
        pad_to : int, optional
            Pad num_global_speakers to this value.
            Useful to deal with unknown speakers that may appear in the future.
            Defaults to no padding
        """
        num_locals, num_globals = predictions.shape
        objective = MaximizationObjective(max_value=1)
        if pad_to is not None and num_globals < pad_to:
            padding = np.ones((num_locals, pad_to - num_globals))
            padding = objective.invalid_value * padding
            predictions = np.concatenate([predictions, padding], axis=1)
        return SpeakerMap(predictions, objective)


class SpeakerMap:
    def __init__(self, mapping_matrix: np.ndarray, objective: MappingMatrixObjective):
        self.mapping_matrix = mapping_matrix
        self.objective = objective
        self.num_source_speakers = self.mapping_matrix.shape[0]
        self.num_target_speakers = self.mapping_matrix.shape[1]
        self.mapped_source_speakers = self.objective.mapped_indices(
            self.mapping_matrix, axis=1
        )
        self.mapped_target_speakers = self.objective.mapped_indices(
            self.mapping_matrix, axis=0
        )
        self._opt_assignments: Optional[List[int]] = None

    @property
    def _raw_optimal_assignments(self) -> List[int]:
        if self._opt_assignments is None:
            self._opt_assignments = self.objective.optimal_assignments(
                self.mapping_matrix
            )
        return self._opt_assignments

    @property
    def shape(self) -> Tuple[int, int]:
        return self.mapping_matrix.shape

    def __len__(self):
        return len(self.mapped_source_speakers)

    def __add__(self, other: SpeakerMap) -> SpeakerMap:
        return self.union(other)

    def _strict_check_valid(self, src: int, tgt: int) -> bool:
        return self.mapping_matrix[src, tgt] != self.objective.invalid_value

    def _loose_check_valid(self, src: int, tgt: int) -> bool:
        return self.is_source_speaker_mapped(src)

    def valid_assignments(
        self,
        strict: bool = False,
        as_array: bool = False,
    ) -> Union[Tuple[List[int], List[int]], Tuple[np.ndarray, np.ndarray]]:
        source, target = [], []
        val_type = "strict" if strict else "loose"
        is_valid = getattr(self, f"_{val_type}_check_valid")
        for src, tgt in enumerate(self._raw_optimal_assignments):
            if is_valid(src, tgt):
                source.append(src)
                target.append(tgt)
        if as_array:
            source, target = np.array(source), np.array(target)
        return source, target

    def is_source_speaker_mapped(self, source_speaker: int) -> bool:
        return source_speaker in self.mapped_source_speakers

    def is_target_speaker_mapped(self, target_speaker: int) -> bool:
        return target_speaker in self.mapped_target_speakers

    def set_source_speaker(self, src_speaker, tgt_speaker: int):
        # if not force:
        #     assert not self.is_source_speaker_mapped(src_speaker)
        #     assert not self.is_target_speaker_mapped(tgt_speaker)
        new_cost_matrix = np.copy(self.mapping_matrix)
        new_cost_matrix[src_speaker, tgt_speaker] = self.objective.best_possible_value
        return SpeakerMap(new_cost_matrix, self.objective)

    def unmap_source_speaker(self, src_speaker: int):
        new_cost_matrix = np.copy(self.mapping_matrix)
        new_cost_matrix[src_speaker] = self.objective.invalid_tensor(
            shape=self.num_target_speakers
        )
        return SpeakerMap(new_cost_matrix, self.objective)

    def unmap_threshold(self, threshold: float) -> SpeakerMap:
        def is_invalid(val):
            if self.objective.maximize:
                return val <= threshold
            else:
                return val >= threshold

        return self.unmap_speakers(
            [
                src
                for src, tgt in zip(*self.valid_assignments())
                if is_invalid(self.mapping_matrix[src, tgt])
            ]
        )

    def unmap_speakers(
        self,
        source_speakers: Optional[Union[List[int], np.ndarray]] = None,
        target_speakers: Optional[Union[List[int], np.ndarray]] = None,
    ) -> SpeakerMap:
        # Set invalid_value to disabled speakers.
        # If they happen to be the best mapping for a local speaker,
        # it means that the mapping of the local speaker should be ignored.
        source_speakers = [] if source_speakers is None else source_speakers
        target_speakers = [] if target_speakers is None else target_speakers
        new_cost_matrix = np.copy(self.mapping_matrix)
        for speaker1 in source_speakers:
            new_cost_matrix[speaker1] = self.objective.invalid_tensor(
                shape=self.num_target_speakers
            )
        for speaker2 in target_speakers:
            new_cost_matrix[:, speaker2] = self.objective.invalid_tensor(
                shape=self.num_source_speakers
            )
        return SpeakerMap(new_cost_matrix, self.objective)

    def compose(self, other: SpeakerMap) -> SpeakerMap:
        """Let's say that `self` is a mapping of `source_speakers` to `intermediate_speakers`
        and `other` is a mapping from `intermediate_speakers` to `target_speakers`.

        Compose `self` with `other` to obtain a new mapping from `source_speakers` to `target_speakers`.
        """
        new_cost_matrix = other.objective.invalid_tensor(
            shape=(self.num_source_speakers, other.num_target_speakers)
        )
        for src_speaker, intermediate_speaker in zip(*self.valid_assignments()):
            target_speaker = other.mapping_matrix[intermediate_speaker]
            new_cost_matrix[src_speaker] = target_speaker
        return SpeakerMap(new_cost_matrix, other.objective)

    def union(self, other: SpeakerMap):
        """`self` and `other` are two maps with the same dimensions.
        Return a new hard speaker map containing assignments in both maps.

        An assignment from `other` is ignored if it is in conflict with
        a source or target speaker from `self`.

        WARNING: The resulting map doesn't preserve soft assignments
        because `self` and `other` might have different objectives.

        :param other: SpeakerMap
            Another speaker map
        """
        assert self.shape == other.shape
        best_val = self.objective.best_possible_value
        new_cost_matrix = self.objective.invalid_tensor(self.shape)
        self_src, self_tgt = self.valid_assignments()
        other_src, other_tgt = other.valid_assignments()
        for src in range(self.num_source_speakers):
            if src in self_src:
                # `self` is preserved by default
                tgt = self_tgt[self_src.index(src)]
                new_cost_matrix[src, tgt] = best_val
            elif src in other_src:
                # In order to add an assignment from `other`,
                # the target speaker cannot be in conflict with `self`
                tgt = other_tgt[other_src.index(src)]
                if not self.is_target_speaker_mapped(tgt):
                    new_cost_matrix[src, tgt] = best_val
        return SpeakerMap(new_cost_matrix, self.objective)

    def apply(self, source_scores: np.ndarray) -> np.ndarray:
        """Apply this mapping to a score matrix of source speakers
        to obtain the same scores aligned to target speakers.

        Parameters
        ----------
        source_scores : SlidingWindowFeature, (num_frames, num_source_speakers)
            Source speaker scores per frame.

        Returns
        -------
        projected_scores : SlidingWindowFeature, (num_frames, num_target_speakers)
            Score matrix for target speakers.
        """
        # Map local speaker scores to the most probable global speaker. Unknown scores are set to 0
        num_frames = source_scores.data.shape[0]
        projected_scores = np.zeros((num_frames, self.num_target_speakers))
        for src_speaker, tgt_speaker in zip(*self.valid_assignments()):
            projected_scores[:, tgt_speaker] = source_scores[:, src_speaker]
        return projected_scores
