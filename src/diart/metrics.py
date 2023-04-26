from pathlib import Path
from typing import Text, Any, List, Union

import pandas as pd
from pyannote.core import Annotation
from pyannote.metrics import diarization as dia, detection as det
from pyannote.metrics.base import BaseMetric as PyannoteBaseMetric
from pyannote.database.util import load_rttm
from torchmetrics import text


class Metric:
    @property
    def name(self) -> Text:
        raise NotImplementedError

    def __call__(self, reference: Any, prediction: Any) -> float:
        raise NotImplementedError

    def report(self, uris: List[Text], display: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    def load_reference(self, filepath: Union[Text, Path]) -> Any:
        raise NotImplementedError


class PyannoteMetric(Metric):
    def __init__(self, metric: PyannoteBaseMetric):
        self._metric = metric

    @property
    def name(self) -> Text:
        return self._metric.name

    def __call__(self, reference: Annotation, prediction: Annotation) -> float:
        return self._metric(reference, prediction)

    def report(self, uris: List[Text], display: bool = False) -> pd.DataFrame:
        return self._metric.report(display)

    def load_reference(self, filepath: Union[Text, Path]) -> Annotation:
        return load_rttm(filepath).popitem()[1]


class DiarizationErrorRate(PyannoteMetric):
    def __init__(self, collar: float = 0, skip_overlap: bool = False):
        super().__init__(dia.DiarizationErrorRate(collar, skip_overlap))


class DetectionErrorRate(PyannoteMetric):
    def __init__(self, collar: float = 0, skip_overlap: bool = False):
        super().__init__(det.DetectionErrorRate(collar, skip_overlap))


class WordErrorRate(Metric):
    def __init__(self, unify_case: bool = False):
        self.unify_case = unify_case
        self._metric = text.WordErrorRate()
        self._values = []

    @property
    def name(self) -> Text:
        return "word error rate"

    def __call__(self, reference: Text, prediction: Text) -> float:
        if self.unify_case:
            prediction = prediction.lower()
            reference = reference.lower()
        # Torchmetrics requires predictions first, then reference
        value = self._metric(prediction, reference).item()
        self._values.append(value)
        return value

    def report(self, uris: List[Text], display: bool = False) -> pd.DataFrame:
        num_uris, num_values = len(uris), len(self._values)
        msg = f"URI list size must match values. Found {num_uris} but expected {num_values}"
        assert num_uris == num_values, msg

        rows = self._values + [self._metric.compute().item()]
        index = uris + ["TOTAL"]
        report = pd.DataFrame(rows, index=index, columns=[self.name])

        if display:
            print(report.to_string(
                index=True,
                sparsify=False,
                justify="right",
                float_format=lambda f: "{0:.2f}".format(f),
            ))

        return report

    def load_reference(self, filepath: Union[Text, Path]) -> Text:
        with open(filepath, "r") as file:
            lines = [line.strip() for line in file.readlines()]
        return " ".join(lines)
