from pathlib import Path
from typing import Union, Text, Optional, Tuple

import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment, SlidingWindowFeature, notebook
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer
from typing_extensions import Literal

from . import utils


class WindowClosedException(Exception):
    pass


def _extract_prediction(value: Union[Tuple, Annotation]) -> Annotation:
    if isinstance(value, tuple):
        return value[0]
    if isinstance(value, Annotation):
        return value
    msg = f"Expected tuple or Annotation, but got {type(value)}"
    raise ValueError(msg)


class RTTMWriter(Observer):
    def __init__(self, uri: Text, path: Union[Path, Text], patch_collar: float = 0.05):
        super().__init__()
        self.uri = uri
        self.patch_collar = patch_collar
        self.path = Path(path).expanduser()
        if self.path.exists():
            self.path.unlink()

    def patch(self):
        """Stitch same-speaker turns that are close to each other"""
        if not self.path.exists():
            return
        annotations = list(load_rttm(self.path).values())
        if annotations:
            annotation = annotations[0]
            annotation.uri = self.uri
            with open(self.path, "w") as file:
                annotation.support(self.patch_collar).write_rttm(file)

    def on_next(self, value: Union[Tuple, Annotation]):
        prediction = _extract_prediction(value)
        # Write prediction in RTTM format
        prediction.uri = self.uri
        with open(self.path, "a") as file:
            prediction.write_rttm(file)

    def on_error(self, error: Exception):
        self.patch()

    def on_completed(self):
        self.patch()


class PredictionAccumulator(Observer):
    def __init__(self, uri: Optional[Text] = None, patch_collar: float = 0.05):
        super().__init__()
        self.uri = uri
        self.patch_collar = patch_collar
        self._prediction: Optional[Annotation] = None

    def patch(self):
        """Stitch same-speaker turns that are close to each other"""
        if self._prediction is not None:
            self._prediction = self._prediction.support(self.patch_collar)

    def get_prediction(self) -> Annotation:
        # Patch again in case this is called before on_completed
        self.patch()
        return self._prediction

    def on_next(self, value: Union[Tuple, Annotation]):
        prediction = _extract_prediction(value)
        prediction.uri = self.uri
        if self._prediction is None:
            self._prediction = prediction
        else:
            self._prediction.update(prediction)

    def on_error(self, error: Exception):
        self.patch()

    def on_completed(self):
        self.patch()


class StreamingPlot(Observer):
    def __init__(
        self,
        duration: float,
        latency: float,
        visualization: Literal["slide", "accumulate"] = "slide",
        reference: Optional[Union[Path, Text]] = None,
    ):
        super().__init__()
        assert visualization in ["slide", "accumulate"]
        self.visualization = visualization
        self.reference = reference
        if self.reference is not None:
            self.reference = list(load_rttm(reference).values())[0]
        self.window_duration = duration
        self.latency = latency
        self.figure, self.axs, self.num_axs = None, None, -1
        # This flag allows to catch the matplotlib window closed event and make the next call stop iterating
        self.window_closed = False

    def _on_window_closed(self, event):
        self.window_closed = True

    def _init_num_axs(self):
        if self.num_axs == -1:
            self.num_axs = 2
            if self.reference is not None:
                self.num_axs += 1

    def _init_figure(self):
        self._init_num_axs()
        self.figure, self.axs = plt.subplots(
            self.num_axs, 1, figsize=(10, 2 * self.num_axs)
        )
        if self.num_axs == 1:
            self.axs = [self.axs]
        self.figure.canvas.mpl_connect("close_event", self._on_window_closed)

    def _clear_axs(self):
        for i in range(self.num_axs):
            self.axs[i].clear()

    def get_plot_bounds(self, real_time: float) -> Segment:
        start_time = 0
        end_time = real_time - self.latency
        if self.visualization == "slide":
            start_time = max(0.0, end_time - self.window_duration)
        return Segment(start_time, end_time)

    def on_next(self, values: Tuple[Annotation, SlidingWindowFeature, float]):
        if self.window_closed:
            raise WindowClosedException

        prediction, waveform, real_time = values

        # Initialize figure if first call
        if self.figure is None:
            self._init_figure()
        # Clear previous plots
        self._clear_axs()
        # Set plot bounds
        notebook.crop = self.get_plot_bounds(real_time)

        # Align prediction and reference if possible
        if self.reference is not None:
            metric = DiarizationErrorRate()
            mapping = metric.optimal_mapping(self.reference, prediction)
            prediction.rename_labels(mapping=mapping, copy=False)

        # Plot prediction
        notebook.plot_annotation(prediction, self.axs[0])
        self.axs[0].set_title("Output")

        # Plot waveform
        notebook.plot_feature(waveform, self.axs[1])
        self.axs[1].set_title("Audio")

        # Plot reference if available
        if self.num_axs == 3:
            notebook.plot_annotation(self.reference, self.axs[2])
            self.axs[2].set_title("Reference")

        # Draw
        plt.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.05)
