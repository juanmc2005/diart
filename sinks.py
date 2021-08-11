import numpy as np
from rx.core import Observer
from pyannote.core import Annotation, Segment, SlidingWindowFeature, notebook
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from typing import Literal, Union, Text, Optional, Tuple
from pathlib import Path
from traceback import print_exc
import matplotlib.pyplot as plt


class OutputBuilder(Observer):
    def __init__(
        self,
        output_path: Optional[Union[Path, Text]] = None,
        merge_collar: float = 0.05,
        visualization: Literal["slide", "accumulate"] = "slide",
        reference: Optional[Union[Path, Text]] = None,
    ):
        super().__init__()
        assert visualization in ["slide", "accumulate"]
        self.collar = merge_collar
        self.visualization = visualization
        self.output_path = output_path
        self.reference = reference
        if self.reference is not None:
            self.reference = load_rttm(reference)
            uri = list(self.reference.keys())[0]
            self.reference = self.reference[uri]
        self.output: Optional[Annotation] = None
        self.waveform: Optional[SlidingWindowFeature] = None
        self.window_duration: Optional[float] = None
        self.figure, self.axs, self.num_axs = None, None, -1

    def init_num_axs(self):
        if self.num_axs == -1:
            self.num_axs = 1
            if self.waveform is not None:
                self.num_axs += 1
            if self.reference is not None:
                self.num_axs += 1

    def init_figure(self):
        self.init_num_axs()
        self.figure, self.axs = plt.subplots(self.num_axs, 1, figsize=(10, 2 * self.num_axs))

    def draw(self):
        # Initialize figure if first call
        if self.figure is None:
            self.init_figure()

        # Clear all axs
        for i in range(self.num_axs):
            self.axs[i].clear()

        # Determine plot bounds
        output_extent = self.output.get_timeline().extent()
        if self.visualization == "slide":
            start_time = output_extent.end - self.window_duration
            notebook.crop = Segment(start_time, output_extent.end)
        else:
            notebook.crop = output_extent

        # Plot internal state
        if self.reference is not None:
            metric = DiarizationErrorRate()
            mapping = metric.optimal_mapping(self.reference, self.output)
            self.output.rename_labels(mapping=mapping, copy=False)
        notebook.plot_annotation(self.output, self.axs[0])
        self.axs[0].set_ylabel("Output")
        if self.num_axs == 2:
            if self.waveform is not None:
                notebook.plot_feature(self.waveform, self.axs[1])
                self.axs[1].set_ylabel("Audio")
            elif self.reference is not None:
                notebook.plot_annotation(self.reference, self.axs[1])
                self.axs[1].set_ylabel("Reference")
        elif self.num_axs == 3:
            notebook.plot_feature(self.waveform, self.axs[1])
            self.axs[1].set_ylabel("Audio")
            notebook.plot_annotation(self.reference, self.axs[2])
            self.axs[2].set_ylabel("Reference")

        # Draw
        plt.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.05)

    def on_next(self, value: Union[Annotation, Tuple[Annotation, SlidingWindowFeature]]):
        if isinstance(value, Annotation):
            annotation, waveform = value, None
        else:
            annotation, waveform = value

        # Update output annotation
        if self.output is None:
            self.output = annotation
            self.window_duration = annotation.get_timeline().extent().duration
        else:
            self.output = self.output.update(annotation).support(self.collar)

        # Update waveform
        if waveform is not None:
            if self.waveform is None:
                self.waveform = waveform
            else:
                new_samples = np.concatenate([self.waveform.data, waveform.data], axis=0)
                self.waveform = SlidingWindowFeature(new_samples, self.waveform.sliding_window)

        # Draw new output
        self.draw()

        # Save RTTM if possible
        if self.output_path is not None:
            with open(Path(self.output_path), 'w') as file:
                self.output.write_rttm(file)

    def on_error(self, error: Exception):
        print_exc()
        exit(1)
