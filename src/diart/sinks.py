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
        duration: float,
        step: float,
        latency: float,
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
        self.window_duration: float = duration
        self.step = step
        self.latency = latency
        self.real_time = 0
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
        if self.num_axs == 1:
            self.axs = [self.axs]

    def draw(self):
        # Initialize figure if first call
        if self.figure is None:
            self.init_figure()

        # Clear all axs
        for i in range(self.num_axs):
            self.axs[i].clear()

        # Determine plot bounds
        start_time = 0
        end_time = self.real_time - self.latency
        if self.visualization == "slide":
            start_time = max(0., end_time - self.window_duration)
        notebook.crop = Segment(start_time, end_time)

        # Plot internal state
        if self.reference is not None:
            metric = DiarizationErrorRate()
            mapping = metric.optimal_mapping(self.reference, self.output)
            self.output.rename_labels(mapping=mapping, copy=False)
        notebook.plot_annotation(self.output, self.axs[0])
        self.axs[0].set_title("Output")
        if self.num_axs == 2:
            if self.waveform is not None:
                notebook.plot_feature(self.waveform, self.axs[1])
                self.axs[1].set_title("Audio")
            elif self.reference is not None:
                notebook.plot_annotation(self.reference, self.axs[1])
                self.axs[1].set_title("Reference")
        elif self.num_axs == 3:
            notebook.plot_feature(self.waveform, self.axs[1])
            self.axs[1].set_title("Audio")
            notebook.plot_annotation(self.reference, self.axs[2])
            self.axs[2].set_title("Reference")

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
            self.real_time = self.window_duration
        else:
            self.output = self.output.update(annotation).support(self.collar)
            self.real_time += self.step

        # Update waveform
        if waveform is not None:
            if self.waveform is None:
                self.waveform = waveform
            else:
                # FIXME time complexity can be better with pre-allocation of a numpy array
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

    def on_completed(self):
        print("Stream completed")
