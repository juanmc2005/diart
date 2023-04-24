import re
from pathlib import Path
from typing import Union, Text, Optional, Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import rich
from pyannote.core import Annotation, Segment, SlidingWindowFeature, SlidingWindow, notebook
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer


class WindowClosedException(Exception):
    pass


def _extract_prediction(value: Union[Tuple, Any]) -> Any:
    if isinstance(value, tuple):
        return value[0]
    return value


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
            with open(self.path, 'w') as file:
                annotation.support(self.patch_collar).write_rttm(file)

    def on_next(self, value: Union[Tuple, Annotation]):
        prediction = _extract_prediction(value)
        # Write prediction in RTTM format
        prediction.uri = self.uri
        with open(self.path, 'a') as file:
            prediction.write_rttm(file)

    def on_error(self, error: Exception):
        self.patch()

    def on_completed(self):
        self.patch()


class TextWriter(Observer):
    def __init__(self, path: Union[Path, Text]):
        super().__init__()
        self.path = Path(path).expanduser()
        if self.path.exists():
            self.path.unlink()

    def on_next(self, value: Union[Tuple, Text]):
        # Write transcription to file
        prediction = _extract_prediction(value)
        with open(self.path, 'a') as file:
            file.write(prediction + "\n")


class DiarizationAccumulator(Observer):
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


class RichScreen(Observer):
    def __init__(self, speaker_colors: Optional[List[Text]] = None):
        super().__init__()
        self.colors = speaker_colors
        if self.colors is None:
            self.colors = [
                "bright_red", "bright_blue", "bright_green", "orange3", "deep_pink1",
                "yellow2", "magenta", "cyan", "bright_magenta", "dodger_blue2"
            ]
        self.num_colors = len(self.colors)

    def on_next(self, value: Union[Tuple, Text]):
        prediction = _extract_prediction(value)
        # Extract speakers
        speakers = sorted(re.findall(r'\[.*?]', prediction))
        # Colorize based on speakers
        colorized = prediction
        for i, speaker in enumerate(speakers):
            colorized = colorized.replace(speaker, f"[{self.colors[i % self.num_colors]}]")
        # Print result
        rich.print(colorized)


class StreamingPlot(Observer):
    def __init__(
        self,
        duration: float,
        step: float,
        latency: float,
        sample_rate: float,
        reference: Optional[Union[Path, Text]] = None,
        patch_collar: float = 0.05,
    ):
        super().__init__()
        self.reference = reference
        if self.reference is not None:
            self.reference = list(load_rttm(reference).values())[0]
        self.window_duration = duration
        self.window_step = step
        self.latency = latency
        self.sample_rate = sample_rate
        self.patch_collar = patch_collar

        self.num_window_samples = int(np.rint(self.window_duration * self.sample_rate))
        self.num_step_samples = int(np.rint(self.window_step * self.sample_rate))
        self.audio_resolution = 1 / self.sample_rate

        self.figure, self.axs, self.num_axs = None, None, -1
        # This flag allows to catch the matplotlib window closed event and make the next call stop iterating
        self.window_closed = False
        self.real_time = 0
        self.pred_buffer, self.audio_buffer = None, None
        self.next_sample = 0

    def _on_window_closed(self, event):
        self.window_closed = True

    def _init_num_axs(self):
        if self.num_axs == -1:
            self.num_axs = 2
            if self.reference is not None:
                self.num_axs += 1

    def _init_figure(self):
        self._init_num_axs()
        self.figure, self.axs = plt.subplots(self.num_axs, 1, figsize=(10, 2 * self.num_axs))
        if self.num_axs == 1:
            self.axs = [self.axs]
        self.figure.canvas.mpl_connect('close_event', self._on_window_closed)

    def _clear_axs(self):
        for i in range(self.num_axs):
            self.axs[i].clear()

    def get_plot_bounds(self) -> Segment:
        end_time = self.real_time - self.latency
        start_time = max(0., end_time - self.window_duration)
        return Segment(start_time, end_time)

    def on_next(
        self,
        values: Tuple[Annotation, SlidingWindowFeature]
    ):
        if self.window_closed:
            raise WindowClosedException

        prediction, waveform = values

        # TODO break this aggregation code into methods

        # Determine the real time of the stream and the start time of the buffer
        self.real_time = waveform.extent.end
        start_time = max(0., self.real_time - self.latency - self.window_duration)

        # Update prediction buffer and constrain its bounds
        if self.pred_buffer is None:
            self.pred_buffer = prediction
        else:
            self.pred_buffer = self.pred_buffer.update(prediction)
            self.pred_buffer = self.pred_buffer.support(self.patch_collar)
            if start_time > 0:
                self.pred_buffer = self.pred_buffer.extrude(Segment(0, start_time))

        # Update the audio buffer if there's audio in the input
        new_next_sample = self.next_sample + self.num_step_samples
        if self.audio_buffer is None:
            # Determine the size of the first chunk
            expected_duration = self.window_duration + self.window_step - self.latency
            expected_samples = int(np.rint(expected_duration * self.sample_rate))
            # Shift indicator to start copying new audio in the buffer
            new_next_sample = self.next_sample + expected_samples
            # Buffer size is duration + step
            new_buffer = np.zeros((self.num_window_samples + self.num_step_samples, 1))
            # Copy first chunk into buffer (slicing because of rounding errors)
            new_buffer[:expected_samples] = waveform.data[:expected_samples]
        elif self.next_sample <= self.num_window_samples:
            # The buffer isn't full, copy into next free buffer chunk
            new_buffer = self.audio_buffer.data
            new_buffer[self.next_sample:new_next_sample] = waveform.data
        else:
            # The buffer is full, shift values to the left and copy into last buffer chunk
            new_buffer = np.roll(self.audio_buffer.data, -self.num_step_samples, axis=0)
            # If running on a file, the online prediction may be shorter depending on the latency
            # The remaining audio at the end is appended, so 'waveform' may be longer than 'num_step_samples'
            # In that case, we simply ignore the appended samples.
            new_buffer[-self.num_step_samples:] = waveform.data[:self.num_step_samples]

        # Wrap waveform in a sliding window feature to include timestamps
        window = SlidingWindow(start=start_time, duration=self.audio_resolution, step=self.audio_resolution)
        self.audio_buffer = SlidingWindowFeature(new_buffer, window)
        self.next_sample = new_next_sample

        # Initialize figure if first call
        if self.figure is None:
            self._init_figure()
        # Clear previous plots
        self._clear_axs()
        # Set plot bounds
        notebook.crop = self.get_plot_bounds()

        # Align prediction and reference if possible
        if self.reference is not None:
            metric = DiarizationErrorRate()
            mapping = metric.optimal_mapping(self.reference, self.pred_buffer)
            self.pred_buffer.rename_labels(mapping=mapping, copy=False)

        # Plot prediction
        notebook.plot_annotation(self.pred_buffer, self.axs[0])
        self.axs[0].set_title("Output")

        # Plot waveform
        notebook.plot_feature(self.audio_buffer, self.axs[1])
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
