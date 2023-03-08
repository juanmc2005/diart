import time
from typing import Optional, Text, Union

import matplotlib.pyplot as plt
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature, notebook
from rich.progress import Progress, TaskID


class ProgressBar:
    def __init__(
        self,
        bar: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
        default_description: Optional[Text] = None,
    ):
        self.bar = bar
        self.task_id = task_id
        self._task_id_given = False

        self.default_description = default_description
        if self.default_description is None:
            self.default_description = "[cyan]Streaming"

        if self.bar is None:
            self.bar = Progress()
            if self.task_id is not None:
                msg = "WARNING: ProgressBar was given a task ID without a progress bar, ignoring..."
                print(msg)
                self.task_id = None
        elif self.task_id is not None:
            # Both progress bar and task id were provided, not closing because of parallel bars
            self._task_id_given = True

        self.bar.start()

    def create(self, total: int, description: Optional[Text] = None, **kwargs):
        if self.task_id is None:
            desc = self.default_description if description is None else description
            self.task_id = self.bar.add_task(
                desc,
                start=False,
                total=total,
                completed=0,
                visible=True,
                **kwargs
            )

    def start(self):
        assert self.task_id is not None
        self.bar.start_task(self.task_id)

    def update(self, n: int = 1):
        assert self.task_id is not None
        self.bar.update(self.task_id, advance=n)

    def stop(self):
        assert self.task_id is not None
        self.bar.stop_task(self.task_id)

    def close(self):
        if not self._task_id_given:
            self.bar.stop()


class ParallelProgressBars:
    def __init__(self, leave: bool = True):
        self.progress = Progress(transient=not leave)

    def add_bar(self, description: Text):
        task_id = self.progress.add_task(description, start=False)
        return ProgressBar(self.progress, task_id)

    def close(self):
        self.progress.stop()


class Chronometer:
    def __init__(self, unit: Text):
        self.unit = unit
        self.current_start_time = None
        self.history = []

    @property
    def is_running(self):
        return self.current_start_time is not None

    def start(self):
        self.current_start_time = time.monotonic()

    def stop(self, do_count: bool = True):
        msg = "No start time available, Did you call stop() before start()?"
        assert self.current_start_time is not None, msg
        end_time = time.monotonic() - self.current_start_time
        self.current_start_time = None
        if do_count:
            self.history.append(end_time)

    def report(self):
        print(
            f"Took {np.mean(self.history).item():.3f} "
            f"(+/-{np.std(self.history).item():.3f}) seconds/{self.unit} "
            f"-- ran {len(self.history)} times"
        )


def parse_hf_token_arg(hf_token: Text) -> Union[bool, Text]:
    if hf_token.lower() == "true":
        return True
    elif hf_token.lower() == "false":
        return False
    return hf_token


def visualize_feature(duration: Optional[float] = None):
    def apply(feature: SlidingWindowFeature):
        if duration is None:
            notebook.crop = feature.extent
        else:
            notebook.crop = Segment(feature.extent.end - duration, feature.extent.end)
        plt.rcParams["figure.figsize"] = (8, 2)
        notebook.plot_feature(feature)
        plt.tight_layout()
        plt.show()
    return apply


def visualize_annotation(duration: Optional[float] = None):
    def apply(annotation: Annotation):
        extent = annotation.get_timeline().extent()
        if duration is None:
            notebook.crop = extent
        else:
            notebook.crop = Segment(extent.end - duration, extent.end)
        plt.rcParams["figure.figsize"] = (8, 2)
        notebook.plot_annotation(annotation)
        plt.tight_layout()
        plt.show()
    return apply
