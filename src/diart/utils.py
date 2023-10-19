import base64
import time
from typing import Optional, Text, Union

import matplotlib.pyplot as plt
import numpy as np
from pyannote.core import Annotation, Segment, SlidingWindowFeature, notebook

from . import blocks
from .progress import ProgressBar


class Chronometer:
    def __init__(self, unit: Text, progress_bar: Optional[ProgressBar] = None):
        self.unit = unit
        self.progress_bar = progress_bar
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
        print_fn = print
        if self.progress_bar is not None:
            print_fn = self.progress_bar.write
        print_fn(
            f"Took {np.mean(self.history).item():.3f} "
            f"(+/-{np.std(self.history).item():.3f}) seconds/{self.unit} "
            f"-- ran {len(self.history)} times"
        )


def parse_hf_token_arg(hf_token: Union[bool, Text]) -> Union[bool, Text]:
    if isinstance(hf_token, bool):
        return hf_token
    if hf_token.lower() == "true":
        return True
    if hf_token.lower() == "false":
        return False
    return hf_token


def encode_audio(waveform: np.ndarray) -> Text:
    data = waveform.astype(np.float32).tobytes()
    return base64.b64encode(data).decode("utf-8")


def decode_audio(data: Text) -> np.ndarray:
    # Decode chunk encoded in base64
    byte_samples = base64.decodebytes(data.encode("utf-8"))
    # Recover array from bytes
    samples = np.frombuffer(byte_samples, dtype=np.float32)
    return samples.reshape(1, -1)


def get_padding_left(stream_duration: float, chunk_duration: float) -> float:
    if stream_duration < chunk_duration:
        return chunk_duration - stream_duration
    return 0


def repeat_label(label: Text):
    while True:
        yield label


def get_pipeline_class(class_name: Text) -> type:
    pipeline_class = getattr(blocks, class_name, None)
    msg = f"Pipeline '{class_name}' doesn't exist"
    assert pipeline_class is not None, msg
    return pipeline_class


def get_padding_right(latency: float, step: float) -> float:
    return latency - step


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
