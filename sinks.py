from rx.core import Observer
from pyannote.core import Annotation, notebook
from typing import Union, Text, Optional
from pathlib import Path
from traceback import print_exc
import matplotlib.pyplot as plt


class OutputBuilder(Observer):
    def __init__(
        self,
        path: Optional[Union[Path, Text]] = None,
        merge_collar: float = 0.05,
        draw: bool = True
    ):
        super().__init__()
        self.output: Optional[Annotation] = None
        self.collar = merge_collar
        self.path = path
        self.enable_draw = draw
        if self.enable_draw:
            self.figure, self.ax = plt.subplots(figsize=(16, 2))

    def draw(self):
        plt.cla()
        notebook.crop = self.output.get_timeline().extent()
        notebook.plot_annotation(self.output, self.ax)
        plt.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.05)

    def on_next(self, value: Annotation):
        # Update output
        if self.output is None:
            self.output = value
        else:
            self.output = self.output.update(value).support(self.collar)
        # Draw if possible
        if self.enable_draw:
            self.draw()
        # Save RTTM if possible
        if self.path is not None:
            with open(Path(self.path), 'w') as file:
                self.output.write_rttm(file)

    def on_error(self, error: Exception):
        print_exc()
        exit(1)
