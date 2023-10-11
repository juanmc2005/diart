from abc import ABC, abstractmethod
from typing import Optional, Text

import rich
from rich.progress import Progress, TaskID
from tqdm import tqdm


class ProgressBar(ABC):
    @abstractmethod
    def create(
        self,
        total: int,
        description: Optional[Text] = None,
        unit: Text = "it",
        **kwargs,
    ):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def update(self, n: int = 1):
        pass

    @abstractmethod
    def write(self, text: Text):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def default_description(self) -> Text:
        pass

    @property
    @abstractmethod
    def initial_description(self) -> Optional[Text]:
        pass

    def resolve_description(self, new_description: Optional[Text] = None) -> Text:
        if self.initial_description is None:
            if new_description is None:
                return self.default_description
            return new_description
        else:
            return self.initial_description


class RichProgressBar(ProgressBar):
    def __init__(
        self,
        description: Optional[Text] = None,
        color: Text = "green",
        leave: bool = True,
        do_close: bool = True,
    ):
        self.description = description
        self.color = color
        self.do_close = do_close
        self.bar = Progress(transient=not leave)
        self.bar.start()
        self.task_id: Optional[TaskID] = None

    @property
    def default_description(self) -> Text:
        return f"[{self.color}]Streaming"

    @property
    def initial_description(self) -> Optional[Text]:
        if self.description is not None:
            return f"[{self.color}]{self.description}"
        return self.description

    def create(
        self,
        total: int,
        description: Optional[Text] = None,
        unit: Text = "it",
        **kwargs,
    ):
        if self.task_id is None:
            self.task_id = self.bar.add_task(
                self.resolve_description(f"[{self.color}]{description}"),
                start=False,
                total=total,
                completed=0,
                visible=True,
                **kwargs,
            )

    def start(self):
        assert self.task_id is not None
        self.bar.start_task(self.task_id)

    def update(self, n: int = 1):
        assert self.task_id is not None
        self.bar.update(self.task_id, advance=n)

    def write(self, text: Text):
        rich.print(text)

    def stop(self):
        assert self.task_id is not None
        self.bar.stop_task(self.task_id)

    def close(self):
        if self.do_close:
            self.bar.stop()


class TQDMProgressBar(ProgressBar):
    def __init__(
        self,
        description: Optional[Text] = None,
        leave: bool = True,
        position: Optional[int] = None,
        do_close: bool = True,
    ):
        self.description = description
        self.leave = leave
        self.position = position
        self.do_close = do_close
        self.pbar: Optional[tqdm] = None

    @property
    def default_description(self) -> Text:
        return "Streaming"

    @property
    def initial_description(self) -> Optional[Text]:
        return self.description

    def create(
        self,
        total: int,
        description: Optional[Text] = None,
        unit: Optional[Text] = "it",
        **kwargs,
    ):
        if self.pbar is None:
            self.pbar = tqdm(
                desc=self.resolve_description(description),
                total=total,
                unit=unit,
                leave=self.leave,
                position=self.position,
                **kwargs,
            )

    def start(self):
        pass

    def update(self, n: int = 1):
        assert self.pbar is not None
        self.pbar.update(n)

    def write(self, text: Text):
        tqdm.write(text)

    def stop(self):
        self.close()

    def close(self):
        if self.do_close:
            assert self.pbar is not None
            self.pbar.close()
