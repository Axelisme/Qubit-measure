from __future__ import annotations

from tqdm.auto import tqdm

from ..base import BaseProgressBar, ProgressTotal, ProgressValue


class TQDMProgressBar(BaseProgressBar):
    def __init__(self, *args, **kwargs) -> None:
        self.pbar = tqdm(*args, **kwargs)

    def update(self, value: ProgressValue = 1) -> None:
        self.pbar.update(value)

    def reset(self) -> None:
        self.pbar.reset()

    def refresh(self) -> None:
        self.pbar.refresh()

    def close(self) -> None:
        self.pbar.close()

    def set_description(self, description: str) -> None:
        self.pbar.set_description(description)

    @property
    def total(self) -> ProgressTotal:
        return self.pbar.total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self.pbar.total = value

    @property
    def n(self) -> ProgressValue:
        return self.pbar.n

    @property
    def desc(self) -> str:
        return self.pbar.desc
