from __future__ import annotations

from collections.abc import Iterable

from tqdm.auto import tqdm

from ..base import BaseProgressBar, ProgressTotal, ProgressValue


class TQDMProgressBar(BaseProgressBar):
    """tqdm display with readable progress state even when display is disabled.

    Initial descriptions are returned unchanged. ``set_description`` appends
    tqdm's ``": "`` suffix to nonempty descriptions, regardless of display state.
    """

    def __init__(
        self,
        iterable: Iterable[object] | None = None,
        desc: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.pbar = tqdm(iterable, desc, *args, **kwargs)
        # Disabled tqdm skips counter updates; progress belongs to this adapter.
        self._n: ProgressValue = self.pbar.n
        # Disabled tqdm skips description initialization before returning.
        self.pbar.desc = desc or ""

    def update(self, value: ProgressValue = 1) -> None:
        self._n += value
        self.pbar.update(value)

    def set_progress(self, value: ProgressValue) -> None:
        self.update(value - self._n)

    def reset(self) -> None:
        self._n = 0
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
        return self._n

    @property
    def desc(self) -> str:
        return self.pbar.desc
