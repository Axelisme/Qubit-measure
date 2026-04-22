from __future__ import annotations

from tqdm.auto import tqdm
from typing_extensions import Optional

from ..base import ProgressSink


class TqdmProgressSink(ProgressSink):
    def __init__(self) -> None:
        self._bar: Optional[tqdm] = None

    def start(
        self, total: Optional[int], desc: str = "progress", leave: bool = True
    ) -> None:
        self._bar = tqdm(total=total, desc=desc, smoothing=0, leave=leave)

    def update_to(self, n: int) -> None:
        if self._bar is None:
            return
        self._bar.update(n - self._bar.n)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
