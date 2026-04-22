from __future__ import annotations

from typing_extensions import Callable, Optional

from ..base import ProgressSink


class QtProgressSink(ProgressSink):
    def __init__(
        self,
        on_start: Optional[Callable[[Optional[int], str], None]] = None,
        on_update_to: Optional[Callable[[int], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_start = on_start
        self._on_update_to = on_update_to
        self._on_close = on_close

    def start(
        self, total: Optional[int], desc: str = "progress", leave: bool = True
    ) -> None:
        del leave
        if self._on_start is not None:
            self._on_start(total, desc)

    def update_to(self, n: int) -> None:
        if self._on_update_to is not None:
            self._on_update_to(n)

    def close(self) -> None:
        if self._on_close is not None:
            self._on_close()
