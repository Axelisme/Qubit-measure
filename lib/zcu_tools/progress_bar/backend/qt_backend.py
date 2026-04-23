from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing_extensions import Callable, Optional

from ..base import ProgressSink

_qt_on_start: ContextVar[Optional[Callable[[Optional[int], str], None]]] = ContextVar(
    "zcu_qt_on_start", default=None
)
_qt_on_update_to: ContextVar[Optional[Callable[[int], None]]]= ContextVar(
    "zcu_qt_on_update_to", default=None
)
_qt_on_close: ContextVar[Optional[Callable[[], None]]] = ContextVar(
    "zcu_qt_on_close", default=None
)


@contextmanager
def qt_progress_callbacks_scope(
    on_start: Optional[Callable[[Optional[int], str], None]] = None,
    on_update_to: Optional[Callable[[int], None]] = None,
    on_close: Optional[Callable[[], None]] = None,
):
    tok_start: Token[Optional[Callable[[Optional[int], str], None]]] = _qt_on_start.set(
        on_start
    )
    tok_update: Token[Optional[Callable[[int], None]]] = _qt_on_update_to.set(
        on_update_to
    )
    tok_close: Token[Optional[Callable[[], None]]] = _qt_on_close.set(on_close)
    try:
        yield
    finally:
        _qt_on_start.reset(tok_start)
        _qt_on_update_to.reset(tok_update)
        _qt_on_close.reset(tok_close)


class QtProgressSink(ProgressSink):
    def __init__(
        self,
        on_start: Optional[Callable[[Optional[int], str], None]] = None,
        on_update_to: Optional[Callable[[int], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_start = on_start or _qt_on_start.get()
        self._on_update_to = on_update_to or _qt_on_update_to.get()
        self._on_close = on_close or _qt_on_close.get()

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
