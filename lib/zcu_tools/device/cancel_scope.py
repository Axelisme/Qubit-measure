from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_device_setup_cancel_signal: ContextVar[threading.Event | None] = ContextVar(
    "zcu_tools_device_setup_cancel_signal",
    default=None,
)


@contextmanager
def device_setup_cancel_scope(
    cancel_signal: threading.Event | None,
) -> Iterator[threading.Event | None]:
    token = _current_device_setup_cancel_signal.set(cancel_signal)
    try:
        yield cancel_signal
    finally:
        _current_device_setup_cancel_signal.reset(token)


def current_device_setup_cancel_signal() -> threading.Event | None:
    return _current_device_setup_cancel_signal.get()
