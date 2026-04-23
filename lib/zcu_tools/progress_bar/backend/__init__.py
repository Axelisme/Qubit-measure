from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token

from ..base import ProgressSink
from .qt_backend import QtProgressSink, qt_progress_callbacks_scope
from .tqdm_backend import TqdmProgressSink

_backend_override: ContextVar[str | None] = ContextVar(
    "zcu_progress_backend_override", default=None
)


@contextmanager
def progress_backend_scope(kind: str):
    token: Token[str | None] = _backend_override.set(kind.strip().lower())
    try:
        yield
    finally:
        _backend_override.reset(token)


def make_progress_sink(kind: str = "auto", **kwargs) -> ProgressSink:
    selected = kind.strip().lower()
    override = _backend_override.get()
    if override:
        selected = override
    if selected == "auto":
        selected = os.environ.get("ZCU_PROGRESS_BACKEND", "tqdm").strip().lower()

    if selected == "qt":
        return QtProgressSink(**kwargs)
    return TqdmProgressSink()


__all__ = [
    "TqdmProgressSink",
    "QtProgressSink",
    "make_progress_sink",
    "progress_backend_scope",
    "qt_progress_callbacks_scope",
]
