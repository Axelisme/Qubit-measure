from __future__ import annotations

import os

from ..base import ProgressSink
from .qt_backend import QtProgressSink
from .tqdm_backend import TqdmProgressSink


def make_progress_sink(kind: str = "auto", **kwargs) -> ProgressSink:
    selected = kind.strip().lower()
    if selected == "auto":
        selected = os.environ.get("ZCU_PROGRESS_BACKEND", "tqdm").strip().lower()

    if selected == "qt":
        return QtProgressSink(**kwargs)
    return TqdmProgressSink()


__all__ = [
    "TqdmProgressSink",
    "QtProgressSink",
    "make_progress_sink",
]
