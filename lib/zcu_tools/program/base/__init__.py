from __future__ import annotations

from typing_extensions import TypedDict

from .improve_acquire import TrackerProtocol, ImproveAcquireMixin


class SweepCfg(TypedDict, closed=True):
    start: float
    stop: float
    expts: int
    step: float


__all__ = [
    # improve acquire
    "ImproveAcquireMixin",
    # sweep
    "SweepCfg",
    # tracker
    "TrackerProtocol",
]
