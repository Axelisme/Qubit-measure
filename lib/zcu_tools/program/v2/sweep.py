from __future__ import annotations

from typing_extensions import TypedDict


class SweepCfg(TypedDict, closed=True):
    start: float
    stop: float
    expts: int
    step: float
