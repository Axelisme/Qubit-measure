from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol, TypeGuard

from qick.asm_v2 import QickParam, QickSweep1D

from .sweep import SweepCfg

logger = logging.getLogger(__name__)


class QickParamLike(Protocol):
    start: float
    spans: Mapping[str, float]

    def is_sweep(self) -> bool: ...

    def minval(self) -> float: ...

    def maxval(self) -> float: ...

    def to_array(
        self, loop_counts: Mapping[str, int], *, all_loops: bool = False
    ) -> object: ...


def is_qick_param(value: object) -> TypeGuard[QickParamLike]:
    return isinstance(value, QickParam)


def sweep2param(name: str, sweep: SweepCfg) -> QickParam:
    """
    Convert formatted sweep dictionary to a QickSweep1D parameter.

    This function creates a QickSweep1D parameter from a formatted sweep dictionary,
    which is used in Qick v2 assembly programming.

    Args:
        name: Name of the sweep parameter
        sweep: Dictionary containing 'start' and 'stop' values for the sweep

    Returns:
        QickSweep1D: Qick v2 sweep parameter object
    """

    # convert formatted sweep to qick v2 sweep param
    return QickSweep1D(name, sweep.start, sweep.stop)


def param2str(param: float | QickParam) -> str:
    """Convert a parameter to a string."""
    if isinstance(param, QickParam):
        if not is_qick_param(param):
            raise TypeError(
                f"unsupported QickParam implementation: {type(param).__name__}"
            )
        if param.is_sweep():
            return f"sweep({param.minval():.3f}, {param.maxval():.3f})"
        else:
            return f"{param.start:.3f}"
    return f"{float(param):.3f}"
