from __future__ import annotations

import logging

from qick.asm_v2 import QickParam, QickSweep1D
from typing_extensions import Union

from .sweep import SweepCfg

logger = logging.getLogger(__name__)


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


def param2str(param: Union[float, QickParam]) -> str:
    """Convert a parameter to a string."""
    if isinstance(param, QickParam):
        if param.is_sweep():
            return f"sweep({param.minval():.3f}, {param.maxval():.3f})"
        else:
            return f"{param.start:.3f}"
    return f"{float(param):.3f}"
