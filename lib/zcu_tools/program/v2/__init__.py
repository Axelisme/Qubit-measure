from typing import Any, Dict

from qick.asm_v2 import QickParam, QickSweep1D

from .base import MyProgramV2
from .modular import BaseCustomProgramV2, ModularProgramV2
from .modules import (
    AbsReadout,
    AbsReset,
    BaseReadout,
    Module,
    NoneReset,
    Pulse,
    PulseReset,
    TwoPulseReadout,
    TwoPulseReset,
    check_no_post_delay,
    make_readout,
    make_reset,
)
from .onetone import OneToneProgram
from .twotone import TwoToneProgram


def sweep2param(name: str, sweep: Dict[str, Any]) -> QickParam:
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
    return QickSweep1D(name, sweep["start"], sweep["stop"])
