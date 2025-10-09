from typing import Any, Dict, List, Union

from qick.asm_v2 import AsmInst, Macro, QickParam, QickSweep1D


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
    if not isinstance(sweep, dict):
        raise ValueError("To convert sweep to QickParam, sweep must be a dict")

    # convert formatted sweep to qick v2 sweep param
    return QickSweep1D(name, sweep["start"], sweep["stop"])


def param2str(param: Union[float, QickParam]) -> str:
    """Convert a parameter to a string."""
    if isinstance(param, QickParam):
        if param.is_sweep():
            return f"sweep({param.minval()}, {param.maxval()})"
        else:
            return str(float(param))
    else:
        return str(param)


class PrintTimeStamp(Macro):
    """A helper macro to print the timestamp of the program."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def expand(self, prog) -> List[AsmInst]:
        return []

    def preprocess(self, prog) -> None:
        timestamps = []
        timestamps += list(prog._gen_ts)
        timestamps += list(prog._ro_ts)
        print(self.prefix)
        for i, t in enumerate(timestamps):
            print(f"\t[{i}] " + param2str(t))
