from __future__ import annotations

from qick.asm_v2 import AsmInst, Macro, QickParam, QickSweep1D
from typing_extensions import Optional, Union

from zcu_tools.program import SweepCfg


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
    return QickSweep1D(name, sweep["start"], sweep["stop"])


def param2str(param: Union[float, QickParam]) -> str:
    """Convert a parameter to a string."""
    if isinstance(param, QickParam) and param.is_sweep():
        return f"sweep({param.minval():.3f}, {param.maxval():.3f})"

    return f"{float(param):.3f}"


class PrintTimeStamp(Macro):
    """A helper macro to print the timestamp of the program."""

    def __init__(
        self,
        name: str,
        t: Union[float, QickParam] = 0.0,
        prefix: str = "",
        gen_chs: Optional[list[int]] = None,
        ro_chs: Optional[list[int]] = None,
    ) -> None:
        self.name = name
        self.t = t
        self.prefix = prefix
        self.gen_chs = gen_chs
        self.ro_chs = ro_chs

    def expand(self, prog) -> list[AsmInst]:  # type: ignore
        return []

    def preprocess(self, prog) -> None:
        gen_chs = self.gen_chs
        ro_chs = self.ro_chs
        if gen_chs is None:
            gen_chs = list(range(len(prog._gen_ts)))
        if ro_chs is None:
            ro_chs = list(range(len(prog._ro_ts)))

        print(self.prefix + self.name)
        print(self.prefix + f"\tglobal time: {param2str(self.t)}")
        for ch in gen_chs:
            t = prog._gen_ts[ch]
            if t != 0.0 or self.gen_chs is not None:
                print(self.prefix + f"\tgen[{ch}] " + param2str(t))
        for ch in ro_chs:
            t = prog._ro_ts[ch]
            if t != 0.0 or self.ro_chs is not None:
                print(self.prefix + f"\t ro[{ch}] " + param2str(t))
