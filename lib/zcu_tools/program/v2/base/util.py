from typing import List, Union

from qick.asm_v2 import AsmInst, Macro, QickParam


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
