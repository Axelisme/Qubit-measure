from __future__ import annotations

import qick.asm_v2 as qasm
from qick.asm_v2 import QickParam
from typing_extensions import Union
from zcu_tools.config import config

from ..base import MyProgramV2
from .base import Module
from ..utils import PrintTimeStamp


class Repeat(Module):
    """
    Repeat a module or a list of modules n times
    It will call delay before and inside each loop, this may cause unexpected behavior
    """

    def __init__(
        self, name: str, n: int, sub_module: Union[Module, list[Module]]
    ) -> None:
        self.name = name
        self.n = n

        if isinstance(sub_module, Module):
            sub_module = [sub_module]
        self.sub_module = sub_module

        if n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def init(self, prog: MyProgramV2) -> None:
        for mod in self.sub_module:
            mod.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        if self.n == 0:
            return t

        # this n must > 0, to prevent infinite loop in qick
        assert self.n > 0

        prog.delay(t=t)
        prog.append_macro(qasm.OpenLoop(name=self.name, n=self.n))
        cur_t = 0.0
        for mod in self.sub_module:
            if config.DEBUG_MODE:
                prog.append_macro(
                    PrintTimeStamp(
                        f"{mod.__class__.__name__}({mod.name})", cur_t, prefix="\t"
                    )
                )
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.append_macro(qasm.CloseLoop())

        return 0.0  # prog.delay will modify ref time


class SoftRepeat(Module):
    """Repeat a module or a list of modules n times"""

    def __init__(
        self, name: str, n: int, sub_module: Union[Module, list[Module]]
    ) -> None:
        self.name = name
        self.n = n

        if isinstance(sub_module, Module):
            sub_module = [sub_module]
        self.sub_module = sub_module

        if n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def init(self, prog: MyProgramV2) -> None:
        for mod in self.sub_module:
            mod.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        for _ in range(self.n):
            for mod in self.sub_module:
                t = mod.run(prog, t)
        return t
