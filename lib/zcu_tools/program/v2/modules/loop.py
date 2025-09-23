from typing import List, Union

import qick.asm_v2 as qasm

from ..base import MyProgramV2
from .base import Module


class Repeat(Module):
    """
    Repeat a module or a list of modules n times
    It will call delay before and inside each loop, this may cause unexpected behavior
    """

    def __init__(
        self, name: str, n: int, sub_module: Union[Module, List[Module]]
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

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        if self.n == 0:
            return t

        # this n must > 0, to prevent infinite loop in qick
        assert self.n > 0

        prog.delay(t=t)
        prog.append_macro(qasm.OpenLoop(name=self.name, n=self.n))
        cur_t = 0.0
        for mod in self.sub_module:
            cur_t = mod.run(prog, cur_t)
        prog.delay(t=cur_t)
        prog.append_macro(qasm.CloseLoop())

        return 0.0  # prog.delay will modify ref time
