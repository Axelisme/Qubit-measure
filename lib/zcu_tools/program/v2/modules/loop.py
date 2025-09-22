import qick.asm_v2 as qasm

from ..base import MyProgramV2
from .base import Module


class Repeat(Module):
    def __init__(self, name: str, n: int, sub_module: Module) -> None:
        self.name = name
        self.n = n
        self.sub_module = sub_module

        if n < 0:
            raise ValueError(f"Repeat n must be greater than or equal to 0, got {n}")

    def init(self, prog: MyProgramV2) -> None:
        self.sub_module.init(prog)

    def run(self, prog: MyProgramV2) -> None:
        if self.n == 0:
            return  # do nothing

        # this n must > 0, to prevent infinite loop in qick
        self.append_macro(qasm.OpenLoop(name=self.name, n=self.n))
        self.sub_module.run(prog)
        self.append_macro(qasm.CloseLoop())
