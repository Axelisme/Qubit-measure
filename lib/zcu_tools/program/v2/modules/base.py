from abc import ABC, abstractmethod
from typing import Union

from qick.asm_v2 import QickParam

from ..base import MyProgramV2

# --- Helpers functions --- #


def param2str(param: Union[float, QickParam]) -> str:
    if isinstance(param, QickParam):
        if param.is_sweep():
            return f"sweep({param.minval()}, {param.maxval()})"
        else:
            return str(float(param))
    else:
        return str(param)


class Module(ABC):
    @abstractmethod
    def init(self, prog: MyProgramV2) -> None:
        pass

    @abstractmethod
    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        pass


class Delay(Module):
    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.delay = delay

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        prog.delay(t=t + self.delay, tag=self.name)

        return 0.0
