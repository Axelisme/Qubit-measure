from abc import ABC, abstractmethod
from typing import Union

from qick.asm_v2 import QickParam

from ..base import MyProgramV2


class Module(ABC):
    @abstractmethod
    def init(self, prog: MyProgramV2) -> None:
        pass

    @abstractmethod
    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        pass


class Delay(Module):
    def __init__(self, name: str, delay: Union[float, QickParam]) -> None:
        self.name = name
        self.delay = delay

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        prog.delay(t=t + self.delay, tag=self.name)

        return 0.0
