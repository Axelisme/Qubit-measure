from abc import ABC, abstractmethod

from qick.asm_v2 import QickParam

from ..base import MyProgramV2


class Module(ABC):
    @abstractmethod
    def init(self, prog: MyProgramV2) -> None:
        pass

    @abstractmethod
    def run(self, prog: MyProgramV2, t: float = 0.0) -> float:
        pass
