from abc import ABC, abstractmethod

from ..base import MyProgramV2


class Module(ABC):
    @abstractmethod
    def init(self, prog: MyProgramV2) -> None:
        pass

    @abstractmethod
    def run(self, prog: MyProgramV2) -> None:
        pass
