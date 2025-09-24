from abc import ABC, abstractmethod


from ..base import MyProgramV2


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
