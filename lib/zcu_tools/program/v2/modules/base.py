import warnings
from abc import ABC, abstractmethod

from qick.asm_v2 import QickParam
from typing_extensions import Sequence, Union

from ..base import MyProgramV2


class Module(ABC):
    def __init__(self) -> None:
        self.name = "UnnamedModule"

    @abstractmethod
    def init(self, prog: MyProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]: ...


class Delay(Module):
    def __init__(
        self,
        name: str,
        delay: Union[float, QickParam],
        absolute: bool = False,
        hard_delay: bool = True,
    ) -> None:
        self.name = name
        self.delay = delay
        self.absolute = absolute
        self.hard_delay = hard_delay

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        delay_t = self.delay if self.absolute else t + self.delay

        if self.hard_delay:
            prog.delay(t=delay_t, tag=self.name)
            return 0.0  # reset reference time

        return delay_t


class NonBlocking(Module):
    def __init__(self, modules: Sequence[Module]) -> None:
        self.modules = modules

    def init(self, prog: MyProgramV2) -> None:
        for module in self.modules:
            module.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        cur_t = t
        for module in self.modules:
            new_cur_t = module.run(prog, cur_t)
            if new_cur_t == 0.0 or new_cur_t < cur_t:
                warnings.warn(
                    "Find time reset in NonBlocking module. "
                    "Maybe you should set Delay to hard_delay=False.",
                )
            cur_t = new_cur_t
        return t  # non-block returns initial time
