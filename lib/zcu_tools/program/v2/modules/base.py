from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Union,
    cast,
)

from ..base import MyProgramV2

if TYPE_CHECKING:
    from zcu_tools.meta_manager import ModuleLibrary


class ModuleCfg(TypedDict, closed=False):
    type: str


class Module(ABC):
    @classmethod
    def declare_submodule(cls) -> None:
        # lazy declaration to avoid overwriting submodule dict when reload the module
        if not hasattr(cls, "_submodule"):
            cls._submodule = {}

    def __init_subclass__(cls, tag: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls._submodule = {}  # use separate submodule dict for each subclass

        if tag is None:
            return

        for base in cls.__bases__:
            if issubclass(base, Module):
                base.declare_submodule()
                base._submodule[tag] = cls

    @staticmethod
    def parse(type: str) -> Type["Module"]:
        Module.declare_submodule()

        cur_cls = Module
        for label in type.split("/"):
            if label not in cur_cls._submodule:
                raise ValueError(f"Invalid module type: {type}")
            cur_cls = cur_cls._submodule[label]

        return cur_cls

    @staticmethod
    def auto_fill(module_cfg: Dict[str, Any], ml: ModuleLibrary) -> ModuleCfg:
        return cast(ModuleCfg, module_cfg)

    def __init__(self, *args, **kwargs) -> None: ...

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
