from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Union,
)

from zcu_tools.config import config

from ..base import MyProgramV2
from .util import round_timestamp
from ..utils import PrintTimeStamp

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


class ModuleCfg(TypedDict, closed=False):
    type: str
    desc: NotRequired[str]


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
    def auto_fill(cfg: Union[str, dict[str, Any]], ml: ModuleLibrary) -> ModuleCfg:
        raise NotImplementedError("auto_fill is not implemented for this module")

    def __init__(self, *args, **kwargs) -> None:
        self.name = "UnnamedModule"

    @abstractmethod
    def init(self, prog: MyProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]: ...


class Delay(Module):
    def __init__(
        self, name: str, delay: Union[float, QickParam], absolute: bool = False
    ) -> None:
        self.name = name
        self.delay = delay
        self.absolute = absolute

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        delay_t = self.delay if self.absolute else t + self.delay
        delay_t = round_timestamp(prog, delay_t)

        prog.delay(t=delay_t, tag=self.name)

        return 0.0  # reset reference time


class SoftDelay(Module):
    def __init__(
        self, name: str, delay: Union[float, QickParam], absolute: bool = False
    ) -> None:
        self.name = name
        self.delay = delay
        self.absolute = absolute

    def init(self, prog: MyProgramV2) -> None:
        pass

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        delay_t = self.delay if self.absolute else t + self.delay

        return round_timestamp(prog, delay_t)


class NonBlocking(Module):
    def __init__(self, modules: Sequence[Module]) -> None:
        self.modules = modules
        self.name = "[" + ", ".join(module.name for module in modules) + "]"

    def init(self, prog: MyProgramV2) -> None:
        for module in self.modules:
            module.init(prog)

    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]:
        cur_t = t
        for module in self.modules:
            if config.DEBUG_MODE:
                prog.append_macro(
                    PrintTimeStamp(
                        f"{module.__class__.__name__}({module.name})", prefix="\t"
                    )
                )
            new_cur_t = module.run(prog, cur_t)
            if new_cur_t < cur_t:
                warnings.warn(
                    "Find time reset in NonBlocking module. "
                    "Maybe you should set SoftDelay instead of Delay.",
                )
            cur_t = new_cur_t

        return t  # non-block returns initial time
