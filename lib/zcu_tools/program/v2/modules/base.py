from __future__ import annotations

from abc import ABC, abstractmethod

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    NotRequired,
    Optional,
    Type,
    TypedDict,
    Union,
)


from ..base import MyProgramV2

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
