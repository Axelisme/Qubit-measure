from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import ValidationInfo
from qick.asm_v2 import QickParam
from typing_extensions import TYPE_CHECKING, Any, Optional, Union, cast

from zcu_tools.cfg_model import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2.ir.builder import IRBuilder
    from zcu_tools.program.v2.modular import ModularProgramV2


def resolve_module_ref(value: Any, info: ValidationInfo) -> Any:
    if isinstance(value, str):
        if info.context is None:
            raise ValueError("ModuleLibrary context not found")
        return cast("ModuleLibrary", info.context["ml"]).get_module(value)
    return value


class AbsModuleCfg(ConfigBase):
    type: str
    desc: Optional[str] = None

    def build(self, name: str) -> Module:
        raise NotImplementedError(f"{type(self).__name__}.build is not implemented")

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support set_param")


class Module(ABC):
    name: str

    @abstractmethod
    def init(self, prog: ModularProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]: ...

    def ir_run(
        self,
        builder: IRBuilder,
        t: Union[float, QickParam],
        prog: ModularProgramV2,
    ) -> Union[float, QickParam]:
        """Emit IR for this module starting at t. Returns next t for the following module."""
        raise NotImplementedError(f"{type(self).__name__}.ir_run() not implemented")

    def allow_rerun(self) -> bool:
        return False
