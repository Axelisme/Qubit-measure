from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field, TypeAdapter, ValidationInfo
from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Optional,
    Union,
    get_origin,
)

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.program.v2.modular import ModularProgramV2

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


def get_ml_from_context(info: ValidationInfo) -> Optional[ModuleLibrary]:
    if info.context is None:
        return None
    return info.context.get("ml")


class ModuleCfg(ConfigBase):
    type: str
    desc: Optional[str] = None

    def build(self, name: str) -> Module:
        raise NotImplementedError(f"{type(self).__name__}.build is not implemented")

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support set_param")


class ModuleCfgFactory:
    _registry: ClassVar[dict[str, type[ModuleCfg]]] = {}
    _adapter: ClassVar[Optional[TypeAdapter]] = None

    @classmethod
    def register(cls, *module_cfgs: type[ModuleCfg]) -> None:
        for mc in module_cfgs:
            field = mc.model_fields.get("type")
            if field is None or get_origin(field.annotation) is not Literal:
                raise TypeError(
                    f"{mc.__name__} cannot be registered: missing Literal 'type' field"
                )
            type_value = field.default
            existing = cls._registry.get(type_value)
            if existing is not None and existing is not mc:
                raise ValueError(
                    f"Type discriminator {type_value!r} already registered to {existing.__name__}"
                )
            cls._registry[type_value] = mc
        cls._adapter = None

    @classmethod
    def _get_adapter(cls) -> TypeAdapter:
        if cls._adapter is None:
            if not cls._registry:
                raise RuntimeError("No ModuleCfg leaf subclasses registered")
            union = Annotated[
                Union[tuple(cls._registry.values())],
                Field(discriminator="type"),
            ]
            cls._adapter = TypeAdapter(union)
        return cls._adapter

    @classmethod
    def from_raw(cls, raw: Any, *, ml: Optional[ModuleLibrary] = None) -> ModuleCfg:
        ctx = {"ml": ml} if ml is not None else None
        return cls._get_adapter().validate_python(raw, context=ctx)


class Module(ABC):
    name: str

    @abstractmethod
    def init(self, prog: ModularProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]: ...

    def allow_rerun(self) -> bool:
        return False
