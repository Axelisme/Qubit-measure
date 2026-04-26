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

from zcu_tools.config import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2.modular import ModularProgramV2


def get_ml_from_context(info: ValidationInfo) -> Optional[ModuleLibrary]:
    if info.context is None:
        return None
    return info.context.get("ml")


class ModuleCfg(ConfigBase):
    type: str
    desc: Optional[str] = None

    _leaf_subclasses: ClassVar[list[type[ModuleCfg]]] = []
    _adapter: ClassVar[Optional[TypeAdapter]] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("type")
        if field is not None and get_origin(field.annotation) is Literal:
            ModuleCfg._leaf_subclasses.append(cls)
            ModuleCfg._adapter = None

    @classmethod
    def _get_adapter(cls) -> TypeAdapter:
        if ModuleCfg._adapter is None:
            if not ModuleCfg._leaf_subclasses:
                raise RuntimeError("No ModuleCfg leaf subclasses registered")
            union = Annotated[
                Union[tuple(ModuleCfg._leaf_subclasses)],
                Field(discriminator="type"),
            ]
            ModuleCfg._adapter = TypeAdapter(union)
        return ModuleCfg._adapter

    @classmethod
    def from_raw(cls, raw: Any, *, ml: Optional[ModuleLibrary] = None) -> ModuleCfg:
        ctx = {"ml": ml} if ml is not None else None
        return cls._get_adapter().validate_python(raw, context=ctx)

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

    def allow_rerun(self) -> bool:
        return False
