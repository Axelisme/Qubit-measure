from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pprint import pformat

from pydantic import BaseModel, ConfigDict
from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Optional,
    Self,
    TypeVar,
    Union,
    overload,
)

from ..base import MyProgramV2

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    def with_updates(self, **kwargs) -> Self:
        cfg = deepcopy(self)

        def deepupdate_cfg(cfg: ConfigBase, d: dict[str, Any]) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    deepupdate_cfg(getattr(cfg, key), value)
                else:
                    setattr(cfg, key, value)

        deepupdate_cfg(cfg, kwargs)
        return cfg.model_validate(cfg)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", by_alias=True, exclude_none=True)

    def __repr__(self) -> str:
        dict_repr = pformat(self.to_dict(), width=88, compact=False, sort_dicts=False)
        return f"{self.__class__.__name__}(\n{dict_repr}\n)"

    __str__ = __repr__


T_ModuleCfg = TypeVar("T_ModuleCfg", bound="ModuleCfg")


class ModuleCfg(ConfigBase):
    _delegated_cls: ClassVar[dict[str, type["ModuleCfg"]]]

    type: str
    desc: Optional[str] = None

    @classmethod
    def register_handler(
        cls, name_id: str
    ) -> Callable[[type[T_ModuleCfg]], type[T_ModuleCfg]]:
        if not hasattr(cls, "_delegated_cls"):
            cls._delegated_cls = {}

        def decorator(sub_cls: type[T_ModuleCfg]) -> type[T_ModuleCfg]:
            registered_cls = cls._delegated_cls.get(name_id)
            if registered_cls is not None and registered_cls is not sub_cls:
                raise ValueError(
                    f"{cls.__name__} already registered name_id {name_id} to {registered_cls.__name__}"
                )

            cls._delegated_cls[name_id] = sub_cls

            return sub_cls

        return decorator

    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: "ModuleLibrary") -> "ModuleCfg":
        if (type := raw_cfg.get("type")) and (
            registered_cls := cls._delegated_cls.get(type)
        ):
            if registered_cls is cls:
                raise ValueError(
                    f"Recursive module cfg type registration detected: {type}"
                )
            # delegate to the registered class
            return registered_cls.from_dict(raw_cfg, ml)

        return cls.model_validate(raw_cfg)  # default parser

    @classmethod
    @overload
    def from_raw(cls, raw_cfg: T_ModuleCfg, ml: "ModuleLibrary") -> T_ModuleCfg: ...

    @classmethod
    @overload
    def from_raw(
        cls: type[T_ModuleCfg], raw_cfg: Union[str, dict[str, Any]], ml: "ModuleLibrary"
    ) -> T_ModuleCfg: ...

    @classmethod
    def from_raw(
        cls, raw_cfg: Union[str, dict[str, Any], ModuleCfg], ml: "ModuleLibrary"
    ) -> ModuleCfg:
        if isinstance(raw_cfg, str):
            raw_cfg = ml.get_module(raw_cfg)

        if isinstance(raw_cfg, dict):
            return cls.from_dict(raw_cfg, ml)

        if not isinstance(raw_cfg, cls):
            raise ValueError(
                f"Invalid cfg type for {cls.__name__}: {raw_cfg.__class__.__name__}"
            )

        return raw_cfg

    def set_param(self, name: str, value: Union[float, QickParam]) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_param"
        )

class Module(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self.name = "UnnamedModule"

    @abstractmethod
    def init(self, prog: MyProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: MyProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]: ...
