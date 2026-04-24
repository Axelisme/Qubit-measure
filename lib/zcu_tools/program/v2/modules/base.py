from __future__ import annotations

from abc import ABC, abstractmethod

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    TypeVar,
    Union,
    overload,
)

from zcu_tools.config import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2.modular import ModularProgramV2


T_ModuleCfg = TypeVar("T_ModuleCfg", bound="ModuleCfg")


class ModuleCfg(ConfigBase):
    # not initiailize here to avoid overwriting when reloading module
    _delegated_cls: ClassVar[dict[str, type["ModuleCfg"]]]

    type: str
    desc: Optional[str] = None

    @classmethod
    def module_type(cls) -> str:
        type_field = cls.model_fields.get("type")
        if type_field is None:
            raise ValueError(f"{cls.__name__} must define a 'type' field")
        if not isinstance(type_field.default, str):
            raise ValueError(
                f"{cls.__name__}.type default must be a string literal for registration"
            )
        return type_field.default

    @classmethod
    def bind_handler(cls, sub_cls: type[T_ModuleCfg]) -> type[T_ModuleCfg]:
        """Bind a module config class to handle it type"""
        if not hasattr(cls, "_delegated_cls"):
            cls._delegated_cls = {}  # initialize here

        name_id = sub_cls.module_type()
        registered_cls = cls._delegated_cls.get(name_id)
        if registered_cls is not None and registered_cls is not sub_cls:
            raise ValueError(
                f"{cls.__name__} already registered name_id {name_id} to {registered_cls.__name__}"
            )

        cls._delegated_cls[name_id] = sub_cls
        return sub_cls

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
    def init(self, prog: ModularProgramV2) -> None: ...

    @abstractmethod
    def run(
        self, prog: ModularProgramV2, t: Union[float, QickParam] = 0.0
    ) -> Union[float, QickParam]: ...

    def allow_rerun(self) -> bool:
        """Whether this module allows being run multiple times (e.g. in a loop).
        If False, running it more than once will raise an error.
        """
        return False
