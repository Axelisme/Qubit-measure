from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from qick.asm_v2 import QickParam
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    TypeVar,
    Union,
    final,
    overload,
)

from zcu_tools.config import ConfigBase

if TYPE_CHECKING:
    from zcu_tools.meta_tool import ModuleLibrary
    from zcu_tools.program.v2.modular import ModularProgramV2


T_ModuleCfg = TypeVar("T_ModuleCfg", bound="ModuleCfg")


class ModuleCfg(ConfigBase):
    # not initiailize here to avoid overwriting when reloading module
    _delegated_cls: ClassVar[dict[str, type[ModuleCfg]]]

    type: str
    desc: Optional[str] = None

    @classmethod
    def bind_handler(cls, label: str):
        """Bind a module config class to handle it type"""
        if not hasattr(cls, "_delegated_cls"):
            cls._delegated_cls = {}  # initialize here

        def decorator(sub_cls: type[T_ModuleCfg]) -> type[T_ModuleCfg]:
            registered_cls = cls._delegated_cls.get(label)
            if registered_cls is not None and registered_cls is not sub_cls:
                raise ValueError(
                    f"{cls.__name__} already registered label {label} to {registered_cls.__name__}"
                )

            # check if sub_cls type is indeed label
            default_type = sub_cls.model_fields["type"].default
            if default_type != label:
                raise ValueError(
                    f"{sub_cls.__name__} type field must be {label} to match the label in bind_handler"
                    f"But got {default_type}"
                )

            cls._delegated_cls[label] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def _from_dict(
        cls: type[T_ModuleCfg], raw_cfg: dict[str, Any], ml: ModuleLibrary
    ) -> T_ModuleCfg:
        return cls.model_validate(raw_cfg)

    @final
    @classmethod
    def from_dict(cls, raw_cfg: dict[str, Any], ml: ModuleLibrary) -> ModuleCfg:
        delegated_cls = cls
        if (type := raw_cfg.get("type")) and type in cls._delegated_cls:
            delegated_cls = cls._delegated_cls[type]

        try:
            return delegated_cls._from_dict(deepcopy(raw_cfg), ml)
        except Exception as e:
            raise ValueError(
                f"Error parsing {delegated_cls.__name__} from raw cfg:\n"
                f"{raw_cfg}\n"
                f"Error details:\n{e}"
            ) from e

    @classmethod
    @overload
    def from_raw(cls, raw_cfg: T_ModuleCfg, ml: ModuleLibrary) -> T_ModuleCfg: ...

    @classmethod
    @overload
    def from_raw(
        cls: type[T_ModuleCfg], raw_cfg: Union[str, dict[str, Any]], ml: ModuleLibrary
    ) -> T_ModuleCfg: ...

    @final
    @classmethod
    def from_raw(
        cls, raw_cfg: Union[str, dict[str, Any], ModuleCfg], ml: ModuleLibrary
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
