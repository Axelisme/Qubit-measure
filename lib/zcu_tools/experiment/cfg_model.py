from __future__ import annotations

import warnings
from copy import deepcopy

from typing_extensions import Any, Mapping, Optional, TypeVar

from zcu_tools.config import ConfigBase
from zcu_tools.device import DeviceInfo

T_CfgModel = TypeVar("T_CfgModel", bound="ExpCfgModel")


class ExpCfgModel(ConfigBase):
    dev: Optional[Mapping[str, DeviceInfo]] = None

    @classmethod
    def validate_or_warn(
        cls: type[T_CfgModel],
        cfg: dict[str, Any],
        *,
        source: str,
    ) -> Optional[Any]:
        try:
            return cls.model_validate(deepcopy(cfg))
        except Exception as exc:
            warnings.warn(
                f"Failed to validate loaded cfg from {source} with {cls.__name__}: {exc}",
                stacklevel=2,
            )
            return None
