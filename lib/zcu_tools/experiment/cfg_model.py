from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Optional, TypeVar

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo

T_CfgModel = TypeVar("T_CfgModel", bound="ExpCfgModel")


class ExpCfgModel(ConfigBase):
    dev: Mapping[str, DeviceInfo] | None = None

    @classmethod
    def validate_or_warn(
        cls: type[T_CfgModel],
        cfg: dict[str, Any],
        *,
        source: str,
    ) -> Any | None:
        try:
            return cls.model_validate(deepcopy(cfg))
        except Exception as exc:
            warnings.warn(
                f"Failed to validate loaded cfg from {source} with {cls.__name__}: {exc}",
                stacklevel=2,
            )
            return None
