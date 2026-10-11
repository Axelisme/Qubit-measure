from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, TypeVar

from zcu_tools.cfg_model import ConfigBase

T_CfgModel = TypeVar("T_CfgModel", bound="ExpCfgModel")


class ExpCfgModel(ConfigBase):
    """Framework base for experiment-owned inputs.

    Concrete experiment cfgs declare their device role-name mappings and units
    when needed. This base carries neither device selections nor settings.
    Drivers are resolved by the experiment through RunContext.devices at use.
    """

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
