from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TypeVar

from zcu_tools.cfg_model import ConfigBase

T_CfgModel = TypeVar("T_CfgModel", bound="ExpCfgModel")


class ExpCfgModel(ConfigBase):
    """Experiment inputs with optional role-to-connected-device-name selection.

    dev maps experiment role keys to driver names, or is None when no roles
    are selected. It contains neither driver settings nor acquisition metadata.
    The experiment resolves a name through RunContext.devices when it uses it.
    """

    dev: Mapping[str, str] | None = None

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
