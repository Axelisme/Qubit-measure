from __future__ import annotations

from copy import deepcopy
from pprint import pformat

from pydantic import BaseModel, ConfigDict
from typing_extensions import Any, Self


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
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
