from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import Any, Optional, Self

from zcu_tools.utils import deepupdate


def _json_fallback(obj: Any) -> str:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise ValidationError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    def with_updates(self, *, context: Optional[dict] = None, **kwargs) -> Self:
        cfg_dict = self.to_dict()
        deepupdate(cfg_dict, kwargs, behavior="force")

        return self.model_validate(cfg_dict, context=context)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(
            indent=4, exclude_none=True, fallback=_json_fallback, **kwargs
        )

    def __str__(self) -> str:
        return self.to_json()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"
