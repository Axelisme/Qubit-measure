from __future__ import annotations

from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from zcu_tools.utils import deepupdate


def _json_fallback(obj: Any) -> str:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise ValidationError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


class ConfigBase(BaseModel):
    """Mutable validated configuration with nested updates and serialization.

    Unknown fields are rejected and assignments are validated. Subclasses own
    their field meanings and validation context. Serialization excludes None
    fields and allows NumPy arrays through the JSON fallback.
    """

    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, validate_assignment=True
    )

    def with_updates(
        self, *, context: dict[str, object] | None = None, **kwargs
    ) -> Self:
        """Return a revalidated nested update without changing this config.

        kwargs supplies field overrides; nested dictionaries merge recursively.
        context is forwarded to subclass validators. Invalid updates raise the
        subclass validation error.
        """
        cfg_dict = self.to_dict()
        deepupdate(cfg_dict, kwargs, behavior="force")

        return self.model_validate(cfg_dict, context=context)

    def to_dict(self) -> dict[str, Any]:
        """Return Python field values, excluding None, for subclass validation."""
        return self.model_dump(mode="python", exclude_none=True)

    def to_json(self, **kwargs) -> str:
        """Serialize fields as indented JSON, excluding None.

        kwargs is forwarded to Pydantic model_dump_json. Unsupported values
        propagate serialization errors; NumPy arrays serialize as lists.
        """
        return self.model_dump_json(
            indent=4, exclude_none=True, fallback=_json_fallback, **kwargs
        )

    def __str__(self) -> str:
        return self.to_json()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"
