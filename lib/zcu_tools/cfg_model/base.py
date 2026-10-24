from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from zcu_tools.utils import deepupdate


class ConfigBase(BaseModel):
    """Mutable validated configuration with nested updates and serialization.

    Unknown fields are rejected and assignments are validated. Subclasses own
    their field meanings and validation context. Serialization excludes None
    fields.
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

    def __str__(self) -> str:
        return self.model_dump_json(indent=4, exclude_none=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"
