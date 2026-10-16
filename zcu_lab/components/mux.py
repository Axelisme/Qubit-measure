"""Ordered mux group component model and member-name validation."""

from pydantic import Field, field_validator
from zcu_tools.resources.entry.schema import validate_component_name

from .base import BuiltinComponent


class MuxGroupSchema(BuiltinComponent):
    """A mux module owner with 1..8 ordered component names.

    members binds full readout tones by index, not a kind constraint or a set.
    Reordering members never changes this group's module ownership. Inherited
    module and ext retain the ordinary component slot and extension semantics.
    """

    members: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Ordered component names, one per mux tone; duplicates are allowed.",
    )

    @field_validator("members")
    @classmethod
    def validate_members(cls, value: list[str]) -> list[str]:
        """Require addressable component names without resolving their kinds."""
        for name in value:
            validate_component_name(name)
        return value
