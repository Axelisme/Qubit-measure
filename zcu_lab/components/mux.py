"""Ordered resonator and optional qubit bindings for a mux module owner."""

from typing import Self

from pydantic import Field, ValidationInfo, field_validator, model_validator
from zcu_tools.resources.entry.schema import validate_component_name

from .base import BuiltinComponent


class MuxGroupSchema(BuiltinComponent):
    """A mux module owner with 1..8 ordered resonators and optional qubits.

    resonators[i] binds full readout tone i; duplicate resonators are allowed.
    qubits is None for resonator bring-up, otherwise an equally sized list of
    distinct qubit names. Lab Forms validate component kinds and each qubit's
    resonator against their fixed capture before Run. Reordering either list
    never changes this group's inherited module slots or ext ownership.
    """

    resonators: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Ordered resonator names, one per full mux tone; duplicates allowed.",
    )
    qubits: list[str] | None = Field(
        default=None,
        description="Optional distinct qubit names, aligned with the full resonators list.",
    )

    @field_validator("resonators", "qubits")
    @classmethod
    def validate_names(
        cls, value: list[str] | None, info: ValidationInfo
    ) -> list[str] | None:
        """Require addressable names, reporting the field, index and actual value."""
        if value is not None:
            for index, name in enumerate(value):
                try:
                    validate_component_name(name)
                except ValueError as exc:
                    raise ValueError(
                        f"{info.field_name}[{index}]={name!r}: {exc}"
                    ) from exc
        return value

    @model_validator(mode="after")
    def validate_qubit_structure(self) -> Self:
        """Reject unequal lists or repeated qubits without resolving components."""
        if self.qubits is None:
            return self
        if len(self.qubits) != len(self.resonators):
            index = min(len(self.qubits), len(self.resonators))
            raise ValueError(
                f"qubits/resonators[{index}]: lengths differ; "
                f"qubits={self.qubits!r}, resonators={self.resonators!r}"
            )
        for index, name in enumerate(self.qubits):
            if name in self.qubits[:index]:
                raise ValueError(
                    f"qubits[{index}]={name!r} duplicates qubits[{self.qubits.index(name)}]"
                )
        return self
