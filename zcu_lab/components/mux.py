"""Ordered resonator and optional qubit bindings for a mux module owner."""

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from zcu_tools.resources.entry.schema import validate_component_name

from .base import BuiltinComponent, Flux


class MuxWiringSchema(BaseModel):
    """Mux generator and ordered tone ADCs, separate from pulse wiring.

    None means wiring is not yet known; Forms require it before Run.
    """

    model_config = ConfigDict(extra="forbid")

    gen_ch: int | None = Field(
        default=None, ge=0, strict=True, description="Mux generator channel, if known."
    )
    ro_ch: list[Annotated[int, Field(ge=0, strict=True)]] | None = Field(
        default=None,
        description="ADC channels aligned with group resonators; None means not yet wired.",
    )


class MuxGroupSchema(BuiltinComponent):
    """A mux module owner with 1..8 ordered resonators and optional qubits.

    resonators[i] binds full readout tone i; duplicate resonators are allowed.
    qubits is None for resonator bring-up, otherwise an equally sized list of
    distinct qubit names. Lab Forms validate component kinds and each qubit's
    resonator against their fixed capture before Run. Reordering either list
    never changes this group's inherited module slots or ext ownership.
    time_of_fly stores this mux route's measured flight time in us, or None
    until known. Resonator pulse-route flight times are separate.
    flux_source names the global coil device, or None until assigned.
    coil_flux_half/period/int hold coil bias calibrations in qubits order,
    using the coil device's native unit. None means the whole calibration is
    unknown; a None element means that qubit has not been calibrated.
    Non-null calibration lists require qubits and must match its length.
    """

    wiring: MuxWiringSchema = Field(
        description="Mux generator and ADC wiring in full resonator order."
    )
    mixer_freq: float | None = Field(
        default=None,
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={"unit": "MHz"},
        description="Mux generator mixer (LO) frequency in MHz; None until known.",
    )
    time_of_fly: float | None = Field(
        default=None,
        strict=True,
        allow_inf_nan=False,
        ge=0,
        json_schema_extra={"unit": "us"},
        description="Mux route flight time in us; None until measured.",
    )
    resonators: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Ordered resonator names, one per full mux tone; duplicates allowed.",
    )
    qubits: list[str] | None = Field(
        default=None,
        description="Optional distinct qubit names, aligned with the full resonators list.",
    )

    flux_source: str | None = Field(
        default=None, description="Device name supplying the global coil bias."
    )
    coil_flux_half: list[Flux] | None = Field(
        default=None,
        description=(
            "Coil biases at half a flux quantum, in coil device units and qubits order; "
            "None for an unknown list or element."
        ),
    )
    coil_flux_period: list[Flux] | None = Field(
        default=None,
        description=(
            "Coil bias spans of one flux quantum, in coil device units and qubits order; "
            "None for an unknown list or element."
        ),
    )
    coil_flux_int: list[Flux] | None = Field(
        default=None,
        description=(
            "Coil biases at integer flux quanta, in coil device units and qubits order; "
            "None for an unknown list or element."
        ),
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
    def validate_wiring_length(self) -> Self:
        """Reject known ADC lists that do not cover the full resonator list."""
        if self.wiring.ro_ch is not None and len(self.wiring.ro_ch) != len(
            self.resonators
        ):
            raise ValueError(
                "wiring.ro_ch/resonators: lengths differ; "
                f"ro_ch={self.wiring.ro_ch!r}, resonators={self.resonators!r}"
            )
        return self

    @model_validator(mode="after")
    def validate_coil_calibration_lengths(self) -> Self:
        """Reject coil calibrations without a matching, nonempty qubit order."""
        for field, values in (
            ("coil_flux_half", self.coil_flux_half),
            ("coil_flux_period", self.coil_flux_period),
            ("coil_flux_int", self.coil_flux_int),
        ):
            if values is None:
                continue
            if not self.qubits:
                raise ValueError(f"{field}: coil 校準需要 qubits 順序對齊")
            if len(values) != len(self.qubits):
                raise ValueError(
                    f"{field}/qubits: lengths differ; "
                    f"{field}={values!r}, qubits={self.qubits!r}"
                )
        return self

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
