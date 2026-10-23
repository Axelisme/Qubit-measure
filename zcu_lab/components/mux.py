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


class CoilFluxCalibration(BaseModel):
    """One qubit's global-coil calibration in the coil device's native unit.

    half is the coil bias at half a flux quantum, period is the bias span of
    one flux quantum, and int is the bias at an integer flux quantum.
    Each field defaults to None until measured. Supplied numbers must be
    finite; unknown fields and non-numeric values raise ValidationError.
    """

    model_config = ConfigDict(extra="forbid")

    half: Flux = Field(
        default=None,
        description="Coil bias at half a flux quantum; None until measured.",
    )
    period: Flux = Field(
        default=None,
        description="Coil bias span of one flux quantum; None until measured.",
    )
    int: Flux = Field(
        default=None,
        description="Coil bias at an integer flux quantum; None until measured.",
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
    coil_flux maps qubit names to CoilFluxCalibration, defaulting to an empty
    mapping until measured. Keys must belong to qubits; when qubits is None
    or empty, only an empty mapping satisfies this condition. Unknown keys
    raise ValidationError. The qubits/resonators length rule still applies.
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
    coil_flux: dict[str, CoilFluxCalibration] = Field(
        default_factory=dict,
        description=(
            "Global-coil calibrations in coil device units, keyed by group qubit name; "
            "empty until calibrated. Keys must be a subset of qubits."
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
    def validate_coil_calibration_keys(self) -> Self:
        """Reject coil calibration keys that are not group qubit names."""
        unknown = self.coil_flux.keys() - set(self.qubits or ())
        if unknown:
            raise ValueError(
                "coil_flux: keys must belong to qubits; "
                f"unknown={sorted(unknown)!r}, qubits={self.qubits!r}"
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
