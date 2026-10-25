"""Shared qubit fields and channel wiring."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.cfg_model import DeviceName

from .base import BuiltinComponent, Duration, Flux, Frequency, reject_null_channel


class QubitWiringSchema(BaseModel):
    """Optional drive and local flux channel indices; supplied values cannot be null."""

    model_config = ConfigDict(extra="forbid")

    ch: int | None = Field(
        default=None, ge=0, strict=True, description="Drive generator channel index."
    )
    lo_flux_ch: int | None = Field(
        default=None, ge=0, strict=True, description="Local flux channel index."
    )

    reject_null_channels = field_validator("ch", "lo_flux_ch", mode="before")(
        reject_null_channel
    )


class QubitBaseSchema(BuiltinComponent):
    """Fields shared by every qubit kind; physics energies belong to subclasses."""

    wiring: QubitWiringSchema = Field(
        default_factory=QubitWiringSchema,
        description="Drive and local flux channel wiring.",
    )
    freq: Frequency = Field(default=None, description="Qubit 0-1 transition frequency.")
    kappa: Frequency = Field(default=None, description="Qubit transition linewidth.")
    t1: Duration = Field(default=None, description="Energy relaxation time T1.")
    t2r: Duration = Field(default=None, description="Ramsey dephasing time T2*.")
    t2e: Duration = Field(default=None, description="Echo dephasing time T2E.")
    # These calibrations belong to this qubit's own DC line, in flux_unit.
    flux_half: Flux = Field(
        default=None,
        description="Own DC line bias at half a flux quantum, in flux_unit.",
    )
    flux_period: Flux = Field(
        default=None,
        description="Own DC line bias span of one flux quantum, in flux_unit.",
    )
    flux_int: Flux = Field(
        default=None,
        description="Own DC line bias at an integer flux quantum, in flux_unit.",
    )
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of this qubit's own DC line bias: A or V."
    )
    resonator: str | None = Field(
        default=None, description="Name of the readout resonator component."
    )
    flux_source: DeviceName | None = Field(
        default=None, description="Device name supplying this qubit's own DC line bias."
    )
