"""Shared qubit fields and channel wiring."""

from math import isclose
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


class QubitFluxCalibration(BaseModel):
    """One DC device's reference points in its native instrument unit.

    half and integer are finite biases with all other contributing devices at
    their integer references. period is the finite nonzero signed bias span
    per flux quantum; negative reverses the flux direction. Missing or invalid
    fields fail model validation. The flux helpers check half/integer spacing
    when using the calibration, not while loading the component.
    """

    model_config = ConfigDict(extra="forbid")

    half: float = Field(
        strict=True,
        allow_inf_nan=False,
        description="Half-flux reference bias in this device's native unit.",
    )
    integer: float = Field(
        strict=True,
        allow_inf_nan=False,
        description="Integer-flux reference bias in this device's native unit.",
    )
    period: float = Field(
        strict=True,
        allow_inf_nan=False,
        description="Signed nonzero device bias span per flux quantum.",
    )

    def validate_spacing(self, device_name: str) -> None:
        """Check this calibration's half/integer spacing when it is used.

        device_name labels the device in errors; the caller validates its name
        and membership separately. Return None without changing the calibration.
        Raise ValueError naming the device unless abs(integer-half) matches
        abs(period)/2 with relative tolerance 1e-9 and absolute tolerance 0.
        Loading a calibration does not call this method.
        """
        if not isclose(
            abs(self.integer - self.half),
            abs(self.period) / 2,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"device {device_name!r} half/integer separation must be abs(period)/2"
            )

    @field_validator("period")
    @classmethod
    def require_nonzero_period(cls, value: float) -> float:
        """Reject a zero period after finite numeric field validation."""
        if value == 0:
            raise ValueError("period must be nonzero")
        return value


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
    flux: dict[str, QubitFluxCalibration] = Field(
        default_factory=dict,
        description="Per-device reference half/integer biases and signed periods.",
    )
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
