"""Shared qubit fields and channel wiring."""

from math import isclose
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from zcu_tools.cfg_model import DeviceName

from .base import BuiltinComponent, Duration, Frequency, reject_null_channel


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
    """Projected or fitted DC coordinates, not a stored component flux block.

    half and integer are finite native biases for flux 0.5 and 0, respectively.
    period is the finite nonzero signed bias span per flux quantum. Callers of
    historical conversion validate spacing before accepting a calibration.
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


class QubitFluxSchema(BaseModel):
    """Reduced flux at zero bias plus independent signed device periods.

    offset is in flux quanta, or None until calibrated. period maps instrument
    names to finite nonzero native bias spans per flux quantum. A device without
    a current value contributes zero without losing its stored period.
    """

    model_config = ConfigDict(extra="forbid")

    offset: float | None = Field(
        default=None,
        strict=True,
        allow_inf_nan=False,
        description="Reduced flux with every device at zero; None until calibrated.",
    )
    period: dict[
        DeviceName, Annotated[float, Field(strict=True, allow_inf_nan=False)]
    ] = Field(
        default_factory=dict,
        description="Signed nonzero native bias span per flux quantum, by device.",
    )

    @field_validator("period")
    @classmethod
    def require_nonzero_periods(cls, values: dict[str, float]) -> dict[str, float]:
        """Reject a zero period and identify its instrument."""
        for name, value in values.items():
            if value == 0:
                raise ValueError(f"device {name!r}: period must be nonzero")
        return values


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
    flux: QubitFluxSchema = Field(
        default_factory=QubitFluxSchema,
        description="Zero-bias reduced flux offset and signed per-device periods.",
    )
    resonator: str | None = Field(
        default=None, description="Name of the readout resonator component."
    )
    flux_dev: DeviceName | None = Field(
        default=None,
        description="Primary (default) flux control device; not part of the flux calculation and need not have a flux calibration. None leaves the form device blank.",
    )
