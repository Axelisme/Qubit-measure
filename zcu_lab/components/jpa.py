"""JPA component model with explicit pump and flux units."""

from typing import Literal

from pydantic import Field
from zcu_tools.cfg_model import DeviceName

from .base import BuiltinComponent, Flux, Frequency, PumpPower


class JpaSchema(BuiltinComponent):
    flux_dev: DeviceName | None = Field(
        default=None,
        description="Primary (default) flux control device; not part of the flux calculation and need not have a flux calibration. None leaves the form device blank.",
    )
    rf_source: DeviceName | None = Field(
        default=None,
        description="RF device name for this JPA's pump; None leaves the form device blank.",
    )
    pump_freq: Frequency = Field(default=None, description="JPA pump frequency.")
    pump_power: PumpPower = Field(default=None, description="JPA pump power.")
    flux: Flux = Field(default=None, description="JPA flux bias, in flux_unit.")
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of the JPA flux bias: A or V."
    )
