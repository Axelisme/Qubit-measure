"""JPA component model with explicit pump and flux units."""

from typing import Literal

from pydantic import Field

from .base import BuiltinComponent, Flux, Frequency, PumpPower


class JpaSchema(BuiltinComponent):
    flux_source: str | None = Field(
        default=None,
        description="DC device name for this JPA's flux bias; None leaves the form device blank.",
    )
    rf_source: str | None = Field(
        default=None,
        description="RF device name for this JPA's pump; None leaves the form device blank.",
    )
    pump_freq: Frequency = Field(default=None, description="JPA pump frequency.")
    pump_power: PumpPower = Field(default=None, description="JPA pump power.")
    flux: Flux = Field(default=None, description="JPA flux bias, in flux_unit.")
    flux_unit: Literal["A", "V"] | None = Field(
        default=None, description="Unit of the JPA flux bias: A or V."
    )
