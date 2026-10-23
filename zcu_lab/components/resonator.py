"""Readout resonator component model."""

from pydantic import Field

from .base import BuiltinComponent, Frequency, WiringSchema


class ResonatorSchema(BuiltinComponent):
    wiring: WiringSchema = Field(
        default_factory=WiringSchema,
        description="Single-tone generator and ADC wiring.",
    )
    freq: Frequency = Field(default=None, description="Resonator frequency.")
    kappa: Frequency = Field(default=None, description="Resonator linewidth.")
    time_of_fly: float | None = Field(
        default=None,
        strict=True,
        allow_inf_nan=False,
        ge=0,
        json_schema_extra={"unit": "us"},
        description="Single-tone readout route flight time in us; None until measured.",
    )
    amplifier: str | None = Field(
        default=None, description="Name of the readout amplifier component."
    )
