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
    amplifier: str | None = Field(
        default=None, description="Name of the readout amplifier component."
    )
