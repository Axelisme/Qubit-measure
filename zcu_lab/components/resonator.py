"""Readout resonator component model."""

from pydantic import Field

from .base import BuiltinComponent, Frequency


class ResonatorSchema(BuiltinComponent):
    freq: Frequency = Field(default=None, description="Resonator frequency.")
    kappa: Frequency = Field(default=None, description="Resonator linewidth.")
    amplifier: str | None = Field(
        default=None, description="Name of the readout amplifier component."
    )
