"""Fluxonium component model with energies in GHz."""

from pydantic import Field

from .base import Energy
from .qubit import QubitBaseSchema


class QubitSchema(QubitBaseSchema):
    """Fluxonium qubit: EJ, EC and EL energies in GHz."""

    EJ: Energy = Field(default=None, description="Josephson energy.")
    EC: Energy = Field(default=None, description="Charging energy.")
    EL: Energy = Field(default=None, description="Inductive energy.")
