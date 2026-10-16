"""SQUID transmon component model with energies in GHz."""

from typing import Annotated

from pydantic import Field

from .base import Energy
from .qubit import QubitBaseSchema


class TransmonSchema(QubitBaseSchema):
    """SQUID transmon: EJmax and EC in GHz, plus dimensionless asymmetry d."""

    EJmax: Energy = Field(default=None, description="Maximum SQUID Josephson energy.")
    EC: Energy = Field(default=None, description="Charging energy.")
    d: Annotated[float | None, Field(strict=True, allow_inf_nan=False)] = Field(
        default=None, description="Dimensionless SQUID junction asymmetry."
    )
