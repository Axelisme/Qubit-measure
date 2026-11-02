"""DAC full-scale fractions in a.u., constrained to [-1, 1]."""

from typing import Annotated

from pydantic import AfterValidator, Field


def validate_gain(value: float) -> float:
    """Return a finite DAC full-scale fraction, or raise ValueError with its value."""
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"Gain must be within [-1, 1], got {value}")
    return value


Gain = Annotated[float, AfterValidator(validate_gain), Field(ge=-1.0, le=1.0)]
"""Finite DAC full-scale fraction in a.u.; both -1 and 1 are accepted."""
