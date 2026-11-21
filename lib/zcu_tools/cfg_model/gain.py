"""DAC full-scale fractions in a.u., constrained to [-1, 1]."""

from typing import Annotated

from pydantic import Field

from .sweep_bounds import SweepBounds

FULL_SCALE_GAIN_BOUNDS = SweepBounds(minimum=-1.0, maximum=1.0)
"""Inclusive DAC full-scale limits in a.u., shared by scalar and sweep gain."""

Gain = Annotated[
    float, Field(ge=FULL_SCALE_GAIN_BOUNDS.minimum, le=FULL_SCALE_GAIN_BOUNDS.maximum)
]
"""Finite DAC full-scale fraction in a.u.; both -1 and 1 are accepted."""
