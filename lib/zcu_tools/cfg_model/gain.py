"""DAC full-scale fractions in a.u., constrained to [-1, 1]."""

from typing import Annotated

from pydantic import Field

Gain = Annotated[float, Field(ge=-1.0, le=1.0)]
"""Finite DAC full-scale fraction in a.u.; both -1 and 1 are accepted."""
