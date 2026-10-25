"""DAC full-scale fractions in a.u., constrained to [-1, 1]."""

from typing import Annotated

from pydantic import AfterValidator, Field
from qick.asm_v2 import QickParam


def validate_gain(value: float) -> float:
    """Return a finite DAC full-scale fraction, or raise ValueError with its value."""
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"Gain must be within [-1, 1], got {value}")
    return value


Gain = Annotated[float, AfterValidator(validate_gain), Field(ge=-1.0, le=1.0)]
"""Finite DAC full-scale fraction in a.u.; both -1 and 1 are accepted."""


def _validate_gain_param(value: float | QickParam) -> float | QickParam:
    if isinstance(value, QickParam):
        validate_gain(value.start)
        spans: dict[str, float] = value.spans
        # Independent loop spans combine at corners. QICK minval/maxval only
        # include one span, so they do not bound a multidimensional gain sweep.
        validate_gain(value.start + sum(min(span, 0.0) for span in spans.values()))
        validate_gain(value.start + sum(max(span, 0.0) for span in spans.values()))
    return value


GainParam = Annotated[Gain | QickParam, AfterValidator(_validate_gain_param)]
"""DAC fraction or QICK sweep whose entire multidimensional range is in [-1, 1]."""
