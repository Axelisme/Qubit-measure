"""Named numeric gates shared by spectroscopy and lifetime stages."""

import math
import operator
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Gate(BaseModel):
    """One numeric decision with its comparison, threshold and observed value.

    actual is null for unavailable/nonfinite values, which cannot pass.
    applied=False marks a diagnostic comparison not used for acceptance.
    name identifies the metric and its decision stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    name: str
    comparison: Literal["<", "<=", ">", ">=", "=="]
    threshold: float
    actual: float | None
    passed: bool
    applied: bool = True


def gate(
    name: str,
    comparison: Literal["<", "<=", ">", ">=", "=="],
    threshold: float,
    actual: float | None,
    *,
    applied: bool = True,
) -> Gate:
    """Record a comparison; missing/nonfinite observations fail, not JSON NaN.

    applied=False reports a diagnostic comparison not used at this stage.
    Thresholds must be finite. Callers supply values in the threshold's units.
    """
    compare = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
    }[comparison]
    finite = actual if actual is not None and math.isfinite(actual) else None
    return Gate(
        name=name,
        comparison=comparison,
        threshold=threshold,
        actual=finite,
        passed=finite is not None and compare(finite, threshold),
        applied=applied,
    )
