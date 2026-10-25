"""Clamp runtime DAC gains where workflows build acquisition cfgs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GainClamp:
    """One changed DAC gain: field identifies its use, before/after are native gains."""

    field: str
    before: float
    after: float


def clamp_gain(
    value: float,
    field: str,
    changes: list[GainClamp] | None = None,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> float:
    """Clip a finite runtime gain to DAC limits and optional narrower policy bounds.

    lower/upper retain the caller's existing floor/cap within [-1, 1]. Append
    only changed values to changes when supplied; no record means no truncation.
    Positive-gain callers supply their existing positive floor as lower.
    """
    clipped = min(1.0, upper, max(-1.0, lower, value))
    if changes is not None and clipped != value:
        changes.append(GainClamp(field, value, clipped))
    return clipped
