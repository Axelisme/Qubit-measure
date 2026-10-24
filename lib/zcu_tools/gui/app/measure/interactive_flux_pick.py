"""Result and analyze-params types shared by the measure flux-dependence adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FluxPickParams:
    """Empty analysis parameter carrier; the caller fixes the display projection."""


@dataclass
class FluxPickResult:
    """Accepted device-axis line positions and period for measure analysis.

    flx_half is the half-flux position, flx_int the integer-flux position, and
    flx_period is twice their absolute distance, all in input device-axis units.
    No context is retained."""

    flx_half: float
    flx_int: float
    flx_period: float
