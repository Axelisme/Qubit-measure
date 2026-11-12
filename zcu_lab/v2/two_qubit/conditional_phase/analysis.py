"""Count-weighted phase-slope fits. This module contains no plotting or run policy."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import phase_slopes

from .._phase import QuadraturePhase, count_phase, sigma_variance
from .._sums import sigma
from .crossing import PhaseCrossings, phase_crossings


@dataclass(frozen=True)
class CountQuadratures:
    """Count reductions with spectator × quadrature × n × frequency axes.

    values is target sigma, variances is its count-derived variance, acquired
    is each projection's shot total. Empty counts give NaN values/variances.
    """

    values: NDArray[np.float64]
    variances: NDArray[np.float64]
    acquired: NDArray[np.float64]


def count_quadratures(
    counts: NDArray[np.float64],
    target: Literal["f1", "f2"],
) -> CountQuadratures:
    """Reduce terminal (...,3,3) counts to sigma, variance and shot totals."""
    return CountQuadratures(
        sigma(counts, target),
        sigma_variance(counts, target),
        counts.sum(axis=(-2, -1)),
    )


@dataclass(frozen=True)
class ConditionalPhaseAnalysis:
    """Unique complete π crossing in MHz, or None with frequency_reason.

    Per-frequency phase, slope and count diagnostics are only used for plots.
    """

    frequency_mhz: float | None
    frequency_reason: str | None


def fitted_crossings(
    frequencies: NDArray[np.float64],
    conditional_deg: NDArray[np.float64],
    fit: phase_slopes.PhaseSlopeFit,
) -> PhaseCrossings:
    """Find per-gate π crossings, preserving fit-unavailability reasons.

    Inputs are the increasing frequency axis, its conditional slope difference,
    and the fits that produced it. Any unavailable slope prevents selection.
    """
    crossings = phase_crossings(frequencies, conditional_deg)
    reasons = [reason for state in fit.reasons for reason in state if reason]
    if reasons:
        return PhaseCrossings((), "; ".join(dict.fromkeys(reasons)))
    return crossings


def phase_points(
    quadratures: NDArray[np.float64],
) -> tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]:
    """Convert (spectator,quadrature,n,frequency) sigma to diagnostic phase points."""
    return tuple(
        tuple(
            tuple(count_phase(float(x), float(y)) for x, y in zip(xs, ys, strict=True))
            for xs, ys in zip(state[0], state[1], strict=True)
        )
        for state in quadratures
    )
