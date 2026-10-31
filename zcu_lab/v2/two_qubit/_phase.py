"""Logical phase from independently acquired X90/Y90 projection shots."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._sums import marginal_probabilities


@dataclass(frozen=True)
class QuadraturePhase:
    """GE-normalized contrast and logical Bloch phase in degrees, wrapped [-180,180).

    Unidentifiable phase is None with a reason, never a fabricated zero.
    Contrast describes thermal ensemble coherence, not pure-state fidelity.
    """

    contrast: float
    logical_phase_deg: float | None
    reason: str | None = None


def wrap_phase(degrees: float) -> float:
    """Wrap a finite logical angle in degrees to [-180,180)."""
    if not np.isfinite(degrees):
        raise ValueError("Logical phase must be finite")
    return float((degrees + 180) % 360 - 180)


def count_phase(x: float, y: float) -> QuadraturePhase:
    """Return phase atan2(y,x) in degrees from classified sigma quadratures.

    x/y are (N_e-N_g)/N_total from independent target projections, including
    other in each denominator. Empty or unmeasured projections have NaN sigma
    and yield a reason. Zero contrast is undefined; weak contrast is not gated.
    """
    if not np.all(np.isfinite([x, y])):
        return QuadraturePhase(np.nan, None, "Missing or empty X/Y projection counts")
    contrast = float(np.hypot(x, y))
    if contrast == 0:
        return QuadraturePhase(contrast, None, "Zero quadrature contrast")
    return QuadraturePhase(contrast, wrap_phase(float(np.degrees(np.arctan2(y, x)))))


def sigma_variance(
    counts: NDArray[np.float64], target: Literal["f1", "f2"]
) -> NDArray[np.float64]:
    """Return count-derived variance of sigma for terminal (...,3,3) counts.

    g/e/other form one multinomial: Var(sigma)=(P_e+P_g-sigma²)/N.
    This includes the negative covariance of the two binomial marginals.
    Zero totals and unmeasured counts yield NaN without divide warnings.
    """
    marginal = marginal_probabilities(counts, target)
    total = counts.sum(axis=(-2, -1))
    value = marginal[..., 1] - marginal[..., 0]
    return np.divide(
        np.maximum(0, marginal[..., 0] + marginal[..., 1] - value**2),
        total,
        out=np.full_like(total, np.nan),
        where=total > 0,
    )


def phase_variance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    var_x: NDArray[np.float64],
    var_y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Propagate independent X/Y sigma variances through atan2, in degree².

    Arrays have matching point axes; zero contrast yields NaN. Other remains
    in each sigma denominator. No variance floor or purity correction is applied.
    """
    radius_squared = x**2 + y**2
    return (
        np.divide(
            y**2 * var_x + x**2 * var_y,
            radius_squared**2,
            out=np.full_like(radius_squared, np.nan),
            where=radius_squared > 0,
        )
        * (180 / np.pi) ** 2
    )
