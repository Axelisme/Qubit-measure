"""Logical phase from independently acquired X90/Y90 projection shots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_slopes import PhaseSlopeFit, fit_phase_slopes

from ._sums import qubit_marginal_probabilities, qubit_sigma


@dataclass(frozen=True)
class QuadraturePhase:
    """GE-normalized contrast and logical Bloch phase in degrees, wrapped [-180,180).

    Unidentifiable phase is None with a reason, never a fabricated zero.
    Contrast describes thermal ensemble coherence, not pure-state fidelity.
    """

    contrast: float
    logical_phase_deg: float | None
    reason: str | None = None


@dataclass(frozen=True)
class CountQuadratures:
    """Aligned classified projection reductions on the leading axes of the counts.

    Values are dimensionless sigma; variances are count-derived sigma variances.
    Acquired is each projection's retained shot count. Empty counts give NaN
    values/variances. Incomplete projections in a fit group have NaN acquired.
    """

    values: NDArray[np.float64]
    variances: NDArray[np.float64]
    acquired: NDArray[np.float64]

    def for_fit(
        self, complete: NDArray[np.bool_], *, sign: int = 1
    ) -> CountQuadratures:
        """Mask incomplete projections and multiply X/Y by the preparation sign.

        Complete is a matching boolean array. Sign is +1, or -1 for a group
        with inverted target preparation. Variances and original inputs stay intact.
        """
        return CountQuadratures(
            np.where(complete, sign * self.values, np.nan),
            self.variances,
            np.where(complete, self.acquired, np.nan),
        )


@dataclass(frozen=True)
class PhaseCrossings:
    """All interpolated π modulo 2π frequencies in MHz and an identifiability reason.

    A unique finite crossing has reason=None. No/multiple crossings, incomplete
    phase or a degenerate π plateau retain a reason; no preferred root is chosen.
    No uncertainty is inferred from interpolation without an uncertainty model.
    """

    frequencies_mhz: tuple[float, ...]
    reason: str | None


def count_quadratures(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> CountQuadratures:
    """Reduce terminal (...,3,3) g/e/other counts for the selected qubit.

    Return sigma, count variance and retained shots on the preceding axes.
    Counts are unchanged; an invalid target raises ValueError.
    """
    return CountQuadratures(
        qubit_sigma(counts, target),
        qubit_sigma_variance(counts, target),
        counts.sum(axis=(-2, -1)),
    )


def fit_count_phase_slopes(
    n: NDArray[np.int64], *groups: CountQuadratures
) -> PhaseSlopeFit:
    """Fit one shared per-gate slope per spectator/frequency from masked groups.

    Supply the integer CZ counts and at least one group, in observation order.
    Groups must have the same shape. Preserve the fitter's SE and failure reasons.
    """
    return fit_phase_slopes(
        n,
        np.stack([group.values for group in groups]),
        np.stack([group.variances for group in groups]),
        np.stack([group.acquired for group in groups]),
    )


def complete_projections(
    shots: NDArray[np.float64], expected_shots: int
) -> NDArray[np.bool_]:
    """Mark projections whose total across all classes matches the planned shots."""
    return np.isclose(shots, expected_shots, rtol=8 * np.finfo(float).eps, atol=0)


def conditional_phase_deg(fit: PhaseSlopeFit) -> NDArray[np.float64]:
    """Return e minus g slope in degrees per CZ, in [-180,180) for crossing."""
    return (fit.slopes_deg[1] - fit.slopes_deg[0] + 180) % 360 - 180


def phase_near(
    phase_deg: NDArray[np.float64], reference_deg: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Choose the 360° representative in [reference-180, reference+180).

    Reference is a scalar or aligned angle array, all in degrees. NaN inputs
    stay NaN. This representation does not fit, smooth or fill measured data.
    """
    return reference_deg + (phase_deg - reference_deg + 180) % 360 - 180


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


def qubit_sigma_variance(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> NDArray[np.float64]:
    """Return the named qubit's multinomial sigma variance from (...,3,3) counts.

    Terminal axes are Qubit1 then Qubit2, each g/e/other. Return
    (P_e+P_g-sigma²)/N, including other in N. Empty or unmeasured points yield
    NaN; invalid targets raise ValueError.
    """
    return _sigma_variance(counts, qubit_marginal_probabilities(counts, target))


def _sigma_variance(
    counts: NDArray[np.float64], marginal: NDArray[np.float64]
) -> NDArray[np.float64]:
    total = counts.sum(axis=(-2, -1))
    value = marginal[..., 1] - marginal[..., 0]
    return np.divide(
        np.maximum(0, marginal[..., 0] + marginal[..., 1] - value**2),
        total,
        out=np.full_like(total, np.nan),
        where=total > 0,
    )


def phase_display_sem(
    variance: NDArray[np.float64],
    x_shots: NDArray[np.float64],
    y_shots: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mask display bars unless both projections retain at least two shots."""
    return np.sqrt(np.where((x_shots >= 2) & (y_shots >= 2), variance, np.nan))


def quadrature_phase_sem(group: CountQuadratures) -> NDArray[np.float64]:
    """Return pointwise shot SEM of phase in degrees for batched X/Y reductions.

    Group axes are series × X/Y × scan axes. The result omits the X/Y axis.
    Missing phase or fewer than two retained shots in either projection yield NaN.
    """
    variances = phase_variance(
        group.values[:, 0],
        group.values[:, 1],
        group.variances[:, 0],
        group.variances[:, 1],
    )
    return phase_display_sem(variances, group.acquired[:, 0], group.acquired[:, 1])


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
