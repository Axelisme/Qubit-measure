"""Sampling density diagnostics for fitted fringes and known frequency bounds."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FringeSamplingQuality:
    """Delay-grid density relative to the supplied fringe frequency bounds.

    max_gap_us is the largest gap between sorted unique delays in us, or None
    when fewer than two distinct delays exist. samples_per_cycle is
    1 / (max_gap_us * max(abs(fit), abs(expected))), or None without a gap or
    nonzero frequency.
    status is adequate for at least ten samples/cycle, undersampled below ten,
    or unavailable when no gap, no nonzero frequency, or no complete cycle exists
    in the scan window. reason is no_gap, zero_frequency or incomplete_period
    for unavailable results, otherwise None. An incomplete period retains the
    computed samples_per_cycle, but cannot establish a usable fringe fit.

    Adequate only describes the supplied frequencies. Without an expected
    frequency, a low fitted alias can look adequate; even a supplied expected
    rate is a risk lower bound, not proof that higher-frequency aliases are absent.
    This is separate from residual/covariance quality and calibration acceptance.
    """

    max_gap_us: float | None
    samples_per_cycle: float | None
    status: Literal["adequate", "undersampled", "unavailable"]
    reason: Literal["no_gap", "zero_frequency", "incomplete_period"] | None


def assess_fringe_sampling(
    times_us: NDArray[np.float64],
    fringe_frequency_mhz: float,
    expected_frequency_mhz: float | None = None,
) -> FringeSamplingQuality:
    """Assess a fringe's sampling density at the actual fitted delay coordinates.

    times_us is a finite 1-D delay array in us after any skip or missing-IQ mask.
    It is sorted and deduplicated without modifying the input. Frequencies are
    finite MHz values; signs are ignored. expected_frequency_mhz optionally
    supplies a known alias-risk lower bound, such as an applied phase ramp.
    The larger absolute frequency controls the result. A fitted rate alone
    cannot exclude aliasing. Raise ValueError for non-1-D/nonfinite delays or
    nonfinite frequencies. Empty/constant delays and all-zero frequencies return
    unavailable with a reason, as documented on FringeSamplingQuality.
    """
    if times_us.ndim != 1 or not np.all(np.isfinite(times_us)):
        raise ValueError("times_us must be a finite one-dimensional array")
    if not np.isfinite(fringe_frequency_mhz) or (
        expected_frequency_mhz is not None and not np.isfinite(expected_frequency_mhz)
    ):
        raise ValueError("Fringe and expected frequencies must be finite MHz")

    times = np.unique(times_us)
    if times.size < 2:
        return FringeSamplingQuality(None, None, "unavailable", "no_gap")
    max_gap = float(np.max(np.diff(times)))
    frequency = max(
        abs(fringe_frequency_mhz),
        0.0 if expected_frequency_mhz is None else abs(expected_frequency_mhz),
    )
    if frequency == 0.0:
        return FringeSamplingQuality(max_gap, None, "unavailable", "zero_frequency")
    samples = 1.0 / (max_gap * frequency)
    if float(times[-1] - times[0]) * frequency < 1.0:
        return FringeSamplingQuality(
            max_gap, samples, "unavailable", "incomplete_period"
        )
    status = "adequate" if samples >= 10.0 else "undersampled"
    return FringeSamplingQuality(max_gap, samples, status, None)
