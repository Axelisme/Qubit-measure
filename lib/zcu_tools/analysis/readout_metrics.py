"""Two-state readout separation and QND estimates from repeated IQ moments."""

from dataclasses import dataclass
from math import erfc, nan, sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReadoutMetrics:
    """Dimensionless scalar estimates; defaults represent unavailable data.

    All fields are floats. No probabilities are clipped to [0, 1].
    Every field is NaN if either calibration branch (2 or 6) has zero shots,
    the R1 preparation contrast is zero, or inferred separation squared is <= 0.

    snr: State separation Delta divided by isotropic noise sigma; NaN also
        when inferred noise variance is nonpositive.
    f_sep: Equal-prior midpoint Gaussian separation fidelity; NaN with snr.
    thermal_population: Excited-state fraction before the preparation pulse.
    retention_on: Preparation contrast retained through R1 and the wait (S).
    retention_off: Contrast retained with R1 disabled, over the same duration.
    qnd_on: (1 + retention_on) / 2, including the wait.
    qnd_off: (1 + retention_off) / 2, the no-measurement baseline.
    flip_up_on: Ground-to-excited probability through R1 and the wait.
    flip_up_off: Ground-to-excited probability with R1 disabled.
    flip_down_on: Excited-to-ground probability through R1 and the wait.
    flip_down_off: Excited-to-ground probability with R1 disabled.
    pi_miss_on: Fraction of ineffective middle pi pulses after R1.
    pi_miss_off: Fraction of ineffective middle pi pulses with R1 disabled.
    retention_from_covariance: Mean projected R1/R2 covariance of calibration
        branches divided by excess R1 variance v; NaN also when v is zero.

    Off retention, QND and flips additionally require nonzero shots in branches
    0 and 4. On values use calibration branches 2 and 6. Each pi_miss additionally
    requires nonzero shots in its middle-pi branches (1/5 off, 3/7 on) and a
    nonzero corresponding retention. Missing branches affect only dependent
    fields. NaN required moments propagate through the formulas.
    """

    snr: float = nan
    f_sep: float = nan
    thermal_population: float = nan
    retention_on: float = nan
    retention_off: float = nan
    qnd_on: float = nan
    qnd_off: float = nan
    flip_up_on: float = nan
    flip_up_off: float = nan
    flip_down_on: float = nan
    flip_down_off: float = nan
    pi_miss_on: float = nan
    pi_miss_off: float = nan
    retention_from_covariance: float = nan


def _transition_metrics(
    projected: NDArray[np.float64],
    low: int,
    contrast: float,
    separation: float,
    ground: float,
) -> tuple[float, float, float, float, float]:
    retention = float((projected[low + 4] - projected[low]) / contrast)
    asymmetry = float(
        (projected[low] + projected[low + 4] - 2 * ground) / separation - 1
    )
    middle_retention = float((projected[low + 5] - projected[low + 1]) / contrast)
    return (
        retention,
        (1 + retention) / 2,
        (1 - retention + asymmetry) / 2,
        (1 - retention - asymmetry) / 2,
        (retention + middle_retention) / (2 * retention) if retention != 0 else nan,
    )


def readout_metrics(sums: NDArray[np.float64]) -> ReadoutMetrics:
    """Estimate separation, thermal population and on/off QND at one scan point.

    sums is float64 with shape (8, 6, 6), in common arbitrary ADC units.
    Each branch stores the shot-summed outer product of R1 and R2 features,
    each ordered (1, I, Q, I², IQ, Q²). sums[:, 0, 0] contains shot counts.
    Branch index is prepare_pi*4 + r1_on*2 + middle_pi (each flag is 0 or 1).
    Invalid shape raises ValueError. The input is not modified; no I/O occurs.

    Calibration uses equally weighted R1 means and within-preparation covariance
    matrices from branches 2 and 6, not covariance of pooled shots. Noise must
    be isotropic in IQ, preparation pi ideal, and R1/R2 have identical response
    and independent Gaussian noise. Both on/off dynamics use this calibration.
    Middle-pi miss estimates assume independent ineffective-pulse probability.
    Covariance retention uses only branches 2 and 6.

    Zero-shot or degenerate data produce NaN in dependent fields as documented
    by ReadoutMetrics. Estimates outside physical ranges remain unclipped,
    including negative excess variance v when separation squared stays positive.
    No leakage correction or automatic parameter selection is performed.
    """
    if sums.shape != (8, 6, 6):
        raise ValueError(f"sums must have shape (8, 6, 6), got {sums.shape}")
    counts = sums[:, 0, 0]
    if counts[2] == 0 or counts[6] == 0:
        return ReadoutMetrics()
    counts = np.where(counts == 0, np.nan, counts)
    first = sums[:, 1:3, 0] / counts[:, None]
    second = sums[:, 0, 1:3] / counts[:, None]
    covariance = sums[:, [[3, 4], [4, 5]], 0] / counts[:, None, None]
    covariance = covariance - first[:, :, None] * first[:, None, :]
    shared_covariance = (covariance[2] + covariance[6]) / 2

    difference = first[6] - first[2]
    contrast = float(np.linalg.norm(difference))
    if not contrast > 0:
        return ReadoutMetrics()
    direction = difference / contrast
    perpendicular = np.array([-direction[1], direction[0]])
    noise_variance = float(perpendicular @ shared_covariance @ perpendicular)
    excess_variance = float(direction @ shared_covariance @ direction) - noise_variance
    separation_squared = contrast**2 + 4 * excess_variance
    if not separation_squared > 0:
        return ReadoutMetrics()
    separation = sqrt(separation_squared)
    snr = separation / sqrt(noise_variance) if noise_variance > 0 else nan
    population = (1 - contrast / separation) / 2
    ground = float(first[2] @ direction) - separation * population

    projected = second @ direction
    on = _transition_metrics(projected, 2, contrast, separation, ground)
    off = _transition_metrics(projected, 0, contrast, separation, ground)
    cross = sums[:, 1:3, 1:3] / counts[:, None, None]
    cross = cross - first[:, :, None] * second[:, None, :]
    projected_cross = float(direction @ ((cross[2] + cross[6]) / 2) @ direction)
    return ReadoutMetrics(
        snr=snr,
        f_sep=1 - 0.5 * erfc(snr / (2 * sqrt(2))),
        thermal_population=population,
        retention_on=on[0],
        retention_off=off[0],
        qnd_on=on[1],
        qnd_off=off[1],
        flip_up_on=on[2],
        flip_up_off=off[2],
        flip_down_on=on[3],
        flip_down_off=off[3],
        pi_miss_on=on[4],
        pi_miss_off=off[4],
        retention_from_covariance=(
            projected_cross / excess_variance if excess_variance != 0 else nan
        ),
    )
