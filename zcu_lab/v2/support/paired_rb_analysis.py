"""Analysis options, estimate and count validation shared by paired RB leaves."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase


class RBOptions(ConfigBase):
    """Analysis-only reproducible paired-seed bootstrap settings.

    bootstrap_samples is at least 100 whole-seed resamples, not independent
    shots. bootstrap_seed is the nonnegative RNG seed for this uncertainty
    calculation, separate from the acquired sequence identities.
    """

    bootstrap_samples: int = Field(default=1000, ge=100)
    bootstrap_seed: int = Field(default=42, ge=0)


@dataclass(frozen=True)
class RBEstimate:
    """Unitless d=4 RB estimates, conditional on the exponential decay model.

    p_reference/p_interleaved are per random Clifford (plus CZ for interleaved).
    reference_epc=3/4*(1-p_reference). gate_error=3/4*(1-p_i/p_r), not clipped.
    error_ci_low/error_ci_high is the 95% paired-seed percentile interval.
    n_paired_seeds counts seeds with complete curves in both arms.
    bootstrap_successes counts valid resampled fits; warning describes ratio
    and model limitations. No field asserts calibrated physical fidelity.
    """

    p_reference: float
    p_interleaved: float
    reference_epc: float
    gate_error: float
    error_ci_low: float
    error_ci_high: float
    n_paired_seeds: int
    bootstrap_successes: int
    warning: str


def validate_counts(counts: NDArray[np.float64], class_axes: tuple[int, ...]) -> None:
    """Reject negative counts or a cell that is neither all finite nor all NaN.

    counts ends in the given class_axes (negative indices allowed).
    All-NaN cells are unacquired. Leave counts unchanged.
    """
    finite = np.isfinite(counts).all(axis=class_axes)
    missing = np.isnan(counts).all(axis=class_axes)
    if not np.all(finite | missing) or np.any(counts < 0):
        raise ValueError("RB sums contains invalid counts or a partial cell")


def validate_axes(
    axes: tuple[NDArray[np.int64], ...],
    expected: tuple[Sequence[int] | NDArray[np.int64], ...],
) -> None:
    """Raise ValueError when persisted axis identities do not match saved cfg.

    axes and expected list persisted and configured identities in the same axis
    order (seed, depth). Compare exact values; unequal sequence lengths also
    raise ValueError. Return None without modifying either sequence or drawing.
    """
    if any(
        not np.array_equal(actual, wanted)
        for actual, wanted in zip(axes, expected, strict=True)
    ):
        raise ValueError("RB count axes do not match saved cfg")
