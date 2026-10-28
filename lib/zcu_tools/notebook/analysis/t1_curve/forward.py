"""Select observable decay modes for T1 fitting and presentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium import calculate_manifold_t1_vs_flux_fast


def validate_secondary_threshold(secondary_threshold: float) -> None:
    """Require a finite [0,1] secondary-fraction threshold, else ValueError."""
    if not np.isfinite(secondary_threshold) or not 0 <= secondary_threshold <= 1:
        raise ValueError("secondary_threshold must be finite and in [0,1]")


def secondary_exceeds_threshold(
    secondary_fractions: NDArray[np.float64], secondary_threshold: float
) -> NDArray[np.bool_]:
    """Flag signed fractions strictly above a validated finite [0,1] threshold.

    T1FitResult and T1ChannelCurves use this rule for sample and dense arrays.
    Preserve array shape; negative and equal-to-threshold values are not flagged.
    """
    return secondary_fractions > secondary_threshold


class ManifoldPrediction(NamedTuple):
    """Intrinsic prediction arrays in input flux order.

    T1_ns holds the largest-signed-amplitude mode's positive lifetime in ns.
    secondary_fractions is a2/(a1+a2), or zero for a single mode.
    It is not clipped and can be negative.
    """

    T1_ns: NDArray[np.float64]
    secondary_fractions: NDArray[np.float64]


def predict_manifold_t1(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    noise_channels: Sequence[tuple[str, dict[str, float]]],
    Temp: float | NDArray[np.float64],
    *,
    cutoff: int = 40,
    qub_dim: int = 20,
) -> ManifoldPrediction:
    """Predict dominant population-decay T1 and secondary weights per flux.

    Circuit, normalized flux, channels, K temperatures and numerical controls
    follow simulate.fluxonium.calculate_manifold_t1_vs_flux_fast. Choose the
    largest signed amplitude, not the longest lifetime. Keep signed a2/(a1+a2)
    without abs or clipping; one mode gives zero. A flux with no decay modes
    raises ValueError identifying its index and flux. Simulation errors propagate.
    """
    modes = calculate_manifold_t1_vs_flux_fast(
        params, fluxs, noise_channels, Temp, cutoff=cutoff, qub_dim=qub_dim
    )
    t1s = np.empty(len(modes), dtype=np.float64)
    fractions = np.zeros(len(modes), dtype=np.float64)
    for index, point in enumerate(modes):
        if not point:
            raise ValueError(
                f"no decay modes at flux index {index} (flux={fluxs[index]:g})"
            )
        t1s[index] = point[0].t1_ns
        if len(point) > 1:
            fractions[index] = point[1].amplitude / (
                point[0].amplitude + point[1].amplitude
            )
    return ManifoldPrediction(t1s, fractions)
