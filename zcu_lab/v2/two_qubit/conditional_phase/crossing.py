"""Adjacent-point phase unwrapping and in-window π crossing diagnostics."""

from dataclasses import dataclass
from math import ceil, floor

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhaseCrossings:
    """All interpolated π modulo 2π frequencies in MHz and an identifiability reason.

    A unique finite crossing has reason=None. No/multiple crossings, incomplete
    phase or a degenerate π plateau retain a reason; no preferred root is chosen.
    No uncertainty is inferred from interpolation without an uncertainty model.
    """

    frequencies_mhz: tuple[float, ...]
    reason: str | None


def phase_crossings(
    frequencies: NDArray[np.float64], phases_deg: NDArray[np.float64]
) -> PhaseCrossings:
    """Find π modulo 2π by adjacent unwrap and piecewise-linear interpolation.

    Inputs are matching 1D frequency MHz and logical phase degree vectors in
    acquisition order. Frequencies must be finite and strictly increasing.
    Nonfinite phases mean incomplete/unidentifiable quadratures, returning a
    reason without bridging gaps. Require at least two points. Unwrap assumes
    phase steps smaller than 180 degrees; exact half-turn steps are ambiguous.
    Flat π segments are degenerate, not a selectable frequency. Endpoints count,
    and a sampled crossing shared by adjacent segments counts once. No default
    proximity, contrast/σ, frequency/σ or cross-n consistency gate is applied.
    """
    if frequencies.ndim != 1 or phases_deg.shape != frequencies.shape:
        raise ValueError("Frequency and phase must be matching 1D arrays")
    if not np.all(np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequencies must be finite and strictly increasing")
    if not np.all(np.isfinite(phases_deg)):
        return PhaseCrossings((), "Missing complete branches or identifiable X/Y phase")
    if len(frequencies) < 2:
        return PhaseCrossings((), "At least two frequency points are required")
    phase = np.unwrap(phases_deg, period=360)
    if np.any(np.abs(np.diff(phase)) == 180):
        return PhaseCrossings((), "Adjacent half-turn phase step has ambiguous unwrap")
    roots: set[float] = set()
    for f0, f1, p0, p1 in zip(
        frequencies[:-1], frequencies[1:], phase[:-1], phase[1:], strict=True
    ):
        lower, upper = sorted((p0, p1))
        for k in range(ceil((lower - 180) / 360), floor((upper - 180) / 360) + 1):
            target = 180 + 360 * k
            if p0 == p1:
                return PhaseCrossings(
                    tuple(sorted(roots)), "Flat π phase segment has no unique frequency"
                )
            root = (
                f0
                if target == p0
                else f1
                if target == p1
                else f0 + (target - p0) * (f1 - f0) / (p1 - p0)
            )
            if not np.isfinite(root):
                return PhaseCrossings((), "Nonfinite interpolated crossing")
            roots.add(float(root))
    values = tuple(sorted(roots))
    reason = (
        None
        if len(values) == 1
        else "No π crossing in scanned frequency window"
        if not values
        else "Multiple π crossings in scanned frequency window"
    )
    return PhaseCrossings(values, reason)
