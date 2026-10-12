"""Numerical shot classification and population conversion."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from zcu_tools.utils.shot_classification import classify_shots


def raw_population_signal(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return the first readout's first population array from acquisition raw data.

    raw has the program's readout/round nesting; indexing errors propagate and
    the returned float array is borrowed, not copied."""
    return raw[0][0]


def calc_populations(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Append other-state probability 1-g-e to final-axis ground/excited fractions.

    signals has final axis containing at least g/e; returns a new float array
    with final axis (g, e, other). Does not clamp or renormalize inputs."""
    g_pop, e_pop = signals[..., 0], signals[..., 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1)


def correct_populations(
    populations: NDArray[np.float64],
    confusion_matrix: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    # Apply readout confusion-matrix correction. None -> returned unchanged
    # (deliberate no-op contract). A singular/non-square matrix raises via
    # np.linalg.solve (Fast-Fail preserved).
    """Correct final-axis state fractions with the readout confusion matrix.

    populations has arbitrary leading dimensions and a final state axis.
    confusion_matrix holds true-state rows and measured-state columns; None
    returns populations unchanged. Otherwise solve its transpose and clip to
    [0, 1], returning float64 with the same shape. Singular/non-square or
    incompatible matrices raise NumPy linear-algebra/value errors."""
    if confusion_matrix is None:
        return populations

    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    state_count = populations.shape[-1]
    flat_populations = populations.reshape(-1, state_count)
    corrected = np.linalg.solve(matrix.T, flat_populations.T).T
    corrected = corrected.reshape(populations.shape)
    return np.asarray(np.clip(corrected, 0.0, 1.0), dtype=np.float64)


def classify_result(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Classify complex IQ shots into ground, excited and other boolean masks.

    g_center/e_center are distinct complex centers; radius is their positive
    finite classification radius. Output masks preserve signals.shape, use
    nearest-center regions and assign exact ties/outside circles to other.
    Invalid centers/radius raise ValueError via the classification kernel."""
    return classify_shots(signals, g_center, e_center, radius)


def classify_rabi_iq(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> NDArray[np.float64]:
    """Compute ground/excited fractions for raw IQ of shape (sweep, shot).

    g_center/e_center and radius follow classify_result. Returns float64 of
    shape (sweep, 2); non-2D signals raise ValueError and invalid classification
    geometry propagates. No readout correction or fit is performed."""
    if signals.ndim != 2:
        raise ValueError("Rabi raw IQ must have shape (sweep, shot)")
    g_mask, e_mask, _ = classify_result(signals, g_center, e_center, radius)
    return np.stack((g_mask.mean(axis=1), e_mask.mean(axis=1)), axis=1)
