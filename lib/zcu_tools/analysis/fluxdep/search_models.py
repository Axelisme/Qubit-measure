# Physical models and calculations for flux-dependent analysis

"""Physical models for flux-dependent analysis.

This module provides functions for calculating physical models related to
flux-dependent spectroscopy, including energy calculations and transition models.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import (
    TransitionDict,
    validate_transition_frequencies,
)


def count_max_evals(transitions: TransitionDict) -> int:
    max_lvl = 0
    for lvls in transitions.values():
        if not isinstance(lvls, list) or len(lvls) == 0:
            continue
        max_lvl = max(max_lvl, *[max(lvl_from, lvl_to) for lvl_from, lvl_to in lvls])
    max_lvl += 1

    return max_lvl


def _transition_groups(
    transitions: TransitionDict, levels: int
) -> Iterator[tuple[str, float, float]]:
    """Keep diagnostic labels in the same order as the numeric compilation."""
    r_f = float(transitions.get("r_f", 0.0))
    sample_f = float(transitions.get("sample_f", 0.0))
    yield "transitions", 1.0, 0.0
    yield "blue side", 1.0, r_f
    yield "red side", -1.0, r_f
    yield "mirror", -1.0, sample_f
    yield "mirror blue", -1.0, sample_f - r_f
    yield "mirror red", 1.0, sample_f - r_f
    for n in range(2, levels):
        yield f"transitions{n}", 1.0 / n, 0.0
        yield f"mirror{n}", -1.0 / n, sample_f


def transition_labels(transitions: TransitionDict, levels: int) -> tuple[str, ...]:
    """Return ``category i-j`` labels in compile_transitions column order.

    levels is the database energy-level count. Pair order and duplicate pairs
    are preserved within each category; empty categories produce no labels.
    Numeric frequency validation belongs to compile_transitions.
    """
    return tuple(
        f"{category} {i}-{j}"
        for category, _, _ in _transition_groups(transitions, levels)
        for i, j in transitions.get(category, [])
    )


def compile_transitions(
    transitions: TransitionDict, M: int
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """Pre-compile a TransitionDict into (pairs, coeffs, offsets) arrays so
    the per-energy linear-form computation can run in njit(nogil=True) code.

    For each transition k, the model is:
        f_k(a, n) = |a * coeffs[k] * (E_j - E_i) + offsets[k]|
    where (i, j) = pairs[k].

    M = energies.shape[1] — needed to know which `transitions{n}` / `mirror{n}`
    keys to consume.
    """
    validate_transition_frequencies(transitions)

    pairs: list[tuple[int, int]] = []
    coeffs: list[float] = []
    offsets: list[float] = []

    def _add(group_pairs, c: float, off: float) -> None:
        for i, j in group_pairs:
            pairs.append((i, j))
            coeffs.append(c)
            offsets.append(off)

    for category, coefficient, offset in _transition_groups(transitions, M):
        _add(transitions.get(category, []), coefficient, offset)

    if not pairs:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    return (
        np.asarray(pairs, dtype=np.int32),
        np.asarray(coeffs, dtype=np.float64),
        np.asarray(offsets, dtype=np.float64),
    )
