# Physical models and calculations for flux-dependent analysis

"""Physical models for flux-dependent analysis.

This module provides functions for calculating physical models related to
flux-dependent spectroscopy, including energy calculations and transition models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import TransitionDict


def count_max_evals(transitions: TransitionDict) -> int:
    max_lvl = 0
    for lvls in transitions.values():
        if not isinstance(lvls, list) or len(lvls) == 0:
            continue
        max_lvl = max(max_lvl, *[max(lvl_from, lvl_to) for lvl_from, lvl_to in lvls])
    max_lvl += 1

    return max_lvl


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
    if any(
        transitions.get(name, [])
        for name in ("blue side", "red side", "mirror blue", "mirror red")
    ):
        if "r_f" not in transitions:
            raise ValueError(
                "r_f is required for blue side, red side, mirror blue, and mirror red transitions"
            )
    if any("mirror" in name for name in transitions.keys()):
        if "r_f" not in transitions and "sample_f" not in transitions:
            raise ValueError("sample_f is required for mirror transitions")

    r_f = float(transitions.get("r_f", 0.0))  # type: ignore[arg-type]
    sample_f = float(transitions.get("sample_f", 0.0))  # type: ignore[arg-type]

    pairs: list[tuple[int, int]] = []
    coeffs: list[float] = []
    offsets: list[float] = []

    def _add(group_pairs, c: float, off: float) -> None:
        for i, j in group_pairs:
            pairs.append((i, j))
            coeffs.append(c)
            offsets.append(off)

    _add(transitions.get("transitions", []), 1.0, 0.0)
    _add(transitions.get("blue side", []), 1.0, r_f)
    _add(transitions.get("red side", []), -1.0, r_f)
    _add(transitions.get("mirror", []), -1.0, sample_f)
    _add(transitions.get("mirror blue", []), -1.0, sample_f - r_f)
    _add(transitions.get("mirror red", []), 1.0, sample_f - r_f)

    for n in range(2, M):
        _add(transitions.get(f"transitions{n}", []), 1.0 / n, 0.0)
        _add(transitions.get(f"mirror{n}", []), -1.0 / n, sample_f)

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
