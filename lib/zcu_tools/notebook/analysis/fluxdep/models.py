# Physical models and calculations for flux-dependent analysis

"""Physical models for flux-dependent analysis.

This module provides functions for calculating physical models related to
flux-dependent spectroscopy, including energy calculations and transition models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict, NotRequired


class TransitionDict(TypedDict, extra_items=list[tuple[int, int]]):
    r_f: NotRequired[float]
    sample_f: NotRequired[float]


def count_max_evals(transitions: TransitionDict) -> int:
    max_lvl = 0
    for lvls in transitions.values():
        if not isinstance(lvls, list) or len(lvls) == 0:
            continue
        max_lvl = max(max_lvl, *[max(lvl_from, lvl_to) for lvl_from, lvl_to in lvls])
    max_lvl += 1

    return max_lvl


def energy2linearform(
    energies: NDArray[np.float64], transitions: TransitionDict
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    將能量E轉換為線性形式B,C的躍遷頻率,使得aE的能量對應到|aB+C|的躍遷頻率,其中a可以是任意實數

    Parameters:
    energies: numpy 陣列, 形狀 (N, M), 其中 N 是通量數量, M 是能量級別
    transitions: TransitionDict, 定義允許的躍遷

    Returns:
    B: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    C: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    names: list, 過渡名稱
    """
    N, M = energies.shape
    K = np.sum([len(v) for v in transitions.values() if isinstance(v, list)])
    Bs = np.empty((N, K))
    Cs = np.empty((N, K))

    if any(
        map(
            lambda name: transitions.get(name, []),
            ["blue side", "red side", "mirror blue", "mirror red"],
        )
    ):
        if "r_f" not in transitions:
            raise ValueError(
                "r_f is required for blue side, red side, mirror blue, and mirror red transitions"
            )
    r_f = transitions.get("r_f", 0.0)

    if any("mirror" in name for name in transitions.keys()):
        if "r_f" not in transitions:
            raise ValueError(
                "r_f is required for blue side, red side, mirror blue, and mirror red transitions"
            )
    sample_f = transitions.get("sample_f", 0.0)

    idx = 0
    for i, j in transitions.get("transitions", []):  # E = E_ji
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = 0.0
        idx += 1
    for i, j in transitions.get("blue side", []):  # E = E_ji + r_f
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = r_f
        idx += 1
    for i, j in transitions.get("red side", []):  # E = abs(r_f - E_ji)
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = r_f
        idx += 1
    for i, j in transitions.get("mirror", []):  # E = sample_f - E_ji
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = sample_f
        idx += 1
    for i, j in transitions.get("mirror blue", []):  # E = sample_f - E_ji - r_f
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = sample_f - r_f
        idx += 1
    for i, j in transitions.get("mirror red", []):  # E = sample_f - r_f + E_ji
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = sample_f - r_f
        idx += 1

    for n in range(2, M):
        # Transition N
        for i, j in transitions.get(f"transitions{n}", []):  # E = E_ji / n
            Bs[:, idx] = (energies[:, j] - energies[:, i]) / n
            Cs[:, idx] = 0.0
            idx += 1

        # Mirror N
        for i, j in transitions.get(f"mirror{n}", []):  # E = sample_f - E_ji / n
            Bs[:, idx] = -(energies[:, j] - energies[:, i]) / n
            Cs[:, idx] = sample_f
            idx += 1

    return Bs, Cs


def energy2transition(
    energies: NDArray[np.float64], transitions: TransitionDict
) -> tuple[NDArray[np.float64], list[str]]:
    """
    將能量E轉換為躍遷頻率。

    Parameters:
    energies: numpy 陣列, 形狀 (N, M), 其中 N 是通量數量, M 是能量級別
    transitions: TransitionDict, 定義允許的躍遷

    Returns:
    fs: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    names: list, 過渡名稱
    """
    N, M = energies.shape

    B, C = energy2linearform(energies, transitions)
    fs = np.abs(B + C)
    names = []
    for i, j in transitions.get("transitions", []):  # E = E_ji
        names.append(f"{i} -> {j}")
    for i, j in transitions.get("blue side", []):  # E = E_ji + r_f
        names.append(f"{i} -> {j} blue side")
    for i, j in transitions.get("red side", []):  # E = abs(E_ji - r_f)
        names.append(f"{i} -> {j} red side")
    for i, j in transitions.get("mirror", []):  # E = 2 * sample_f - E_ji
        names.append(f"{i} -> {j} mirror")
    for i, j in transitions.get("mirror blue", []):  # E = 2 * sample_f - E_ji - r_f
        names.append(f"{i} -> {j} mirror blue")
    for i, j in transitions.get("mirror red", []):  # E = 2 * sample_f - r_f + E_ji
        names.append(f"{i} -> {j} mirror red")
    for n in range(2, M):
        for i, j in transitions.get(f"transitions{n}", []):  # E = E_ji / n
            names.append(f"{n} {i} -> {j}")
        for i, j in transitions.get(f"mirror{n}", []):  # E = 2 * sample_f - E_ji / n
            names.append(f"{n} {i} -> {j} mirror")

    return fs, names
