# Physical models and calculations for flux-dependent analysis

"""Physical models for flux-dependent analysis.

This module provides functions for calculating physical models related to
flux-dependent spectroscopy, including energy calculations and transition models.
"""

from typing import Any, Dict, Tuple

import numpy as np


def count_max_evals(allows: Dict[str, Any]) -> int:
    evals_count = 0
    for name, lvl in allows.items():
        if not isinstance(lvl, list) or len(lvl) == 0 or name == "r_f":
            continue
        evals_count = max(evals_count, *[max(lv) for lv in lvl])
    evals_count += 1

    return evals_count


def energy2linearform(
    energies: np.ndarray, allows: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    將能量E轉換為線性形式B,C的躍遷頻率,使得aE的能量對應到|aB+C|的躍遷頻率,其中a可以是任意實數

    Parameters:
    energies: numpy 陣列, 形狀 (N, M), 其中 N 是通量數量, M 是能量級別
    allows: dict, 允許的過渡

    Returns:
    B: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    C: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    names: list, 過渡名稱
    """
    N, M = energies.shape
    K = np.sum([len(v) for v in allows.values() if isinstance(v, list)])
    Bs = np.empty((N, K))
    Cs = np.empty((N, K))
    idx = 0
    for i, j in allows.get("transitions", []):  # E = E_ji
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = 0.0
        idx += 1
    for i, j in allows.get("blue side", []):  # E = E_ji + r_f
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = allows["r_f"]
        idx += 1
    for i, j in allows.get("red side", []):  # E = abs(r_f - E_ji)
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = allows["r_f"]
        idx += 1

    for i, j in allows.get("mirror", []):  # E = 2 * sample_f - E_ji
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = 2 * allows["sample_f"]
        idx += 1
    for i, j in allows.get("mirror blue", []):  # E = 2 * sample_f - E_ji - r_f
        Bs[:, idx] = -1 * (energies[:, j] - energies[:, i])
        Cs[:, idx] = 2 * allows["sample_f"] - allows["r_f"]
        idx += 1
    for i, j in allows.get("mirror red", []):  # E = 2 * sample_f - r_f + E_ji
        Bs[:, idx] = energies[:, j] - energies[:, i]
        Cs[:, idx] = 2 * allows["sample_f"] - allows["r_f"]
        idx += 1

    for n in range(2, M):
        # Transition N
        for i, j in allows.get(f"transitions{n}", []):  # E = E_ji / n
            Bs[:, idx] = (energies[:, j] - energies[:, i]) / n
            Cs[:, idx] = 0.0
            idx += 1

        # Mirror N
        for i, j in allows.get(f"mirror{n}", []):  # E = 2 * sample_f - E_ji / n
            Bs[:, idx] = -(energies[:, j] - energies[:, i]) / n
            Cs[:, idx] = 2 * allows["sample_f"]
            idx += 1

    return Bs, Cs


def energy2transition(
    energies: np.ndarray, allows: Dict[str, Any]
) -> Tuple[np.ndarray, list]:
    """
    將能量E轉換為躍遷頻率。

    Parameters:
    energies: numpy 陣列, 形狀 (N, M), 其中 N 是通量數量, M 是能量級別
    allows: dict, 允許的過渡

    Returns:
    fs: numpy 陣列, 形狀 (N, K), 其中 N 是通量數量, K 是過渡數量
    names: list, 過渡名稱
    """
    N, M = energies.shape

    B, C = energy2linearform(energies, allows)
    fs = np.abs(B + C)
    names = []
    for i, j in allows.get("transitions", []):  # E = E_ji
        names.append(f"{i} -> {j}")
    for i, j in allows.get("blue side", []):  # E = E_ji + r_f
        names.append(f"{i} -> {j} blue side")
    for i, j in allows.get("red side", []):  # E = abs(E_ji - r_f)
        names.append(f"{i} -> {j} red side")
    for i, j in allows.get("mirror", []):  # E = 2 * sample_f - E_ji
        names.append(f"{i} -> {j} mirror")
    for i, j in allows.get("mirror blue", []):  # E = 2 * sample_f - E_ji - r_f
        names.append(f"{i} -> {j} mirror blue")
    for i, j in allows.get("mirror red", []):  # E = 2 * sample_f - r_f + E_ji
        names.append(f"{i} -> {j} mirror red")
    for n in range(2, M):
        for i, j in allows.get(f"transitions{n}", []):  # E = E_ji / n
            names.append(f"{n} {i} -> {j}")
        for i, j in allows.get(f"mirror{n}", []):  # E = 2 * sample_f - E_ji / n
            names.append(f"{n} {i} -> {j} mirror")

    return fs, names
