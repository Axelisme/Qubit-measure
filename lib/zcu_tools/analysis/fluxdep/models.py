from __future__ import annotations

from typing import NotRequired

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict  # extra_items (PEP 728) not in stdlib 3.13

from zcu_tools.analysis.spectrum import SpectrumData


class TransitionDict(TypedDict, extra_items=list[tuple[int, int]]):
    r_f: NotRequired[float]
    sample_f: NotRequired[float]


class PointsData(TypedDict):
    dev_values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]


class DeviceSnapshotValue(TypedDict):
    """One device's fixed native value and its nonempty unit at acquisition."""

    value: float
    unit: str


class SpectrumResult(TypedDict):
    """Processed spectrum with its acquisition-specific affine flux mapping.

    offset is reduced flux at zero scanned-device value, including all fixed
    devices and flux jumps. period is the signed nonzero native span per Φ0.
    device names the scanned instrument; unit describes its native axis.
    spectrum and points retain their stored arrays, including reduced fluxs.
    type optionally identifies the picker. device_snapshot optionally preserves
    acquisition values, never additional contributions to this affine mapping.
    """

    type: NotRequired[str]
    offset: float
    period: float
    device: str
    unit: str
    device_snapshot: NotRequired[dict[str, DeviceSnapshotValue]]
    spectrum: SpectrumData
    points: PointsData


def validate_transition_frequencies(transitions: TransitionDict) -> None:
    """Raise ValueError when a non-empty category lacks its frequency.

    Side-band categories (blue side, red side, mirror blue, mirror red) need
    r_f. Every mirror category (mirror, mirror blue, mirror red, mirror{n})
    needs sample_f. Empty categories need nothing.
    """
    if (
        any(
            transitions.get(name, [])
            for name in ("blue side", "red side", "mirror blue", "mirror red")
        )
        and "r_f" not in transitions
    ):
        raise ValueError(
            "r_f is required for blue side, red side, mirror blue, and mirror red transitions"
        )
    if (
        any("mirror" in name and pairs for name, pairs in transitions.items())
        and "sample_f" not in transitions
    ):
        raise ValueError("sample_f is required for mirror transitions")


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

    validate_transition_frequencies(transitions)
    r_f = transitions.get("r_f", 0.0)
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
    _, M = energies.shape

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
