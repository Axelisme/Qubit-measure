from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from numpy.typing import NDArray


class RawShotProgram(Protocol):
    @property
    def ro_chs(self) -> object: ...

    def get_raw(self) -> Sequence[NDArray[Any]] | None: ...


def raw_shots_to_signal(program: RawShotProgram) -> NDArray[np.complex128]:
    acc_buf = program.get_raw()
    if acc_buf is None:
        raise RuntimeError("program did not expose raw shot buffer")
    ro_chs = program.ro_chs
    if not isinstance(ro_chs, Mapping):
        raise TypeError("program readout channels must be a mapping")
    if len(ro_chs) != 1:
        raise ValueError(
            "singleshot raw conversion requires exactly one readout channel"
        )

    (ro_info,) = ro_chs.values()
    if not isinstance(ro_info, Mapping):
        raise TypeError("readout channel info must be a mapping")
    length = ro_info.get("length")
    if not isinstance(length, int | float):
        raise TypeError("readout channel length must be numeric")

    avgiq = acc_buf[0] / float(length)
    i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]
    return np.asarray(i0 + 1j * q0, dtype=np.complex128)


def raw_population_signal(
    raw: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    return raw[0][0]


def calc_populations(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    g_pop, e_pop = signals[..., 0], signals[..., 1]
    return np.stack([g_pop, e_pop, 1 - g_pop - e_pop], axis=-1)


def correct_populations(
    populations: NDArray[np.float64],
    confusion_matrix: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    # Apply readout confusion-matrix correction. None -> returned unchanged
    # (deliberate no-op contract). A singular/non-square matrix raises via
    # np.linalg.inv (Fast-Fail preserved).
    if confusion_matrix is None:
        return populations
    populations = populations @ np.linalg.inv(confusion_matrix)
    return np.clip(populations, 0.0, 1.0)


def classify_result(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Classify shots into ground, excited, and other."""
    dists_g = np.abs(signals - np.array(g_center))
    dists_e = np.abs(signals - np.array(e_center))
    mask_g = dists_g < radius
    mask_e = dists_e < radius
    mask_o = ~(mask_g | mask_e)
    return mask_g, mask_e, mask_o


def plot_with_classified(
    ax: Axes,
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
    max_point: int = 5000,
) -> None:
    # Classify shots
    mask_g, mask_e, mask_o = classify_result(signals, g_center, e_center, radius)

    colors = np.full(signals.shape, "gray", dtype=object)
    colors[mask_g] = "blue"
    colors[mask_e] = "red"
    colors[mask_o] = "green"

    # plot shots
    num_sample = signals.shape[0]
    downsample_num = min(max_point, num_sample)
    downsample_mask = np.arange(0, num_sample, max(1, num_sample // downsample_num))
    downsample_signals = signals[downsample_mask]
    ax.scatter(
        downsample_signals.real,
        downsample_signals.imag,
        c=colors[downsample_mask].tolist(),
        s=1,
        alpha=0.5,
    )

    # plot centers with circle
    plt_params = dict(linestyle=":", color="k", marker="o", markersize=5)
    ax.plot(
        g_center.real,
        g_center.imag,
        markerfacecolor="b",
        label="Ground",
        **plt_params,  # type: ignore
    )
    ax.plot(
        e_center.real,
        e_center.imag,
        markerfacecolor="r",
        label="Excited",
        **plt_params,  # type: ignore
    )
    ax.add_patch(
        Circle(
            (g_center.real, g_center.imag),
            radius,
            color="b",
            fill=False,
            linestyle="--",
        )
    )
    ax.add_patch(
        Circle(
            (e_center.real, e_center.imag),
            radius,
            color="r",
            fill=False,
            linestyle="--",
        )
    )

    ax.set_aspect("equal")
    ax.legend()
    ax.set_xlabel("I value (a.u.)")
    ax.set_ylabel("Q value (a.u.)")
