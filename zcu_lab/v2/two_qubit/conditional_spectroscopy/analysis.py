"""Count-derived excitation spectra; no peak assignment or calibration fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from .._sums import marginal_probabilities

if TYPE_CHECKING:
    from .core import SpectroscopyCfg, SpectroscopyResult


@dataclass(frozen=True)
class SpectroscopyAnalysis:
    """excitation is (DC,4,frequency,2), ordered F1 then F2.

    acquired_shots is the total of all acquired shots at each
    (DC,branch,frequency), before any herald selection.
    complete is (DC,4): every frequency has all requested reps × rounds.
    Available counts enter P_e even for partial branches; zero counts give NaN.
    Other remains in the denominator. These are classified probabilities, not
    confusion-corrected energy-state populations.
    """

    excitation: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    complete: NDArray[np.bool_]


def analyze_spectra(
    cfg: SpectroscopyCfg | None,
    data: SpectroscopyResult,
    *,
    acquired_shots: NDArray[np.float64],
    plots: Plots,
) -> SpectroscopyAnalysis:
    """Validate saved coordinates and plot F1/F2 P_e from terminal joint counts.

    data.sums holds terminal counts, optionally selected on herald (g,g).
    acquired_shots is (DC,4,frequency), totaled over all 9 ordinary or 81 heralded
    classes before selection. It determines whether every frequency in a branch
    has the requested reps × rounds. Partial counts still enter P_e.
    Use only saved cfg and counts. No live library, IQ classification, peak fit
    or writeback is involved.
    acquired_shots must contain nonnegative totals, with NaN for unmeasured
    points. Wrong shapes or invalid terminal counts raise ValueError; whole
    unmeasured terminal points may be NaN.
    """
    if cfg is None:
        raise ValueError(
            "Spectroscopy analysis requires saved cfg; current Form/library is not a substitute"
        )
    dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
    frequencies = sweep2array(cfg.frequency)
    if (
        not np.array_equal(data.dc_values, dc)
        or data.frequencies.shape != frequencies.shape
        or not np.allclose(
            data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
        )
        or not np.array_equal(data.branches, np.arange(4))
        or data.sums.shape != (len(dc), 4, len(frequencies), 3, 3)
        or acquired_shots.shape != (len(dc), 4, len(frequencies))
    ):
        raise ValueError(
            "Spectroscopy axes/shape must match saved cfg: sums (DC,4,frequency,3,3), "
            "acquired_shots (DC,4,frequency)"
        )
    finite = np.all(np.isfinite(data.sums) & (data.sums >= 0), axis=(-2, -1))
    missing = np.all(np.isnan(data.sums), axis=(-2, -1))
    if not np.all(finite | missing):
        raise ValueError(
            "Spectroscopy contains invalid counts or a partially missing point"
        )
    excitation = np.stack(
        [marginal_probabilities(data.sums, target)[..., 1] for target in ("f1", "f2")],
        axis=-1,
    )
    answer = SpectroscopyAnalysis(
        excitation,
        acquired_shots,
        np.all(acquired_shots >= cfg.reps * cfg.rounds, axis=-1),
    )
    _plot_spectra(cfg, data, answer, plots=plots)
    return answer


def _plot_spectra(
    cfg: SpectroscopyCfg,
    data: SpectroscopyResult,
    answer: SpectroscopyAnalysis,
    *,
    plots: Plots,
) -> None:
    contrast = "on" if cfg.contrast else "off"
    dc_label = (
        "DC inactive" if cfg.dc_device is None else f"{cfg.dc_device} ({cfg.dc_unit})"
    )
    for target, role in enumerate(("F1", "F2")):
        title = f"{role} excitation · double Xπ contrast {contrast}"
        fig, _ = plots.subplots(
            f"spectra_{role.lower()}",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle(title)
        for branch, ax in enumerate(fig.axes):
            for d, dc in enumerate(data.dc_values):
                label = (
                    "DC inactive" if cfg.dc_device is None else f"{dc:g} {cfg.dc_unit}"
                )
                ax.plot(
                    data.frequencies,
                    answer.excitation[d, branch, :, target],
                    label=label,
                )
            ax.set(
                title=f"B{branch:02b} · nominal preparation",
                xlabel="Common probe frequency (MHz)",
                ylabel=f"{role} P_e",
                ylim=(0, 1),
            )
            ax.legend(fontsize="small")
        if cfg.dc_device is None:
            continue
        fig, _ = plots.subplots(
            f"map_{role.lower()}",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle(title)
        for branch, ax in enumerate(fig.axes):
            image = ax.pcolormesh(
                data.frequencies,
                data.dc_values,
                answer.excitation[:, branch, :, target],
                shading="nearest",
                vmin=0,
                vmax=1,
            )
            fig.colorbar(image, ax=ax, label=f"{role} P_e")
            ax.set(
                title=f"B{branch:02b} · nominal preparation",
                xlabel="Common probe frequency (MHz)",
                ylabel=dc_label,
            )
