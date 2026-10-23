"""Saved-source diagnostic IQ spectra; no level assignment or peak calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

if TYPE_CHECKING:
    from .core import SpectroscopyCfg, SpectroscopyResult


@dataclass(frozen=True)
class SpectroscopyAnalysis:
    """Diagnostic means and completion, retaining full-tone order.

    mean_iq has shape (D,4,F,N); incomplete DC/branches remain complex NaN.
    complete, acquired_shots and analyzed_shots have shape (D,4). A complete
    branch uses all frequencies, rounds and reps without discarding any shot.
    Counts include all frequencies; analyzed_shots is zero for incomplete branches.
    """

    mean_iq: NDArray[np.complex128]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.int64]
    analyzed_shots: NDArray[np.int64]


def analyze_spectra(
    source: RunRecord[SpectroscopyCfg, SpectroscopyResult], *, plots: Plots
) -> SpectroscopyAnalysis:
    """Validate saved cfg/raw, average only complete DC/branches and draw spectra.

    Require axes and shape to match saved cfg, allowing only SI conversion
    roundoff in frequency coordinates. Missing slots must be whole NaN rounds;
    any other nonfinite acquired IQ raises ValueError. This does not
    consult current Form/library, select shots, fit peaks or propose calibration.
    Register four-branch I/Q curves per original tone and, for DC scans, separate
    four-branch I/Q heatmaps. No GE centers or level identity assumptions are used.
    """
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Spectroscopy analysis requires saved cfg; current Form/library is not a substitute"
        )
    dc = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
    frequencies = sweep2array(cfg.frequency)
    ntones, nshots = len(cfg.modules.readout.tones), cfg.rounds * cfg.reps
    if (
        not np.array_equal(data.dc_values, dc)
        or data.frequencies.shape != frequencies.shape
        or not np.allclose(
            data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
        )
        or not np.array_equal(data.branches, np.arange(4))
        or not np.array_equal(data.tone_indices, np.arange(ntones))
        or not np.array_equal(data.shot_indices, np.arange(nshots))
        or data.signals.shape != (len(dc), 4, len(frequencies), nshots, ntones)
    ):
        raise ValueError(
            "Spectroscopy raw axes/shape must match saved cfg: (DC,4,frequency,rounds*reps,full tones)"
        )
    rounds = data.signals.reshape(
        len(dc), 4, len(frequencies), cfg.rounds, cfg.reps, ntones
    )
    acquired = np.all(np.isfinite(rounds), axis=(-2, -1))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag),
        axis=(-2, -1),
    )
    if not np.all(acquired | missing):
        raise ValueError(
            "Spectroscopy raw contains nonfinite acquired IQ or a partially missing round"
        )
    complete = np.all(acquired, axis=(-2, -1))
    counts = acquired.sum(axis=(-2, -1), dtype=np.int64) * cfg.reps
    mean = np.full(
        (len(dc), 4, len(frequencies), ntones),
        complex(np.nan, np.nan),
        dtype=np.complex128,
    )
    for d, b in np.argwhere(complete):
        mean[d, b] = data.signals[d, b].mean(axis=1)
    answer = SpectroscopyAnalysis(mean, complete, counts, np.where(complete, counts, 0))
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
        "DC not swept" if cfg.dc_device is None else f"{cfg.dc_device} ({cfg.dc_unit})"
    )
    for tone, readout in enumerate(cfg.modules.readout.tones):
        roles = ", ".join(
            role
            for role, index in (("F1", cfg.roles.f1_tone), ("F2", cfg.roles.f2_tone))
            if tone == index
        )
        title = f"Tone {tone} {readout.label} {roles} · contrast {contrast} · thermal nominal branches"
        fig, _ = plots.subplots(
            f"spectra_tone_{tone}",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle(title)
        for branch, ax in enumerate(fig.axes):
            for d, dc in enumerate(data.dc_values):
                if answer.complete[d, branch]:
                    label = (
                        "DC inactive"
                        if cfg.dc_device is None
                        else f"{dc:g} {cfg.dc_unit}"
                    )
                    iq = answer.mean_iq[d, branch, :, tone]
                    ax.plot(data.frequencies, iq.real, label=f"I · {label}")
                    ax.plot(data.frequencies, iq.imag, "--", label=f"Q · {label}")
            ax.set_title(f"B{branch:02b} · nominal preparation")
            ax.set_xlabel("Common probe frequency (MHz)")
            ax.set_ylabel("Mean IQ (ADC units/sample)")
            if ax.lines:
                ax.legend(fontsize="small")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Incomplete branch: no spectrum",
                    ha="center",
                    transform=ax.transAxes,
                )
        if cfg.dc_device is None:
            continue
        for quadrature, values in (
            ("I", answer.mean_iq.real),
            ("Q", answer.mean_iq.imag),
        ):
            fig, _ = plots.subplots(
                f"map_{quadrature}_tone_{tone}",
                nrows=2,
                ncols=2,
                figsize=(11, 7),
                squeeze=False,
                layout="constrained",
            )
            fig.suptitle(f"{title} · {quadrature}")
            for branch, ax in enumerate(fig.axes):
                image = ax.pcolormesh(
                    data.frequencies,
                    data.dc_values,
                    values[:, branch, :, tone],
                    shading="nearest",
                )
                fig.colorbar(
                    image, ax=ax, label=f"Mean {quadrature} (ADC units/sample)"
                )
                ax.set_title(f"B{branch:02b} · nominal preparation")
                ax.set_xlabel("Common probe frequency (MHz)")
                ax.set_ylabel(dc_label)
