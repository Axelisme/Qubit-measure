"""Saved-source same-shot return populations and all-tone IQ diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from .fit import CZReturnFit, fit_return

if TYPE_CHECKING:
    from .core import CZAmplitudeCfg, CZAmplitudeResult


@dataclass(frozen=True)
class CZReturnData:
    """Complete-scale classification and mean IQ, with no shot rejection.

    counts/probabilities have shape (scale,4), ordered joint 00/01/10/11.
    mean_iq is (scale,full tones). Incomplete scales retain NaN in all three.
    complete, acquired_shots and analyzed_shots are one-dimensional per scale;
    acquired_shots includes partial rounds but analyzed_shots is zero unless all
    requested rounds at that scale exist. No radius cut or inversion is used.
    """

    counts: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    mean_iq: NDArray[np.complex128]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.int64]
    analyzed_shots: NDArray[np.int64]


def classify_return(
    source: RunRecord[CZAmplitudeCfg, CZAmplitudeResult],
) -> CZReturnData:
    """Validate source cfg/axes/raw and classify all shots of each complete scale.

    Missing cfg, mismatched axes/shape, nonfinite acquired IQ or partly missing
    rounds raise ValueError. Only whole NaN rounds count as unwritten slots.
    F1/F2 use saved original tone positions and nearest saved GE centers (ties
    to G); joint outcomes come from the same shot, never marginal products.
    This pure function neither reads current Form/library nor mutates source.
    """
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "CZ return analysis requires saved cfg and GE centers; current Form/library is not a substitute"
        )
    scales = sweep2array(cfg.scales)
    ntones, nshots = len(cfg.modules.readout.tones), cfg.rounds * cfg.reps
    if (
        not np.array_equal(data.scales, scales)
        or not np.array_equal(data.shot_indices, np.arange(nshots))
        or not np.array_equal(data.tone_indices, np.arange(ntones))
        or data.signals.shape != (len(scales), nshots, ntones)
    ):
        raise ValueError(
            "CZ return raw axes/shape must match saved cfg: (scale,rounds*reps,full tones)"
        )
    rounds = data.signals.reshape(len(scales), cfg.rounds, cfg.reps, ntones)
    acquired = np.all(np.isfinite(rounds), axis=(-2, -1))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag),
        axis=(-2, -1),
    )
    if not np.all(acquired | missing):
        raise ValueError(
            "CZ return raw contains nonfinite acquired IQ or a partially missing round"
        )
    complete = np.all(acquired, axis=-1)
    acquired_shots = acquired.sum(axis=-1, dtype=np.int64) * cfg.reps
    counts = np.full((len(scales), 4), np.nan, dtype=np.float64)
    mean_iq = np.full(
        (len(scales), ntones), complex(np.nan, np.nan), dtype=np.complex128
    )
    for index in np.flatnonzero(complete):
        shots = data.signals[index]
        f1, f2 = shots[:, cfg.roles.f1_tone], shots[:, cfg.roles.f2_tone]
        e1 = np.abs(f1 - cfg.f1_centers.e) < np.abs(f1 - cfg.f1_centers.g)
        e2 = np.abs(f2 - cfg.f2_centers.e) < np.abs(f2 - cfg.f2_centers.g)
        labels = 2 * e1.astype(np.int64) + e2.astype(np.int64)
        counts[index] = np.bincount(labels, minlength=4)
        mean_iq[index] = shots.mean(axis=0)
    return CZReturnData(
        counts,
        counts / nshots,
        mean_iq,
        complete,
        acquired_shots,
        np.where(complete, acquired_shots, 0),
    )


@dataclass(frozen=True)
class CZAmplitudeAnalysis:
    """Saved-source joint/IQ diagnostics and the zero-phase first-cycle fit."""

    data: CZReturnData
    fit: CZReturnFit


def analyze_return(
    source: RunRecord[CZAmplitudeCfg, CZAmplitudeResult], *, plots: Plots
) -> CZAmplitudeAnalysis:
    """Classify complete scales and fit their contiguous measured prefix only.

    Missing saved cfg or invalid raw raises ValueError. Numeric/coverage failures
    retain diagnostic data, curve and an explicit fit.reason with no fit.scale.
    Register joint_return with fit diagnostics and all original-tone I/Q figures.
    No current Form/library, hardware, saving or writeback is used.
    """
    data = classify_return(source)
    fit = fit_return(source.result.scales, data.probabilities[:, 0], data.complete)
    _plot_return(source, data, fit, plots=plots)
    return CZAmplitudeAnalysis(data, fit)


def _plot_return(
    source: RunRecord[CZAmplitudeCfg, CZAmplitudeResult],
    answer: CZReturnData,
    fit: CZReturnFit,
    *,
    plots: Plots,
) -> None:
    """Register joint_return and one iq_tone_i I/Q diagnostic per original tone.

    source must have cfg and answer must be its classify_return result. Plot only
    complete scales, leaving missing scale gaps visible. Figures describe thermal
    nominal preparation and a leakage proxy, never physical gate fidelity.
    """
    cfg = source.cfg
    if cfg is None:
        raise ValueError("CZ return plots require saved cfg")
    scales = source.result.scales
    _, ax = plots.subplots("joint_return", figsize=(9, 5), layout="constrained")
    for index, label in enumerate(("00 return", "01", "10", "11")):
        ax.plot(scales, answer.probabilities[:, index], ".-", label=label)
    ax.set(
        xlabel="Common gain scale",
        ylabel="Same-shot joint probability",
        title="B11 → CZ → Xπ⊗Xπ · thermal nominal preparation · leakage proxy",
    )
    if np.any(np.isfinite(fit.curve)):
        ax.plot(scales, fit.curve, "k--", label="Zero-phase P00 fit")
    diagnostic = (
        fit.reason or f"First positive return scale = {fit.scale:.6g} ± {fit.error:.2g}"
    )
    ax.text(
        0.02,
        0.02,
        diagnostic,
        transform=ax.transAxes,
        fontsize="small",
        bbox={"facecolor": "white", "alpha": 0.85},
    )
    ax.legend()
    for tone, readout in enumerate(cfg.modules.readout.tones):
        _, ax = plots.subplots(f"iq_tone_{tone}", figsize=(9, 5), layout="constrained")
        iq = answer.mean_iq[:, tone]
        ax.plot(scales, iq.real, ".-", label="I")
        ax.plot(scales, iq.imag, ".-", label="Q")
        ax.set(
            xlabel="Common gain scale",
            ylabel="Mean IQ (ADC units/sample)",
            title=f"Tone {tone} {readout.label} · thermal nominal B11 · complete scales only",
        )
        ax.legend()
