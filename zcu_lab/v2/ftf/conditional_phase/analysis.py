"""Saved-source ensemble quadratures, logical phase and conditional phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from .._phase import QuadraturePhase, quadrature_phase, wrap_phase
from .crossing import PhaseCrossings, phase_crossings

if TYPE_CHECKING:
    from .core import ConditionalPhaseCfg, ConditionalPhaseResult


@dataclass(frozen=True)
class ConditionalPhaseAnalysis:
    """Branch/n/frequency phase results and Xπ-minus-I conditional phase (degrees).

    phases has shape (2,Nn,Nf), conditional_deg has (Nn,Nf), absent angles are
    NaN with reasons. mean_iq retains all tones, shape (2,2,Nn,Nf,Ntones).
    acquired_shots retains counts per independent projection, shape (2,2,Nn,Nf).
    gate_counts/frequencies_mhz identify those coordinates. crossings stores every
    n's in-window π intersections; frequency_mhz selects only the unique n=1
    result, otherwise None with frequency_reason. No frequency uncertainty model
    or multi-n consistency gate is implied.
    """

    phases: tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]
    conditional_deg: NDArray[np.float64]
    mean_iq: NDArray[np.complex128]
    acquired_shots: NDArray[np.int64]
    gate_counts: tuple[int, ...]
    frequencies_mhz: tuple[float, ...]
    crossings: tuple[PhaseCrossings, ...]
    frequency_mhz: float | None
    frequency_reason: str | None


def analyze_phase(
    source: RunRecord[ConditionalPhaseCfg, ConditionalPhaseResult], *, plots: Plots
) -> ConditionalPhaseAnalysis:
    """Validate raw against saved cfg and analyze only complete X/Y round pairs.

    Missing quadrature/branch or calibration gives a reason, not an inferred phase.
    Frequency axes allow only SI conversion roundoff; other axes match exactly.
    Any partly missing round or nonfinite acquired value raises ValueError.
    Thermal mixture and weak contrast remain in the result; no shots are removed.
    """
    cfg, data = source.cfg, source.result
    if cfg is None:
        raise ValueError(
            "Phase analysis requires saved source cfg, not current Form/library"
        )
    mean, counts, complete = _source_quadratures(cfg, data)
    nn, nf = len(cfg.gate_counts), len(data.frequencies)
    tone = cfg.roles.f1_tone if cfg.target == "f1" else cfg.roles.f2_tone
    centers = cfg.f1_centers if cfg.target == "f1" else cfg.f2_centers
    branches: list[tuple[tuple[QuadraturePhase, ...], ...]] = []
    conditional = np.full((nn, nf), np.nan)
    for b in range(2):
        trains: list[tuple[QuadraturePhase, ...]] = []
        for n in range(nn):
            points: list[QuadraturePhase] = []
            for f in range(nf):
                if not np.all(complete[b, :, n, f]):
                    point = QuadraturePhase(
                        np.nan, None, "Missing complete X/Y projection rounds"
                    )
                elif centers is None:
                    point = QuadraturePhase(
                        np.nan, None, "Missing saved target GE centers"
                    )
                else:
                    point = quadrature_phase(
                        data.signals[b, 0, n, f, :, tone],
                        data.signals[b, 1, n, f, :, tone],
                        centers,
                    )
                points.append(point)
            trains.append(tuple(points))
        branches.append(tuple(trains))
    for n in range(nn):
        for f in range(nf):
            p0, p1 = (
                branches[0][n][f].logical_phase_deg,
                branches[1][n][f].logical_phase_deg,
            )
            if p0 is not None and p1 is not None:
                conditional[n, f] = wrap_phase(p1 - p0)
    crossings = tuple(phase_crossings(data.frequencies, row) for row in conditional)
    selected = (
        crossings[cfg.gate_counts.index(1)]
        if 1 in cfg.gate_counts
        else PhaseCrossings((), "Missing n=1; higher n cannot select frequency")
    )
    answer = ConditionalPhaseAnalysis(
        tuple(branches),
        conditional,
        mean,
        counts,
        tuple(cfg.gate_counts),
        tuple(float(f) for f in data.frequencies),
        crossings,
        selected.frequencies_mhz[0] if selected.reason is None else None,
        selected.reason,
    )
    _plot_phase(cfg, data, answer, plots=plots)
    return answer


def _source_quadratures(
    cfg: ConditionalPhaseCfg,
    data: ConditionalPhaseResult,
) -> tuple[NDArray[np.complex128], NDArray[np.int64], NDArray[np.bool_]]:
    """Validate saved raw axes/whole-round sentinels and average complete projections."""
    frequencies = sweep2array(cfg.frequency)
    nf, nn, nt, ns = (
        len(frequencies),
        len(cfg.gate_counts),
        len(cfg.modules.readout.tones),
        cfg.rounds * cfg.reps,
    )
    if data.frequencies.shape != frequencies.shape or not np.allclose(
        data.frequencies, frequencies, rtol=8 * np.finfo(float).eps, atol=0
    ):
        raise ValueError("Conditional phase raw axes must match saved cfg")
    for actual, expected in (
        (data.branches, np.arange(2)),
        (data.quadratures, np.arange(2)),
        (data.gate_counts, np.array(cfg.gate_counts)),
        (data.shot_indices, np.arange(ns)),
        (data.tone_indices, np.arange(nt)),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError("Conditional phase raw axes must match saved cfg")
    if data.signals.shape != (2, 2, nn, nf, ns, nt):
        raise ValueError(
            "Expected raw shape (branch,quadrature,n,frequency,rounds*reps,full tones)"
        )
    rounds = data.signals.reshape(2, 2, nn, nf, cfg.rounds, cfg.reps, nt)
    acquired = np.all(np.isfinite(rounds), axis=(-2, -1))
    missing = np.all(
        np.isnan(rounds) & ~np.isinf(rounds.real) & ~np.isinf(rounds.imag),
        axis=(-2, -1),
    )
    if not np.all(acquired | missing):
        raise ValueError("Nonfinite acquired IQ or partially missing round")
    complete = np.all(acquired, axis=-1)
    counts = acquired.sum(axis=-1, dtype=np.int64) * cfg.reps
    mean = np.full((2, 2, nn, nf, nt), complex(np.nan, np.nan), dtype=np.complex128)
    for b, q, n, f in np.argwhere(complete):
        mean[b, q, n, f] = data.signals[b, q, n, f].mean(axis=0)
    return mean, counts, complete


def _plot_phase(
    cfg: ConditionalPhaseCfg,
    data: ConditionalPhaseResult,
    answer: ConditionalPhaseAnalysis,
    *,
    plots: Plots,
) -> None:
    fig, _ = plots.subplots(
        "phase", nrows=3, ncols=1, figsize=(10, 9), layout="constrained"
    )
    fig.suptitle(
        f"{cfg.target.upper()} target · thermal nominal spectator branches · X/Y independent acquisitions"
    )
    contrast_ax, phase_ax, conditional_ax = fig.axes
    for n, count in enumerate(data.gate_counts):
        for b, branch in enumerate(("I", "Xπ")):
            points = answer.phases[b][n]
            contrast_ax.plot(
                data.frequencies,
                [p.contrast for p in points],
                ".-",
                label=f"{branch} n={count}",
            )
            phase_ax.plot(
                data.frequencies,
                [
                    np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                    for p in points
                ],
                ".-",
                label=f"{branch} n={count}",
            )
        conditional_ax.plot(
            data.frequencies, answer.conditional_deg[n], ".-", label=f"n={count}"
        )
        for crossing in answer.crossings[n].frequencies_mhz:
            conditional_ax.plot(crossing, -180, "x", label=f"n={count} π crossing")
    contrast_ax.set_ylabel("GE-normalized contrast")
    phase_ax.set_ylabel("Logical phase (deg, wrapped)")
    conditional_ax.set_ylabel("Xπ − I phase (deg, wrapped)")
    conditional_ax.axhline(-180, color="gray", linestyle="--", label="π modulo 2π")
    for ax in fig.axes:
        ax.set_xlabel("Common CZ frequency (MHz)")
        if len(data.frequencies) > 1:
            ax.set_xlim(data.frequencies[0], data.frequencies[-1])
        ax.legend(fontsize="small", ncols=3)
    conditional_ax.set_title(
        answer.frequency_reason
        or f"n=1 unique crossing: {answer.frequency_mhz:g} MHz; higher n diagnostic only"
    )
    for tone, readout in enumerate(cfg.modules.readout.tones):
        fig, _ = plots.subplots(
            f"quadratures_tone_{tone}",
            nrows=2,
            ncols=2,
            figsize=(11, 7),
            layout="constrained",
        )
        fig.suptitle(
            f"Tone {tone} {readout.label} · mean IQ · thermal nominal branches"
        )
        for b in range(2):
            for q in range(2):
                ax = fig.axes[2 * b + q]
                for n, count in enumerate(data.gate_counts):
                    iq = answer.mean_iq[b, q, n, :, tone]
                    ax.plot(data.frequencies, iq.real, ".-", label=f"I n={count}")
                    ax.plot(data.frequencies, iq.imag, ".--", label=f"Q n={count}")
                ax.set_title(
                    f"Spectator {('I', 'Xπ')[b]} · {('X', 'Y')[q]} projection pulse"
                )
                ax.set_xlabel("Common CZ frequency (MHz)")
                ax.set_ylabel("Mean IQ (ADC units/sample)")
                ax.legend(fontsize="small")
