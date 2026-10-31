"""Saved-source ensemble quadratures, logical phase and conditional phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots

from .._phase import QuadraturePhase, count_phase, wrap_phase
from .._sums import herald_probabilities, sigma, terminal_counts
from .crossing import PhaseCrossings, phase_crossings

if TYPE_CHECKING:
    from .core import ConditionalPhaseCfg, ConditionalPhaseResult


@dataclass(frozen=True)
class ConditionalPhaseAnalysis:
    """Branch/n/frequency classified phase and Xπ-minus-I phase in degrees.

    phases is (2,Nn,Nf); conditional_deg is (Nn,Nf). quadratures is
    (2,2,Nn,Nf) sigma, and acquired_shots is each projection's terminal total.
    crossings retains each n's in-window π intersections; frequency_mhz selects
    only a unique n=1 crossing. herald_matrix is the full herald distribution
    or None for ordinary acquisition; pass_fraction is its (g,g) cell.
    """

    phases: tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]
    conditional_deg: NDArray[np.float64]
    quadratures: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    gate_counts: tuple[int, ...]
    frequencies_mhz: tuple[float, ...]
    crossings: tuple[PhaseCrossings, ...]
    frequency_mhz: float | None
    frequency_reason: str | None
    herald_matrix: NDArray[np.float64] | None = None
    pass_fraction: NDArray[np.float64] | None = None


def analyze_phase(
    cfg: ConditionalPhaseCfg,
    data: ConditionalPhaseResult,
    *,
    plots: Plots,
    heralded: bool = False,
) -> ConditionalPhaseAnalysis:
    """Derive classified X/Y sigma and phase using saved cfg and count axes.

    Each projection uses all acquired counts, including other in its denominator.
    Missing/empty projections and zero contrast retain a reason, not a phase.
    Heralded analysis selects herald (g,g) terminal counts and plots all nine
    herald fractions. No IQ projection, shot reclassification or purity correction
    is applied. Only unique n=1 crossings select common CZ frequency.
    """
    counts = _source_quadratures(cfg, data, heralded=heralded)
    acquired = counts.sum(axis=(-2, -1))
    quadratures = sigma(counts, cfg.target)
    nn, nf = len(cfg.gate_counts), len(data.frequencies)
    branches: list[tuple[tuple[QuadraturePhase, ...], ...]] = []
    conditional = np.full((nn, nf), np.nan)
    for b in range(2):
        trains: list[tuple[QuadraturePhase, ...]] = []
        for n in range(nn):
            points: list[QuadraturePhase] = []
            for f in range(nf):
                point = count_phase(
                    float(quadratures[b, 0, n, f]), float(quadratures[b, 1, n, f])
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
        quadratures,
        acquired,
        tuple(cfg.gate_counts),
        tuple(float(f) for f in data.frequencies),
        crossings,
        selected.frequencies_mhz[0] if selected.reason is None else None,
        selected.reason,
        herald_probabilities(data.sums) if heralded else None,
        herald_probabilities(data.sums)[..., 0, 0] if heralded else None,
    )
    _plot_phase(cfg, data, answer, plots=plots)
    return answer


def _source_quadratures(
    cfg: ConditionalPhaseCfg, data: ConditionalPhaseResult, *, heralded: bool
) -> NDArray[np.float64]:
    """Validate saved scan axes and select terminal counts without averaging."""
    frequencies = sweep2array(cfg.frequency)
    if data.frequencies.shape != frequencies.shape or not np.allclose(
        data.frequencies,
        frequencies,
        rtol=8 * np.finfo(float).eps,
        atol=0,
    ):
        raise ValueError("Conditional phase count axes must match saved cfg")
    for actual, expected in (
        (data.branches, np.arange(2)),
        (data.quadratures, np.arange(2)),
        (data.gate_counts, np.array(cfg.gate_counts)),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError("ConditionalPhase count axes must match saved cfg")
    shape = (2, 2, len(cfg.gate_counts), len(frequencies)) + (
        (3, 3, 3, 3) if heralded else (3, 3)
    )
    if data.sums.shape != shape:
        raise ValueError(f"Expected summed count shape {shape}, got {data.sums.shape}")
    return terminal_counts(data.sums, heralded=heralded)


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
        f"{cfg.target.upper()} target · nominal spectator branches · X/Y independent acquisitions"
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
    contrast_ax.set_ylabel("Classified quadrature contrast")
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
    fig, axes = plots.subplots("quadratures", figsize=(10, 5))
    for b, branch in enumerate(("I", "Xπ")):
        for q, projection in enumerate(("X", "Y")):
            for n, count in enumerate(data.gate_counts):
                axes.plot(
                    data.frequencies,
                    answer.quadratures[b, q, n],
                    ".-",
                    label=f"{branch} {projection} n={count}",
                )
    axes.set(xlabel="Common CZ frequency (MHz)", ylabel="Target sigma")
    axes.legend(fontsize="small", ncols=3)

    if answer.herald_matrix is not None:
        fig, axes = plots.subplots("herald", figsize=(12, 5))
        matrix = answer.herald_matrix.reshape(-1, 9)
        image = axes.imshow(
            np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="Blues", aspect="auto"
        )
        axes.set(
            xticks=range(9),
            xticklabels=[
                f"{a},{b}" for a in ("g", "e", "other") for b in ("g", "e", "other")
            ],
            xlabel="Herald class (F1, F2)",
            ylabel="Scan row (branch, projection, n, frequency)",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        fig.colorbar(image, ax=axes, label="Herald probability")
        fig.tight_layout()
