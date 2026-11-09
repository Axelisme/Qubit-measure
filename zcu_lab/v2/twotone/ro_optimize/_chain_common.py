"""Shared shot, saved sums and plots for thermal-chain ro_optimize sweeps.

Each core keeps its sweep and acquisition in run. This module supplies the
16-read shot, the count + 107 sums layout and the same three-panel display
for live updates and offline analysis.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import (
    THERMAL_MOMENT_COUNT,
    SettingScore,
    ThermalChainScore,
    ThermalChainSums,
    thermal_chain_score,
)
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, TimelineBuilder

READS_PER_SHOT = 16


def emit_read_shot(builder: TimelineBuilder, readout: PulseReadout) -> None:
    """Emit 16 consecutive reads with no drive or extra wait within the shot.

    Each read starts when the previous pulse and ADC window have ended.
    The builder's relax_delay supplies K after the shot. After build, declare
    prog.set_reads_per_shot(READS_PER_SHOT) before constructing the processor.
    """
    with builder.repeat("read", READS_PER_SHOT):
        builder.emit(readout)
        builder.delay_auto()


def readout_sums_variables(setting: VariableAxisSpec) -> tuple[VariableSpec, ...]:
    """Map the sweep axis and all ThermalChainSums fields to saved variables.

    setting supplies the label, disk unit and scale of the result's axis.
    All 107 sums remain available, including every adjacent pair component.
    """
    return (
        VariableSpec(
            "shot_count",
            (setting,),
            VariableZSpec("shot_count", label="Shots", unit=""),
        ),
        VariableSpec(
            "feature_sums",
            (
                VariableAxisSpec(
                    label="Moment", unit="", values=tuple(range(THERMAL_MOMENT_COUNT))
                ),
                setting,
            ),
            VariableZSpec("feature_sums", label="Moment sum", unit="a.u."),
        ),
    )


def restore_readout_sums(data: GroupedLoadData[Any]) -> ThermalChainSums:
    """Restore shot counts, axis and 107 sums from their saved variables."""
    shot_count = data.variable("shot_count")
    return ThermalChainSums(
        axis=shot_count.axes[0],
        shot_count=shot_count.z,
        feature_sums=data.variable("feature_sums").z,
    )


def finished_sums(
    axis: NDArray[np.float64], sums: NDArray[np.float64]
) -> ThermalChainSums:
    """Keep settings with a completed round from a (settings,108) buffer.

    axis uses the experiment's units. Unfinished settings are NaN rows.
    Raise RuntimeError if no setting has completed a round.
    """
    finished = ~np.isnan(sums[:, 0])
    if not finished.any():
        raise RuntimeError(
            "Stopped before any setting finished a round; no sums to keep"
        )
    return ThermalChainSums.from_processor(axis[finished], sums[finished])


def draw_metric(
    axes: tuple[Axes, ...], result: ThermalChainScore, *, label: str, unit: str
) -> tuple[Artist, ...]:
    """Replace three panels with score, QND and correlation-growth diagnostics.

    axes are the three vertically stacked axes of one figure. label and unit
    describe the swept parameter. Score and QND show model-conditional
    approximate error scales, not general one-sigma errors. Score limits use
    only score values; QND limits stay at 0..1 without clipping reported data.
    M and |rho| share the diagnostic panel. Red lines mark the largest score,
    with exact ties resolved by acquisition order, in live and analysis plots.
    """
    ax_score, ax_qnd, ax_m = axes
    for ax in axes:
        ax.clear()
    settings = result.settings
    xs = [point.setting for point in settings]
    scores = [point.score for point in settings]
    best = settings[result.ranking[0]]
    ax_score.errorbar(
        xs,
        scores,
        yerr=[point.score_error for point in settings],
        fmt="o-",
        capsize=3,
        color="C0",
        label="Score ± model approx. scale",
    )
    low, high = min(scores), max(scores)
    padding = 0.1 * (high - low) if high > low else 0.1 * max(abs(high), 0.01)
    ax_score.set_ylim(low - padding, high + padding)
    ax_score.set_ylabel("Heuristic score")
    ax_score.set_title("Errors: model-conditional approximate scales")
    ax_score.axvline(
        best.setting,
        color="r",
        ls="--",
        label=f"Max score: {best.setting:.5g} {unit}",
    )
    ax_qnd.errorbar(
        xs,
        [point.qnd for point in settings],
        yerr=[point.qnd_error for point in settings],
        fmt="o-",
        capsize=3,
        color="C0",
        label="QND ± model approx. scale",
    )
    ax_qnd.set(ylabel="QND", ylim=(0, 1))
    ax_qnd.axvline(best.setting, color="r", ls="--")
    ax_m.plot(
        xs, [point.mist for point in settings], "o-", label="M: correlation growth"
    )
    ax_m.plot(xs, [abs(point.correlation) for point in settings], "s--", label="|ρ|")
    ax_m.set(xlabel=f"{label} ({unit})", ylabel="M / |ρ|")
    ax_m.axvline(best.setting, color="r", ls="--")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(loc="best")
        ax.grid(True)
    return tuple(axes)


@dataclass(frozen=True)
class ChainAnalysis:
    """Best setting by descending thermal-chain score; no writeback proposed.

    best: SettingScore in the sweep's units (gain a.u., frequency MHz or
        ADC window us). Errors are model-conditional approximate scales.
        M diagnoses correlation growth, not identifiable physical leakage.
        QND is reported without clipping. Exact score ties keep acquisition order.
    """

    best: SettingScore


def analyze_sums(
    sums: ThermalChainSums, plots: Plots, *, label: str, unit: str
) -> ChainAnalysis:
    """Score saved moments and publish score, QND and M / |rho| panels.

    label and unit describe the sweep. The metric figure marks the largest
    score in red; errors do not affect ranking. Invalid moments propagate
    ValueError from thermal_chain_score. No fit or secondary threshold is used.
    """
    result = thermal_chain_score(sums)
    fig, _ = plots.subplots(3, 1, num="metric", sharex=True, figsize=(8, 8))
    draw_metric(tuple(fig.axes), result, label=label, unit=unit)
    fig.tight_layout()
    return ChainAnalysis(best=result.settings[result.ranking[0]])
