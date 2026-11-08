"""Shared parts of the three readout-chain ro_optimize experiments.

chain_power, chain_freq and chain_length each sweep one readout parameter.
Their shot is one block of READS_PER_BLOCK consecutive reads of one
PulseReadout, with no reset, qubit pulse or wait. Blocks are reduced to
moment sums by chain_readout_sums and scored by readout_chain.readout_metric.

This module holds what the three share: the read block, the saved record
layout, the metric plot and the analysis. Each core keeps its own sweep,
program and live plot in its run method.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import (
    MOMENT_COUNT,
    ReadoutMetric,
    ReadoutSums,
    SettingFailure,
    SettingFit,
    readout_metric,
)
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, TimelineBuilder

# The block moments and readout_metric are defined for 32-read blocks.
READS_PER_BLOCK = 32


def emit_read_block(builder: TimelineBuilder, readout: PulseReadout) -> None:
    """Record one shot: READS_PER_BLOCK reads of readout and nothing else.

    Each read starts as soon as the previous read's pulse and ADC window have
    ended (delay_auto). There is no reset, drive pulse or extra wait inside
    the block. After build, call prog.set_reads_per_shot(READS_PER_BLOCK).
    """
    with builder.repeat("read", READS_PER_BLOCK):
        builder.emit(readout)
        builder.delay_auto()


def readout_sums_variables(setting: VariableAxisSpec) -> tuple[VariableSpec, ...]:
    """Saved layout of ReadoutSums: block counts and the moment sums per setting.

    setting is the experiment's sweep axis, mapped to the ReadoutSums field
    "axis" with the experiment's label, unit and disk scale.
    """
    return (
        VariableSpec(
            "block_count",
            (setting,),
            VariableZSpec("block_count", label="Blocks", unit=""),
        ),
        VariableSpec(
            "feature_sums",
            (
                VariableAxisSpec(
                    label="Moment", unit="", values=tuple(range(MOMENT_COUNT))
                ),
                setting,
            ),
            VariableZSpec("feature_sums", label="Moment sum", unit="a.u."),
        ),
    )


def restore_readout_sums(data: GroupedLoadData[Any]) -> ReadoutSums:
    """Rebuild the saved ReadoutSums from its two variables."""
    block_count = data.variable("block_count")
    return ReadoutSums(
        axis=block_count.axes[0],
        block_count=block_count.z,
        feature_sums=data.variable("feature_sums").z,
    )


def finished_sums(axis: NDArray[np.float64], sums: NDArray[np.float64]) -> ReadoutSums:
    """ReadoutSums of the settings that finished at least one round.

    axis is (F,) in the experiment's unit. sums is the summed buffer of
    chain_readout_sums, one row per setting: the block count, then the moment
    sums. A setting without a finished round is a NaN row.
    Raise RuntimeError when no setting has finished a round, for example
    after a stop during the first round.
    """
    finished = ~np.isnan(sums[:, 0])
    if not finished.any():
        raise RuntimeError(
            "Stopped before any setting finished a round; no sums to keep"
        )
    return ReadoutSums.from_processor(axis[finished], sums[finished])


def plot_metric(ax: Axes, result: ReadoutMetric) -> None:
    """Draw the metric of each fitted setting with its one-sigma error bar.

    The curve is blue (C0) and labelled "metric ± 1σ" for a legend. Settings
    that could not be fitted are left out.
    """
    fits = result.fits.values()
    ax.errorbar(
        [fit.setting for fit in fits],
        [fit.metric for fit in fits],
        yerr=[fit.standard_error for fit in fits],
        fmt="o-",
        capsize=3,
        color="C0",
        label="metric ± 1σ",
    )


def draw_metric(ax: Axes, result: ReadoutMetric) -> tuple[Artist, ...]:
    """Replace the error-bar curve on ax with the metric of the fitted settings.

    The title counts the settings left out because they could not be fitted.
    """
    for container in list(ax.containers):
        container.remove()
    ax.relim()
    plot_metric(ax, result)
    ax.autoscale_view()
    failed = len(result.failures)
    ax.set_title(f"Settings without a metric: {failed}" if failed else "")
    return (ax,)


@dataclass(frozen=True)
class ChainAnalysis:
    """Best setting of a readout-chain sweep. No writeback is proposed.

    best: the fitted setting with the largest metric: its setting in the
        experiment's axis unit (gain a.u., frequency MHz or ADC window us),
        metric with one-sigma error, d/sigma, λ of its two most populated
        states and the population of its third fitted state.
    failures: settings that could not be fitted and why, in axis order.
    """

    best: SettingFit
    failures: tuple[SettingFailure, ...]


def analyze_sums(
    sums: ReadoutSums, plots: Plots, *, label: str, unit: str
) -> ChainAnalysis:
    """Score every setting, plot the metric and d/sigma, and pick the best.

    label and unit name the swept parameter, for example "Readout gain" and
    "a.u.". Publish the "metric" figure: the metric with error bars and the
    best setting marked, and below it d/sigma of each setting. Settings that
    could not be fitted are left out of both panels and listed in failures.
    Raise ValueError when no setting could be fitted.
    """
    result = readout_metric(sums)
    if not result.fits:
        reasons = "; ".join(
            f"{failure.setting:g}: {failure.reason}"
            for failure in result.failures.values()
        )
        raise ValueError(f"No setting has a readout metric ({reasons})")
    best = result.fits[result.ranking[0]]
    fits = result.fits.values()

    fig, _ = plots.subplots(2, 1, num="metric", sharex=True, figsize=(8, 6))
    ax_metric, ax_separation = fig.axes
    plot_metric(ax_metric, result)
    ax_metric.axvline(
        best.setting, color="r", ls="--", label=f"best = {best.setting:.5g} {unit}"
    )
    ax_metric.set_ylabel("Read-drive-read information")
    if result.failures:
        ax_metric.set_title(f"Settings without a metric: {len(result.failures)}")
    ax_metric.legend()
    ax_metric.grid(True)

    ax_separation.plot(
        [fit.setting for fit in fits], [fit.separation for fit in fits], "o-"
    )
    ax_separation.axvline(best.setting, color="r", ls="--")
    ax_separation.set_xlabel(f"{label} ({unit})")
    ax_separation.set_ylabel("d/σ")
    ax_separation.grid(True)
    fig.tight_layout()

    return ChainAnalysis(best=best, failures=tuple(result.failures.values()))
