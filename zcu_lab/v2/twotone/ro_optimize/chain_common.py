"""Shared parts of the three readout-chain ro_optimize experiments.

chain_power, chain_freq and chain_length each sweep one readout parameter.
Their shot is one block of READS_PER_BLOCK consecutive reads of one
PulseReadout, with no reset, qubit pulse or wait. Blocks are reduced to 113
moment sums by chain_readout_sums and scored by readout_chain.readout_metric.

This module holds what the three share: the read block, the saved record
layout, the metric of each setting and the analysis. Each core keeps its own
sweep, program and live plot in its run method.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import ReadoutSums, readout_metric
from zcu_tools.experiment import (
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import PulseReadout, TimelineBuilder

# The 113 block moments and readout_metric are defined for 32-read blocks.
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
    """Saved layout of ReadoutSums: block counts and the 113 sums per setting.

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
                VariableAxisSpec(label="Moment", unit="", values=tuple(range(113))),
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

    axis is (F,) in the experiment's unit. sums is the (F, 114) summed buffer
    of chain_readout_sums; a setting without a finished round is a NaN row.
    Raise RuntimeError when no setting has finished a round, for example
    after a stop during the first round.
    """
    finished = ~np.isnan(sums[:, 0])
    if not finished.any():
        raise RuntimeError(
            "Stopped before any setting finished a round; no sums to keep"
        )
    return ReadoutSums.from_processor(axis[finished], sums[finished])


@dataclass(frozen=True)
class SettingFailure:
    """One setting whose readout metric could not be computed.

    setting: its coordinate in the experiment's axis unit.
    reason: the readout_metric error, for example no identifiable signal.
    """

    setting: float
    reason: str


@dataclass(frozen=True)
class SettingMetrics:
    """readout_metric results of each setting; NaN where it failed.

    axis: (F,) settings in the experiment's unit.
    metric, standard_error: (F,) read-drive-read information and its error.
    separation: (F,) d/sigma. retention: (F, 2) λ of the two most populated
    states. third_population: (F,) population of the third fitted state.
    failures: one entry per NaN setting, in axis order.
    """

    axis: NDArray[np.float64]
    metric: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    separation: NDArray[np.float64]
    retention: NDArray[np.float64]
    third_population: NDArray[np.float64]
    failures: tuple[SettingFailure, ...]


def metric_per_setting(sums: ReadoutSums) -> SettingMetrics:
    """Run readout_metric on each setting alone, so a failure stays local.

    readout_metric raises ValueError for the whole input when one setting has
    no identifiable signal or a non-finite fit. Evaluating one setting at a
    time keeps the others. The cost is about 10 ms per setting; the first
    call in a process can take about 6 s while Numba compiles.
    """
    count = len(sums.axis)
    metric = np.full(count, np.nan)
    standard_error = np.full(count, np.nan)
    separation = np.full(count, np.nan)
    retention = np.full((count, 2), np.nan)
    third_population = np.full(count, np.nan)
    failures: list[SettingFailure] = []
    for i in range(count):
        one_setting = ReadoutSums(
            sums.axis[i : i + 1],
            sums.block_count[i : i + 1],
            sums.feature_sums[i : i + 1],
        )
        try:
            result = readout_metric(one_setting)
        except ValueError as error:
            # Isolate: one setting without readout signal must not fail the run.
            failures.append(SettingFailure(float(sums.axis[i]), str(error)))
            continue
        metric[i] = result.metric[0]
        standard_error[i] = result.standard_error[0]
        separation[i] = result.separation[0]
        retention[i] = result.retention[0]
        third_population[i] = result.third_population[0]
    return SettingMetrics(
        axis=sums.axis,
        metric=metric,
        standard_error=standard_error,
        separation=separation,
        retention=retention,
        third_population=third_population,
        failures=tuple(failures),
    )


def draw_metric(ax: Axes, values: SettingMetrics) -> tuple[Artist, ...]:
    """Replace the error-bar curve on ax with the settings that have a metric.

    The title counts the settings left out because their metric failed.
    """
    for container in list(ax.containers):
        container.remove()
    ax.relim()
    shown = ~np.isnan(values.metric)
    ax.errorbar(
        values.axis[shown],
        values.metric[shown],
        yerr=values.standard_error[shown],
        fmt="o-",
        capsize=3,
        color="C0",
    )
    ax.autoscale_view()
    failed = len(values.failures)
    ax.set_title(f"Settings without a metric: {failed}" if failed else "")
    return (ax,)


@dataclass(frozen=True)
class ChainAnalysis:
    """Best setting of a readout-chain sweep. No writeback is proposed.

    best_setting: the setting with the largest metric, in the experiment's
        axis unit (gain a.u., frequency MHz or ADC window us).
    metric, metric_err: its read-drive-read information and one-sigma error.
    separation: its d/sigma, centre distance over noise width.
    retention: λ of its two most populated states, most populated first.
    third_population: stationary population of its third fitted state.
    failures: settings without a metric and why, in axis order.
    """

    best_setting: float
    metric: float
    metric_err: float
    separation: float
    retention: tuple[float, float]
    third_population: float
    failures: tuple[SettingFailure, ...]


def analyze_sums(
    sums: ReadoutSums, plots: Plots, *, label: str, unit: str
) -> ChainAnalysis:
    """Score every setting, plot the metric and d/sigma, and pick the best.

    label and unit name the swept parameter, for example "Readout gain" and
    "a.u.". Publish the "metric" figure: the metric with error bars and the
    best setting marked, and below it d/sigma of each setting. Settings
    without a metric are left out of both panels and listed in failures.
    Raise ValueError when no setting has a metric.
    """
    values = metric_per_setting(sums)
    if np.isnan(values.metric).all():
        reasons = "; ".join(f"{f.setting:g}: {f.reason}" for f in values.failures)
        raise ValueError(f"No setting has a readout metric ({reasons})")
    best = int(np.nanargmax(values.metric))
    best_setting = float(values.axis[best])
    shown = ~np.isnan(values.metric)

    fig, _ = plots.subplots(2, 1, num="metric", sharex=True, figsize=(8, 6))
    ax_metric, ax_separation = fig.axes
    ax_metric.errorbar(
        values.axis[shown],
        values.metric[shown],
        yerr=values.standard_error[shown],
        fmt="o-",
        capsize=3,
        label="metric ± 1σ",
    )
    ax_metric.axvline(
        best_setting, color="r", ls="--", label=f"best = {best_setting:.5g} {unit}"
    )
    ax_metric.set_ylabel("Read-drive-read information")
    if values.failures:
        ax_metric.set_title(f"Settings without a metric: {len(values.failures)}")
    ax_metric.legend()
    ax_metric.grid(True)

    ax_separation.plot(values.axis[shown], values.separation[shown], "o-")
    ax_separation.axvline(best_setting, color="r", ls="--")
    ax_separation.set_xlabel(f"{label} ({unit})")
    ax_separation.set_ylabel("d/σ")
    ax_separation.grid(True)
    fig.tight_layout()

    return ChainAnalysis(
        best_setting=best_setting,
        metric=float(values.metric[best]),
        metric_err=float(values.standard_error[best]),
        separation=float(values.separation[best]),
        retention=(float(values.retention[best, 0]), float(values.retention[best, 1])),
        third_population=float(values.third_population[best]),
        failures=values.failures,
    )
