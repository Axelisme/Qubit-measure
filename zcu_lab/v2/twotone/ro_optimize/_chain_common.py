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

# y axis of the live and the analysis information plots.
INFORMATION_YLABEL = "Information per read (all lags)"


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
    """Draw the information per read of each fitted setting with its one-sigma error.

    The curve is blue (C0) and labelled "information ± 1σ" for a legend.
    Settings with no signal and failed fits are left out.
    """
    fits = result.fits.values()
    ax.errorbar(
        [fit.setting for fit in fits],
        [fit.metric for fit in fits],
        yerr=[fit.standard_error for fit in fits],
        fmt="o-",
        capsize=3,
        color="C0",
        label="information ± 1σ",
    )


def left_out_title(result: ReadoutMetric) -> str:
    """Count the settings left off the curve, no-signal and failed fits apart.

    Return an empty string when every setting is on the curve.
    """
    if not result.no_signal and not result.failures:
        return ""
    return (
        f"Not plotted: {len(result.no_signal)} no signal, "
        f"{len(result.failures)} fit failed"
    )


def draw_metric(ax: Axes, result: ReadoutMetric) -> tuple[Artist, ...]:
    """Replace the error-bar curve on ax with the information of the fitted settings.

    The title counts the settings left off the curve (see left_out_title).
    """
    for container in list(ax.containers):
        container.remove()
    ax.relim()
    plot_metric(ax, result)
    ax.autoscale_view()
    ax.set_title(left_out_title(result))
    return (ax,)


@dataclass(frozen=True)
class ChainAnalysis:
    """Best setting of a readout-chain sweep. No writeback is proposed.

    best: the fitted setting with the largest information minus its one-sigma
        error, which need not be the highest point of the curve: its setting
        in the experiment's axis unit (gain a.u., frequency MHz or ADC window
        us), information (metric) and QND with their one-sigma errors, d/sigma,
        λ of its two most populated states and the population of its third
        fitted state.
    no_signal: settings the no-signal test stopped before the fit, in the
        axis unit and in axis order. They have no information value.
    failures: settings the estimator could not fit and why, in axis order.
    """

    best: SettingFit
    no_signal: tuple[float, ...]
    failures: tuple[SettingFailure, ...]


def analyze_sums(
    sums: ReadoutSums, plots: Plots, *, label: str, unit: str
) -> ChainAnalysis:
    """Score every setting, plot information and QND, and pick the best.

    label and unit name the swept parameter, for example "Readout gain" and
    "a.u.". Publish the "metric" figure: the information with error bars and
    the best setting marked, and below it the QND of each setting with its
    error bars. The best setting has the largest information minus its
    one-sigma error, so the red line need not be on the highest point. Settings
    with no signal and failed fits are left out of both panels and counted
    apart in the title. Raise ValueError stating both counts when no setting
    is fitted.
    """
    result = readout_metric(sums)
    if not result.fits:
        message = (
            f"No setting has a readout metric: {len(result.no_signal)} no signal, "
            f"{len(result.failures)} fit failed"
        )
        if result.failures:
            reasons = "; ".join(
                f"{failure.setting:g}: {failure.reason}"
                for failure in result.failures.values()
            )
            message += f" ({reasons})"
        raise ValueError(message)
    best = result.fits[result.ranking[0]]
    fits = result.fits.values()

    fig, _ = plots.subplots(2, 1, num="metric", sharex=True, figsize=(8, 6))
    ax_information, ax_qnd = fig.axes
    plot_metric(ax_information, result)
    ax_information.axvline(
        best.setting,
        color="r",
        ls="--",
        label=f"best by information − 1σ = {best.setting:.5g} {unit}",
    )
    ax_information.set_ylabel(INFORMATION_YLABEL)
    ax_information.set_title(left_out_title(result))
    ax_information.legend()
    ax_information.grid(True)

    ax_qnd.errorbar(
        [fit.setting for fit in fits],
        [fit.qnd for fit in fits],
        yerr=[fit.qnd_error for fit in fits],
        fmt="o-",
        capsize=3,
    )
    ax_qnd.axvline(best.setting, color="r", ls="--")
    ax_qnd.set_xlabel(f"{label} ({unit})")
    ax_qnd.set_ylabel("QND")
    ax_qnd.grid(True)
    fig.tight_layout()

    return ChainAnalysis(
        best=best,
        no_signal=tuple(float(sums.axis[index]) for index in result.no_signal),
        failures=tuple(result.failures.values()),
    )
