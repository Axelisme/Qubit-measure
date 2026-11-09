"""Decoding, ranking and the score figure of the R-pi readout sweeps.

rpi_power and rpi_freq share what they do with saved rpi_sums records. Each core
keeps its own sweep, program, live plot and result types.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import RPiAnalysis, RPiSums, rpi_analysis
from zcu_tools.experiment import VariableAxisSpec, VariableSpec, VariableZSpec
from zcu_tools.plotting.plots import Plots

# rpi_sums needs one fixed Repeat of at least 16 reads. Its count does not cut the
# chain: lags up to 15 run across reps.
READS_PER_REP = 16

# Columns of one rpi_sums record. Only RPiSums.from_processor reads their meaning.
RECORD_COLUMNS = 35

SCORE_LABEL = "Alternating-contrast score"


@dataclass(frozen=True)
class SweepAxis:
    """How one sweep names and formats its setting in titles, axes and legends.

    setting_name: Word for one setting in titles, for example "gain".
    label: X axis label with its unit, for example "Readout gain (a.u.)".
    unit: Unit printed after the best setting in the legend.
    digits: Significant digits of the best setting in the legend.
    """

    setting_name: str
    label: str
    unit: str
    digits: int


def records_variable(setting: VariableAxisSpec) -> VariableSpec:
    """Map float64 (F, 35) rpi_sums records to the saved "records" variable.

    setting supplies the sweep axis label, disk unit, scale and Result field.
    """
    return VariableSpec(
        "records",
        (
            VariableAxisSpec(
                label="Record column", unit="", values=tuple(range(RECORD_COLUMNS))
            ),
            setting,
        ),
        VariableZSpec("records", label="R-pi sums", unit="a.u."),
    )


def analyze_records(records: NDArray[np.float64]) -> list[RPiAnalysis]:
    """Run the public R-pi analyzer on each row of float64 (F, 35) records."""
    return [rpi_analysis(RPiSums.from_processor(record)) for record in records]


def valid_values(analyses: list[RPiAnalysis], name: str) -> NDArray[np.float64]:
    """Return RPiAnalysis field name per setting, NaN at every invalid setting.

    NaN leaves a gap in a line plot and nanargmax skips it, so invalid settings
    are neither drawn nor ranked.
    """
    return np.array(
        [
            getattr(analysis, name) if analysis.valid else np.nan
            for analysis in analyses
        ],
        dtype=np.float64,
    )


def best_index(analyses: list[RPiAnalysis]) -> int | None:
    """Return the index of the largest valid score, or None when none is valid."""
    scores = valid_values(analyses, "score")
    return None if np.isnan(scores).all() else int(np.nanargmax(scores))


def invalid_reason_counts(analyses: list[RPiAnalysis]) -> dict[str, int]:
    """Count the settings reporting each invalid reason; empty when all are valid."""
    return dict(
        Counter(reason for analysis in analyses for reason in analysis.invalid_reasons)
    )


def not_plotted_title(analyses: list[RPiAnalysis], setting_name: str) -> str:
    """Count the invalid settings left off the curves; empty when there are none.

    setting_name is the word for one setting, for example "gain" or "frequency".
    """
    invalid = sum(not analysis.valid for analysis in analyses)
    if invalid == 0:
        return ""
    if invalid == len(analyses):
        return f"Not plotted: {invalid} invalid; no valid {setting_name} to select"
    return f"Not plotted: {invalid} invalid"


def fix_sweep_xlim(ax: Axes, settings: NDArray[np.float64]) -> None:
    """Fix the x axis to the sweep so rounds with few valid settings do not shrink it.

    A single-point sweep has no range and leaves the axis unchanged.
    """
    margin = 0.05 * float(np.ptp(settings))
    if margin > 0:
        ax.set_xlim(float(settings.min()) - margin, float(settings.max()) + margin)


def draw_score_figure(
    plots: Plots,
    settings: NDArray[np.float64],
    analyses: list[RPiAnalysis],
    best: int | None,
    axis: SweepAxis,
) -> None:
    """Publish the "score" figure of one sweep.

    settings is (F,) in the experiment's unit, in the order of analyses. best is
    the index from best_index. axis names and formats the setting.

    The six panels share the x axis: the score with the best setting marked, the
    invalid reasons per setting, then Q_eff, q, amplitude and noise variance.
    Invalid settings are left off every curve and appear only in the reason
    panel. The figure title counts them.
    """
    fig, _ = plots.subplots(3, 2, num="score", sharex=True, figsize=(11, 8))
    ax_score, ax_reasons, ax_q_eff, ax_q, ax_amplitude, ax_noise = fig.axes

    ax_score.plot(settings, valid_values(analyses, "score"), "o-", color="C0")
    ax_score.set_ylabel(SCORE_LABEL)
    for ax, name, ylabel in (
        (ax_q_eff, "q_eff", "Q_eff (includes π error)"),
        (ax_q, "q", "q = a / σ²"),
        (ax_amplitude, "amplitude", "Amplitude a (ADC units²)"),
        (ax_noise, "noise_variance", "Noise variance σ² (ADC units²)"),
    ):
        ax.plot(settings, valid_values(analyses, name), "o-")
        ax.set_ylabel(ylabel)

    # One row per reason that occurred: a mark at each setting reporting it.
    rows = sorted(invalid_reason_counts(analyses))
    for row, reason in enumerate(rows):
        rejected = [
            setting
            for setting, analysis in zip(settings, analyses, strict=True)
            if reason in analysis.invalid_reasons
        ]
        ax_reasons.plot(rejected, [row] * len(rejected), "x", color="C3")
    ax_reasons.set_yticks(range(len(rows)), rows)
    ax_reasons.set_ylim(-0.5, max(len(rows), 1) - 0.5)
    ax_reasons.set_title(f"Invalid reason per {axis.setting_name}")

    for ax in fig.axes:
        ax.grid(True)
        if best is not None:
            ax.axvline(float(settings[best]), color="r", ls="--")
    if best is not None:
        value = f"{float(settings[best]):.{axis.digits}g}"
        ax_score.legend([ax_score.lines[-1]], [f"largest score = {value} {axis.unit}"])
    ax_amplitude.set_xlabel(axis.label)
    ax_noise.set_xlabel(axis.label)
    title = not_plotted_title(analyses, axis.setting_name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
