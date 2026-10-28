"""Plot fixed-flux lifetime drift from workflow-owned scalar pair summaries."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from math import isfinite, nan
from textwrap import fill
from typing import TYPE_CHECKING

from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from zcu_tools.plotting.plots import Plots

if TYPE_CHECKING:
    from zcu_lab.workflows.overnight.workflow import OvernightPoint


def _draw_rejection_notes(figure: Figure, notes: Sequence[str]) -> None:
    """Wrap numbered diagnoses and reserve space below both panels and labels."""
    width = float(figure.get_figwidth())
    font_size = 9
    # Budget 0.7 em per monospace character, including pixel rounding at render.
    # Leave 0.25-inch side margins for the complete wrapped diagnoses.
    columns = max(1, int((width - 0.5) * 72 / (0.7 * font_size)))
    text = "\n".join(
        fill(note, width=columns, subsequent_indent="    ") for note in notes
    )
    line_count = len(text.splitlines())
    notes_height = 0.3 + line_count * font_size / 72 * 1.5 if notes else 0.0
    # Keep the two panels at their original height, including on repeated updates.
    height = 4.8 + notes_height
    figure.set_size_inches(width, height, forward=False)
    figure.set_layout_engine("tight", rect=(0, notes_height / height, 1, 1))
    if notes:
        figure.text(
            0.25 / width,
            0.15 / height,
            text,
            fontsize=font_size,
            fontfamily="DejaVu Sans Mono",
            linespacing=1.5,
            va="bottom",
            in_layout=False,
        )


def draw_overnight_trend(
    plots: Plots, points: Sequence[OvernightPoint], *, start_interval_s: float
) -> None:
    """Show T1 and Echo lifetime drift without recomputing any scientific values.

    plots is the active env.plots, not finished/released. points contains typed
    pair summaries in workflow order with aware UTC actual_start timestamps.
    start_interval_s is the finite positive plan interval in seconds; a single
    pair spans its actual start plus/minus that interval. All ticks show UTC
    HH:MM:SS. Lifetimes stay in microseconds; missing values remain NaN gaps
    with short numbered markers. Complete diagnoses wrap below both panels;
    figure height grows with the footnote lines. No missing values means no
    footnote space. Empty input draws empty axes.
    Updates occur when the workflow calls this function, not after a journal
    commit notification. No file is saved. Invalid time/interval raises
    ValueError; plotting errors propagate without retry.
    """
    if not isfinite(start_interval_s) or start_interval_s <= 0:
        raise ValueError("Overnight trend interval must be finite and positive")
    times: list[datetime] = []
    for point in points:
        actual_start = point.actual_start
        if actual_start is None:
            raise ValueError("Overnight trend requires an actual_start timestamp")
        offset = actual_start.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Overnight actual_start must be aware UTC")
        times.append(actual_start)
    if "overnight_trend" not in plots:
        plots.subplots("overnight_trend", nrows=2, ncols=1, sharex=True)

    def draw(figure: Figure) -> None:
        figure.clear()
        axes = figure.subplots(2, 1, sharex=True, squeeze=False)[:, 0]
        notes: list[str] = []
        for axis, member in zip(axes, ("t1", "t2"), strict=True):
            values = [
                point.t1_us if member == "t1" else point.t2_us for point in points
            ]
            reasons = [
                point.t1_reason if member == "t1" else point.t2_reason
                for point in points
            ]
            axis.plot(
                times, [nan if value is None else value for value in values], "o-"
            )
            label = "T1" if member == "t1" else "T2 Echo"
            axis.set_ylabel(f"{label} (µs)")
            for time, value, reason in zip(times, values, reasons, strict=True):
                if value is None:
                    number = len(notes) + 1
                    diagnosis = reason if reason is not None else "missing"
                    notes.append(
                        f"[{number}] {label}, {time:%H:%M:%S} UTC: {diagnosis}"
                    )
                    axis.text(
                        time,
                        0.02,
                        f"× [{number}]",
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        color="tab:red",
                    )
        # Bound a single pair by its plan interval, not Matplotlib's date fallback.
        if len(points) == 1:
            interval = timedelta(seconds=start_interval_s)
            axes[-1].set_xlim(times[0] - interval, times[0] + interval)
        locator = AutoDateLocator(tz="UTC")
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz="UTC"))
        axes[-1].set_xlabel("Actual start (UTC)")
        _draw_rejection_notes(figure, notes)

    plots.draw("overnight_trend", draw)
