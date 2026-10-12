"""Plot fixed-flux lifetime drift from workflow-owned scalar pair summaries."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from math import isfinite, nan
from typing import TYPE_CHECKING

from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from zcu_tools.plotting.plots import Plots

if TYPE_CHECKING:
    from zcu_lab.workflows.overnight.workflow import OvernightPoint


def draw_overnight_trend(
    plots: Plots, points: Sequence[OvernightPoint], *, start_interval_s: float
) -> None:
    """Show T1 and Echo lifetime drift without recomputing any scientific values.

    plots is the active env.plots, not finished/released. points contains typed
    pair summaries in workflow order with aware UTC actual_start timestamps.
    start_interval_s is the finite positive plan interval in seconds; a single
    pair spans its actual start plus/minus that interval. All ticks show UTC
    HH:MM:SS. Lifetimes stay in microseconds; missing values remain NaN gaps
    with their diagnoses below the trace. Empty input draws empty axes.
    Updates occur when the workflow calls this function, not after a journal
    commit notification. No file is saved. Invalid time/interval raises
    ValueError; plotting errors propagate without retry.
    """
    if not isfinite(start_interval_s) or start_interval_s <= 0:
        raise ValueError("Overnight trend interval must be finite and positive")
    for point in points:
        offset = point.actual_start.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Overnight actual_start must be aware UTC")
    if "overnight_trend" not in plots:
        plots.subplots("overnight_trend", nrows=2, ncols=1, sharex=True)

    def draw(figure: Figure) -> None:
        figure.clear()
        axes = figure.subplots(2, 1, sharex=True, squeeze=False)[:, 0]
        times = [point.actual_start for point in points]
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
            axis.set_ylabel("T1 (µs)" if member == "t1" else "T2 Echo (µs)")
            for time, value, reason in zip(times, values, reasons, strict=True):
                if value is None:
                    axis.text(
                        time,
                        0.02,
                        reason if reason is not None else "missing",
                        transform=axis.get_xaxis_transform(),
                        rotation=90,
                        va="bottom",
                    )
        # Bound a single pair by its plan interval, not Matplotlib's date fallback.
        if len(points) == 1:
            interval = timedelta(seconds=start_interval_s)
            axes[-1].set_xlim(times[0] - interval, times[0] + interval)
        locator = AutoDateLocator(tz="UTC")
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz="UTC"))
        axes[-1].set_xlabel("Actual start (UTC)")

    plots.draw("overnight_trend", draw)
