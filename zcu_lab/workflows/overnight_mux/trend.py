"""Draw one scalar lifetime trace per mux participant without refitting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from math import isfinite, nan
from typing import TYPE_CHECKING

from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from zcu_tools.plotting.plots import Plots

if TYPE_CHECKING:
    from .workflow import OvernightMuxPoint


def draw_overnight_mux_trend(
    plots: Plots, points: Sequence[OvernightMuxPoint], *, start_interval_s: float
) -> None:
    """Draw T1/Echo panels with named qubit lines and gaps at missing values.

    plots is active env.plots. points is workflow-ordered scalar history, each
    with aware UTC actual_start and the same ordered participants. A single
    pair spans +/- start_interval_s seconds. UTC ticks show HH:MM:SS. Updates
    precede journal commit, do not fit data and do not save a file. Invalid
    interval/timestamps raise ValueError; plotting errors propagate.
    """
    if not isfinite(start_interval_s) or start_interval_s <= 0:
        raise ValueError("Overnight mux trend interval must be finite and positive")
    times: list[datetime] = []
    for point in points:
        actual = point.actual_start
        if actual is None or actual.utcoffset() != timedelta(0):
            raise ValueError("Overnight mux trend requires aware UTC actual_start")
        times.append(actual)
    if "overnight_mux_trend" not in plots:
        plots.subplots("overnight_mux_trend", nrows=2, ncols=1, sharex=True)

    def draw(figure: Figure) -> None:
        figure.clear()
        axes = figure.subplots(2, 1, sharex=True, squeeze=False)[:, 0]
        for axis, member in zip(axes, ("t1", "t2"), strict=True):
            for index, first in enumerate(points[0].qubits if points else ()):
                qubits = [point.qubits[index] for point in points]
                values = [q.t1_us if member == "t1" else q.t2_us for q in qubits]
                reasons = [
                    q.t1_reason if member == "t1" else q.t2_reason for q in qubits
                ]
                (line,) = axis.plot(
                    times,
                    [nan if v is None else v for v in values],
                    "o-",
                    label=first.name,
                )
                for time, value, reason in zip(times, values, reasons, strict=True):
                    if value is None:
                        axis.text(
                            time,
                            0.02,
                            f"{first.name}: {reason or 'missing'}",
                            color=line.get_color(),
                            transform=axis.get_xaxis_transform(),
                            rotation=90,
                            va="bottom",
                        )
            axis.set_ylabel("T1 (µs)" if member == "t1" else "T2 Echo (µs)")
            if points:
                axis.legend()
        if times:
            # All-rejected traces contain only NaN, so Matplotlib otherwise
            # discards their dates and falls back to the epoch instead.
            interval = (
                timedelta(seconds=start_interval_s)
                if len(points) == 1
                else (max(times) - min(times)) * 0.05
            )
            axes[-1].set_xlim(min(times) - interval, max(times) + interval)
        axes[-1].xaxis.set_major_locator(AutoDateLocator(tz="UTC"))
        axes[-1].xaxis.set_major_formatter(DateFormatter("%H:%M:%S", tz="UTC"))
        axes[-1].set_xlabel("Actual start (UTC)")

    plots.draw("overnight_mux_trend", draw)
