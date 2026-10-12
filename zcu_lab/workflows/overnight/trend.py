"""Overnight scalar trends drawn only from journal-authoritative commits."""

from __future__ import annotations

from collections.abc import Sequence
from math import nan

from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from pydantic import TypeAdapter
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.plotting.plots import Plots

from zcu_lab.workflows.overnight.workflow import OvernightPoint

_point_adapter = TypeAdapter(OvernightPoint)


def project_overnight_trend(
    records: Sequence[CommittedRecord],
) -> tuple[OvernightPoint, ...]:
    """Decode committed overnight summaries in input commit order.

    records must come from Engine.records for an overnight run. Returns detached
    scalar evidence in aware UTC actual-start time and microseconds, retaining
    None uncertainties, values and diagnoses.
    Empty commits return (). Repr-degraded or invalid summaries raise ValueError;
    no native trace, GUI draft, fit, threshold or missing-value substitute is used.
    """
    points: list[OvernightPoint] = []
    for record in records:
        if record.encoded_record.mode != "json":
            raise ValueError(
                f"Overnight commit {record.commit_seq} has no scalar JSON evidence"
            )
        point = _point_adapter.validate_python(record.encoded_record.value)
        offset = point.actual_start.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("Overnight actual_start must be aware UTC")
        points.append(point)
    return tuple(points)


def draw_overnight_trend(plots: Plots, records: Sequence[CommittedRecord]) -> None:
    """Update the named overnight_trend Figure from committed scalar evidence.

    plots is the host's active operation-owned Plots, not finished/released.
    Both members use actual UTC start times and lifetimes in microseconds.
    Missing/rejected values remain NaN gaps; diagnoses appear at the bottom of
    each axes in axes coordinates, never as fabricated lifetime measurements.
    This synchronous owner-thread draw refreshes one Figure, including zero
    commits. Scalar decoding and host/draw failures propagate without retry.
    """
    points = project_overnight_trend(records)
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
        locator = AutoDateLocator(tz="UTC")
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(ConciseDateFormatter(locator, tz="UTC"))
        axes[-1].set_xlabel("Actual start (UTC)")

    plots.draw("overnight_trend", draw)
