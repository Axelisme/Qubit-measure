"""Committed overnight scalar projection and its public plot product."""

from datetime import UTC, datetime

import numpy as np
from pydantic import TypeAdapter
from zcu_tools.experiment.workflows import CommittedRecord
from zcu_tools.experiment.workflows.encoding import encode_record
from zcu_tools.plotting.plots import NonPresentingHost, Plots

import zcu_lab.workflows as workflows
from zcu_lab.workflows.overnight import OvernightPoint


def committed_points() -> tuple[CommittedRecord, ...]:
    """Independent scalar examples, including rejection and missing evidence."""
    return tuple(
        CommittedRecord(
            index + 1, index + 1, 0, encode_record(point), f"iter/{index + 1:06}", ()
        )
        for index, point in enumerate(
            (
                OvernightPoint(
                    0,
                    datetime(2026, 1, 1, tzinfo=UTC),
                    datetime(2026, 1, 1, tzinfo=UTC),
                    0.2,
                    30.0,
                    1.0,
                    None,
                    40.0,
                    2.0,
                    None,
                ),
                OvernightPoint(
                    1,
                    datetime(2026, 1, 1, 1, tzinfo=UTC),
                    datetime(2026, 1, 1, 1, tzinfo=UTC),
                    0.2,
                    None,
                    None,
                    "T1 fit rejected",
                    42.0,
                    2.5,
                    None,
                ),
                OvernightPoint(
                    2,
                    datetime(2026, 1, 1, 2, tzinfo=UTC),
                    datetime(2026, 1, 1, 2, tzinfo=UTC),
                    0.2,
                    31.0,
                    1.1,
                    None,
                    None,
                    None,
                    "acquisition failed",
                ),
                OvernightPoint(
                    3,
                    datetime(2026, 1, 1, 3, tzinfo=UTC),
                    datetime(2026, 1, 1, 3, tzinfo=UTC),
                    0.2,
                    None,
                    None,
                    None,
                    45.0,
                    None,
                    None,
                ),
            )
        )
    )


def test_projection_preserves_committed_times_missing_values_and_reasons() -> None:
    records = committed_points()
    points = workflows.project_overnight_trend(records)
    assert [point.actual_start for point in points] == [
        datetime(2026, 1, 1, hour, tzinfo=UTC) for hour in range(4)
    ]
    assert [point.t1_us for point in points] == [30.0, None, 31.0, None]
    assert [point.t2_us for point in points] == [40.0, 42.0, None, 45.0]
    assert [point.t1_reason for point in points] == [
        None,
        "T1 fit rejected",
        None,
        None,
    ]
    assert [point.t2_reason for point in points] == [
        None,
        None,
        "acquisition failed",
        None,
    ]
    assert points[0].t1_error_us == 1.0
    assert workflows.project_overnight_trend(()) == ()
    # The stored JSON projection and the live tuple produce identical evidence.
    assert (
        workflows.project_overnight_trend(
            TypeAdapter(tuple[CommittedRecord, ...]).validate_json(
                TypeAdapter(tuple[CommittedRecord, ...]).dump_json(records)
            )
        )
        == points
    )


def test_plot_updates_same_named_figure_without_fabricating_missing_values() -> None:
    plots = Plots(NonPresentingHost())
    records = committed_points()
    workflows.draw_overnight_trend(plots, records[:1])
    figure = plots["overnight_trend"]
    workflows.draw_overnight_trend(plots, records)
    assert plots["overnight_trend"] is figure
    t1_axes, t2_axes = figure.axes
    np.testing.assert_allclose(
        t1_axes.lines[0].get_ydata(), [30.0, np.nan, 31.0, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        t2_axes.lines[0].get_ydata(), [40.0, 42.0, np.nan, 45.0], equal_nan=True
    )
    assert "T1 fit rejected" in [text.get_text() for text in t1_axes.texts]
    assert "acquisition failed" in [text.get_text() for text in t2_axes.texts]
    plots.finish()
    plots.release()
