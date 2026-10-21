"""Neutral diagnostic aggregation retains distinct caller point identities."""

from contextlib import closing
from dataclasses import replace
from inspect import signature
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate import evolution

from ._engine import batch, source


def _merge(parts):
    merge = getattr(evolution, "merge_path_diagnostics", None)
    assert callable(merge), "evolution must own diagnostic membership merging"
    return merge(parts)


def test_engine_diagnostic_members_are_caller_ids_and_reuse_merges_directly():
    with closing(evolution.Engine(1, 1024**2)) as engine:
        src = source()
        timeline = replace(batch(2), points=np.array([17, 4]))
        first = engine.evolve(src, (timeline,), cancel=Event(), telemetry=True)
        again = engine.evolve(src, (timeline,), cancel=Event(), telemetry=True)
    assert hasattr(first.paths, "point_members"), "counts alone cannot union span participation"
    assert all(members == (4, 17) for members in first.paths.point_members.values())
    merged = _merge((first.paths, again.paths))
    assert merged.point_members == first.paths.point_members
    assert merged.points_by_class == first.paths.points_by_class
    assert sum(merged.segments_by_kernel.values()) == 4
    for family, report in first.paths.routing.items():
        assert merged.routing[family].rows == report.rows + again.paths.routing[family].rows
        assert merged.routing[family].cache_hits == report.cache_hits + again.paths.routing[family].cache_hits


def test_merge_unions_members_sums_work_and_takes_max_trace():
    assert "point_members" in signature(evolution.PathDiagnostics).parameters
    first = evolution.PathDiagnostics(
        {"t4": 2}, {"idle": 2}, {"t4": 1}, .002,
        {"t4": evolution.FamilyRouting(rows=2, prepare_cpu_s=.1, apply_cpu_s=.2, cache_hits=1, rejected_rows=2, estimated_cost=3., estimated_cpu_s=.4, prepare_builds=2, admission_rows=3)},
        .05, point_members={"idle": (1, 3)},
    )
    second = evolution.PathDiagnostics(
        {"t4": 3}, {"idle": 2, "drive": 1}, {"t4": 2}, .001,
        {"t4": evolution.FamilyRouting(rows=3, prepare_cpu_s=.2, apply_cpu_s=.4, cache_hits=2, rejected_rows=1, estimated_cost=4., estimated_cpu_s=.5, prepare_builds=3, admission_rows=4)},
        .07, point_members={"idle": (3, 9), "drive": (1,)},
    )
    result = _merge((first, second))
    assert result.point_members == {"idle": (1, 3, 9), "drive": (1,)}
    assert result.points_by_class == {"idle": 3, "drive": 1}
    assert result.segments_by_kernel == {"t4": 5}
    assert result.exceeded_by_kernel == {"t4": 3}
    assert result.max_trace_error == .002
    assert result.frontend_cpu_s == pytest.approx(.12)
    report = result.routing["t4"]
    assert (report.rows, report.cache_hits, report.rejected_rows, report.prepare_builds, report.admission_rows) == (5, 3, 3, 5, 7)
    assert report.prepare_cpu_s == pytest.approx(.3)
    assert report.apply_cpu_s == pytest.approx(.6)
    assert report.estimated_cost == 7.
    assert report.estimated_cpu_s == pytest.approx(.9)
    unknown = replace(second, routing={"t4": replace(second.routing["t4"], estimated_cpu_s=None)})
    assert _merge((first, unknown)).routing["t4"].estimated_cpu_s is None
    assert _merge(()) == evolution.PathDiagnostics()


@pytest.mark.parametrize("counts,members", [
    ({"idle": 1}, {}), ({}, {"idle": ()}),
    ({"idle": 1}, {"idle": (1, 2)}), ({"idle": 2}, {"idle": (2, 1)}),
    ({"idle": 2}, {"idle": (1, 1)}), ({"idle": 1}, {"idle": (-1,)}),
])
def test_diagnostic_members_reject_misalignment_and_noncanonical_ids(counts, members):
    assert "point_members" in signature(evolution.PathDiagnostics).parameters
    with pytest.raises(ValueError, match="point"):
        evolution.PathDiagnostics(points_by_class=counts, point_members=members)
