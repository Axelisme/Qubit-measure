"""Small public support-seam examples, independent of numerical kernels."""

import pytest
from zcu_tools.simulate.evolution import PathDiagnostics

from .accuracy import (
    AccuracyMeasurement,
    accuracy_table,
    measurement_property,
    reported_measurements,
    require_families,
)


def test_missing_expected_family_fails_even_when_another_family_ran() -> None:
    paths = PathDiagnostics(segments_by_kernel={"dop853": 2})
    with pytest.raises(AssertionError, match="t6"):
        require_families(paths, ("dop853", "t6"))


def test_report_keeps_failed_error_and_worst_case() -> None:
    lines = accuracy_table(
        [
            AccuracyMeasurement("small", "t6", 1e-4, 2e-10),
            AccuracyMeasurement("coherent", "t6", 0.003, 3e-10),
        ],
        ("t6",),
    )
    row = next(line for line in lines if line.startswith("t6 "))
    assert row.split() == [
        "t6",
        "2",
        "3.000000e-03",
        "3.000000e-10",
        "coherent",
        "FAIL",
    ]


def test_report_keeps_each_guard_edge_visible_below_the_family_maximum() -> None:
    lines = accuracy_table(
        [
            AccuracyMeasurement("representative", "t6", 1e-3, 2e-10),
            AccuracyMeasurement("guard_edge/t6", "t6", 1e-4, 3e-10),
            AccuracyMeasurement("guard_edge/gate_cache", "gate_cache", 0.0093, 1e-12),
        ],
        ("t6", "gate_cache"),
    )
    assert "t6 guard_edge/t6 1.000000e-04 3.000000e-10 PASS" in lines
    assert "gate_cache guard_edge/gate_cache 9.300000e-03 1.000000e-12 FAIL" in lines


def test_report_does_not_turn_unmeasured_family_into_zero_error() -> None:
    lines = accuracy_table([], ("floquet",))
    assert lines[1:] == ["floquet 0 - - - NOT EXERCISED"]


def test_transported_results_keep_boundary_pass_and_failed_observations() -> None:
    measurements = [
        AccuracyMeasurement("boundary", "t4", 2e-3, 1e-12),
        AccuracyMeasurement("known_defect", "t6", 2.1e-3, 2e-12),
    ]
    properties: list[tuple[str, object]] = [("unrelated", 7)]
    properties.extend(measurement_property(item) for item in measurements)
    restored = reported_measurements(properties)
    assert restored == measurements
    rows = accuracy_table(restored, ("t4", "t6"))
    assert rows[1].split() == [
        "t4",
        "1",
        "2.000000e-03",
        "1.000000e-12",
        "boundary",
        "PASS",
    ]
    assert rows[2].split()[-2:] == ["known_defect", "FAIL"]
