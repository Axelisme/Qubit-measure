"""Tests for the SampleTable v2 flat coordinate contract (sample_schema.py)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from zcu_tools.meta_tool import (
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    FLUX_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
    SAMPLE_COORDINATE_COLUMNS,
    DeviceValueUnit,
    LegacyDeviceValueUnit,
    LegacySampleFluxFrame,
    SampleFluxFrame,
    SampleTableV2Error,
    migrate_sample_table_v2,
    resolve_sample_flux,
    validate_sample_table_v2,
)


def _assert_reason(exc: SampleTableV2Error, reason: str) -> None:
    assert exc.reason == reason


# ---------------------------------------------------------------------------
# Shared constants (A1)


def test_coordinate_constants_unique_and_in_order() -> None:
    expected = (
        FLUX_COLUMN,
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
        FLUX_INT_COLUMN,
        FLUX_PERIOD_COLUMN,
    )
    assert SAMPLE_COORDINATE_COLUMNS == expected
    assert FLUX_COLUMN == "flux"
    assert DEV_VALUE_COLUMN == "dev_value"
    assert DEV_UNIT_COLUMN == "dev_unit"
    assert FLUX_INT_COLUMN == "flux_int"
    assert FLUX_PERIOD_COLUMN == "flux_period"
    assert len(set(SAMPLE_COORDINATE_COLUMNS)) == len(SAMPLE_COORDINATE_COLUMNS)


# ---------------------------------------------------------------------------
# Validation (A1)


def test_validate_accepts_minimal_v2_table() -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [-0.007, 0.002, 1.5],
            DEV_UNIT_COLUMN: ["A", "A", "V"],
        }
    )
    validate_sample_table_v2(samples)


def test_validate_accepts_full_v2_table_with_mixed_units() -> None:
    samples = pd.DataFrame(
        {
            FLUX_COLUMN: [0.5, np.nan, np.nan],
            DEV_VALUE_COLUMN: [-0.007, 0.002, 1.5],
            DEV_UNIT_COLUMN: ["A", "A", "V"],
            FLUX_INT_COLUMN: [np.nan, -0.001, 0.0],
            FLUX_PERIOD_COLUMN: [np.nan, 0.0246, 0.05],
        }
    )
    validate_sample_table_v2(samples)


def test_validate_requires_dev_value_and_dev_unit() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(pd.DataFrame({FLUX_COLUMN: [0.5]}))
    _assert_reason(exc.value, "missing_required_column")
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(pd.DataFrame({DEV_VALUE_COLUMN: [1.0]}))
    _assert_reason(exc.value, "missing_required_column")


def test_validate_rejects_invalid_dev_unit() -> None:
    samples = pd.DataFrame({DEV_VALUE_COLUMN: [1.0], DEV_UNIT_COLUMN: ["mA"]})
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    _assert_reason(exc.value, "invalid_unit")


def test_validate_rejects_null_and_nonfinite_dev_value() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(
            pd.DataFrame({DEV_VALUE_COLUMN: [np.nan], DEV_UNIT_COLUMN: ["A"]})
        )
    _assert_reason(exc.value, "null_required_value")
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(
            pd.DataFrame({DEV_VALUE_COLUMN: [np.inf], DEV_UNIT_COLUMN: ["A"]})
        )
    _assert_reason(exc.value, "non_finite_value")


def test_validate_rejects_non_numeric_coordinate() -> None:
    samples = pd.DataFrame({DEV_VALUE_COLUMN: ["high"], DEV_UNIT_COLUMN: ["A"]})
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    _assert_reason(exc.value, "non_numeric_coordinate")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(pd.Timestamp("2020-01-01"), id="timestamp"),
        pytest.param(pd.Timedelta(days=1), id="timedelta"),
        pytest.param(True, id="boolean"),
        pytest.param(1 + 2j, id="complex"),
    ],
)
def test_validate_rejects_non_real_numeric_coordinate(value: object) -> None:
    samples = pd.DataFrame({DEV_VALUE_COLUMN: [value], DEV_UNIT_COLUMN: ["A"]})
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    _assert_reason(exc.value, "non_numeric_coordinate")


def test_validate_rejects_duplicate_columns() -> None:
    samples = pd.DataFrame(
        [[1.0, "A", 2.0]],
        columns=[DEV_VALUE_COLUMN, DEV_UNIT_COLUMN, DEV_VALUE_COLUMN],
    )
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    _assert_reason(exc.value, "duplicate_columns")


@pytest.mark.parametrize(
    "stale", ["calibrated mA", "calibrated A", "Flux", "flux_bias"]
)
def test_validate_rejects_stale_legacy_coordinate_columns(stale: str) -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [1.0],
            DEV_UNIT_COLUMN: ["A"],
            stale: [0.5],
        }
    )
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    _assert_reason(exc.value, "legacy_coordinate_column")


def test_validate_rejects_orphan_frame_columns() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(
            pd.DataFrame(
                {
                    DEV_VALUE_COLUMN: [1.0],
                    DEV_UNIT_COLUMN: ["A"],
                    FLUX_INT_COLUMN: [0.0],
                }
            )
        )
    _assert_reason(exc.value, "orphan_frame_column")
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(
            pd.DataFrame(
                {
                    DEV_VALUE_COLUMN: [1.0],
                    DEV_UNIT_COLUMN: ["A"],
                    FLUX_PERIOD_COLUMN: [2.0],
                }
            )
        )
    _assert_reason(exc.value, "orphan_frame_column")


def test_validate_rejects_row_orphan_frame_cell() -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [1.0, 2.0],
            DEV_UNIT_COLUMN: ["A", "A"],
            FLUX_INT_COLUMN: [0.0, np.nan],
            FLUX_PERIOD_COLUMN: [2.0, 2.0],  # row 0 both, row 1 only period
        }
    )
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    _assert_reason(exc.value, "orphan_frame_column")
    assert exc.value.data["indexes"] == [1]


@pytest.mark.parametrize("period", [0.0, -1.0, np.inf])
def test_validate_rejects_non_positive_or_non_finite_period(period: float) -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [1.0],
            DEV_UNIT_COLUMN: ["A"],
            FLUX_INT_COLUMN: [0.0],
            FLUX_PERIOD_COLUMN: [period],
        }
    )
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(samples)
    assert exc.value.reason in ("invalid_period", "non_finite_value")


def test_validate_rejects_non_finite_explicit_flux_but_allows_null() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(
            pd.DataFrame(
                {
                    FLUX_COLUMN: [np.inf],
                    DEV_VALUE_COLUMN: [1.0],
                    DEV_UNIT_COLUMN: ["A"],
                }
            )
        )
    _assert_reason(exc.value, "non_finite_value")
    validate_sample_table_v2(
        pd.DataFrame(
            {
                FLUX_COLUMN: [np.nan, 0.25],
                DEV_VALUE_COLUMN: [1.0, 2.0],
                DEV_UNIT_COLUMN: ["A", "A"],
            }
        )
    )


def test_validate_empty_table_modes() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(pd.DataFrame())
    _assert_reason(exc.value, "empty_table")
    validate_sample_table_v2(pd.DataFrame(), allow_empty=True)
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(pd.DataFrame(index=[0]), allow_empty=True)
    _assert_reason(exc.value, "missing_required_column")
    headers = pd.DataFrame(columns=[DEV_VALUE_COLUMN, DEV_UNIT_COLUMN])
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(headers)
    _assert_reason(exc.value, "empty_table")
    validate_sample_table_v2(headers, allow_empty=True)
    stale_headers = pd.DataFrame(columns=[DEV_VALUE_COLUMN, DEV_UNIT_COLUMN, "Flux"])
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(stale_headers, allow_empty=True)
    _assert_reason(exc.value, "legacy_coordinate_column")
    orphan_headers = pd.DataFrame(
        columns=[DEV_VALUE_COLUMN, DEV_UNIT_COLUMN, FLUX_INT_COLUMN]
    )
    with pytest.raises(SampleTableV2Error) as exc:
        validate_sample_table_v2(orphan_headers, allow_empty=True)
    _assert_reason(exc.value, "orphan_frame_column")


# ---------------------------------------------------------------------------
# Frames (A1)


def test_sample_flux_frame_rejects_invalid_frames() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        SampleFluxFrame(
            dev_unit=cast(DeviceValueUnit, "mA"), flux_int=0.0, flux_period=1.0
        )
    _assert_reason(exc.value, "invalid_unit")
    with pytest.raises(SampleTableV2Error) as exc:
        SampleFluxFrame(dev_unit="A", flux_int=np.nan, flux_period=1.0)
    _assert_reason(exc.value, "non_finite_value")
    with pytest.raises(SampleTableV2Error) as exc:
        SampleFluxFrame(dev_unit="A", flux_int=0.0, flux_period=0.0)
    _assert_reason(exc.value, "invalid_period")
    with pytest.raises(SampleTableV2Error) as exc:
        SampleFluxFrame(dev_unit="A", flux_int=0.0, flux_period=np.inf)
    assert exc.value.reason in ("invalid_period", "non_finite_value")


def test_sample_flux_frame_round_trip_scalar_and_array() -> None:
    frame = SampleFluxFrame(dev_unit="A", flux_int=-0.001, flux_period=0.0246)
    flux = frame.flux_from_dev_value(0.0113)
    assert flux == pytest.approx((0.0113 + 0.001) / 0.0246)
    assert frame.dev_value_from_flux(flux) == pytest.approx(0.0113)
    values = np.array([-0.007, 0.002])
    fluxs = frame.flux_from_dev_value(values)
    np.testing.assert_allclose(fluxs, (values + 0.001) / 0.0246)
    np.testing.assert_allclose(frame.dev_value_from_flux(fluxs), values)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(datetime(2020, 1, 1), id="python-datetime"),
        pytest.param(timedelta(days=1), id="python-timedelta"),
        pytest.param(True, id="python-boolean"),
        pytest.param(1 + 2j, id="python-complex"),
        pytest.param(np.datetime64("2020-01-01"), id="numpy-datetime"),
        pytest.param(np.timedelta64(1, "D"), id="numpy-timedelta"),
        pytest.param(np.bool_(True), id="numpy-boolean"),
        pytest.param(np.complex128(1 + 2j), id="numpy-complex"),
    ],
)
def test_sample_flux_frame_rejects_non_real_scalar_conversion_input(
    value: object,
) -> None:
    frame = SampleFluxFrame(dev_unit="A", flux_int=0.0, flux_period=1.0)
    invalid = cast(float, value)
    for convert in (frame.flux_from_dev_value, frame.dev_value_from_flux):
        with pytest.raises(SampleTableV2Error) as exc:
            convert(invalid)
        _assert_reason(exc.value, "non_numeric_coordinate")


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array(["2020-01-01"], dtype="datetime64[D]"), id="datetime"),
        pytest.param(np.array([1], dtype="timedelta64[D]"), id="timedelta"),
        pytest.param(np.array([True], dtype=np.bool_), id="boolean"),
        pytest.param(np.array([1 + 2j], dtype=np.complex128), id="complex"),
    ],
)
def test_sample_flux_frame_rejects_non_real_array_conversion_input(
    values: object,
) -> None:
    frame = SampleFluxFrame(dev_unit="A", flux_int=0.0, flux_period=1.0)
    invalid = cast(NDArray[np.float64], values)
    for convert in (frame.flux_from_dev_value, frame.dev_value_from_flux):
        with pytest.raises(SampleTableV2Error) as exc:
            convert(invalid)
        _assert_reason(exc.value, "non_numeric_coordinate")


def test_legacy_sample_flux_frame_rejects_invalid_frames() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        LegacySampleFluxFrame(
            dev_unit=cast(LegacyDeviceValueUnit, "kV"),
            flux_int=0.0,
            flux_period=1.0,
        )
    _assert_reason(exc.value, "invalid_unit")
    with pytest.raises(SampleTableV2Error) as exc:
        LegacySampleFluxFrame(dev_unit="mA", flux_int=np.nan, flux_period=1.0)
    _assert_reason(exc.value, "non_finite_value")
    with pytest.raises(SampleTableV2Error) as exc:
        LegacySampleFluxFrame(dev_unit="mA", flux_int=0.0, flux_period=-2.0)
    _assert_reason(exc.value, "invalid_period")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(pd.Timestamp("2020-01-01"), id="timestamp"),
        pytest.param(pd.Timedelta(days=1), id="timedelta"),
        pytest.param(True, id="boolean"),
        pytest.param(1 + 0j, id="complex"),
    ],
)
def test_flux_frames_reject_non_real_numeric_values(value: object) -> None:
    invalid = cast(float, value)
    with pytest.raises(SampleTableV2Error) as exc:
        SampleFluxFrame(dev_unit="A", flux_int=invalid, flux_period=1.0)
    _assert_reason(exc.value, "non_numeric_coordinate")
    with pytest.raises(SampleTableV2Error) as exc:
        SampleFluxFrame(dev_unit="A", flux_int=0.0, flux_period=invalid)
    _assert_reason(exc.value, "non_numeric_coordinate")
    with pytest.raises(SampleTableV2Error) as exc:
        LegacySampleFluxFrame(dev_unit="mA", flux_int=invalid, flux_period=1.0)
    _assert_reason(exc.value, "non_numeric_coordinate")
    with pytest.raises(SampleTableV2Error) as exc:
        LegacySampleFluxFrame(dev_unit="mA", flux_int=0.0, flux_period=invalid)
    _assert_reason(exc.value, "non_numeric_coordinate")


# ---------------------------------------------------------------------------
# Resolution (A2)


def test_resolve_explicit_flux_wins_over_row_frame() -> None:
    samples = pd.DataFrame(
        {
            FLUX_COLUMN: [0.30],
            DEV_VALUE_COLUMN: [0.0],  # row frame would derive (0 - 0)/2 = 0.0
            DEV_UNIT_COLUMN: ["A"],
            FLUX_INT_COLUMN: [0.0],
            FLUX_PERIOD_COLUMN: [2.0],
        }
    )
    resolution = resolve_sample_flux(samples)
    np.testing.assert_allclose(resolution.values, [0.30])
    assert resolution.sources == ("explicit",)
    np.testing.assert_array_equal(resolution.explicit_mask, [True])


def test_resolve_row_frame_derivation() -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [0.0113, -0.007],
            DEV_UNIT_COLUMN: ["A", "A"],
            FLUX_INT_COLUMN: [-0.001, 0.0],
            FLUX_PERIOD_COLUMN: [0.0246, 0.02],
        }
    )
    resolution = resolve_sample_flux(samples)
    expected = np.array([(0.0113 + 0.001) / 0.0246, (-0.007 - 0.0) / 0.02])
    np.testing.assert_allclose(resolution.values, expected)
    assert resolution.sources == ("row-frame", "row-frame")
    np.testing.assert_array_equal(resolution.explicit_mask, [False, False])


def test_resolve_fallback_frame_derivation() -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [0.0113],
            DEV_UNIT_COLUMN: ["A"],
        }
    )
    fallback = SampleFluxFrame(dev_unit="A", flux_int=-0.001, flux_period=0.0246)
    resolution = resolve_sample_flux(samples, fallback_frame=fallback)
    assert resolution.sources == ("fallback-frame",)
    np.testing.assert_allclose(resolution.values, [(0.0113 + 0.001) / 0.0246])
    np.testing.assert_array_equal(resolution.explicit_mask, [False])


def test_resolve_precedence_closed_provenance_combined() -> None:
    samples = pd.DataFrame(
        {
            FLUX_COLUMN: [0.5, np.nan, np.nan, np.nan],
            DEV_VALUE_COLUMN: [0.0, 1.0, 3.0, 2.0],
            DEV_UNIT_COLUMN: ["A", "A", "A", "A"],
            FLUX_INT_COLUMN: [np.nan, 0.0, np.nan, np.nan],
            FLUX_PERIOD_COLUMN: [np.nan, 2.0, np.nan, np.nan],
        }
    )
    fallback = SampleFluxFrame(dev_unit="A", flux_int=1.0, flux_period=4.0)
    resolution = resolve_sample_flux(samples, fallback_frame=fallback)
    np.testing.assert_allclose(resolution.values, [0.5, 0.5, 0.5, 0.25])
    assert resolution.sources == (
        "explicit",
        "row-frame",
        "fallback-frame",
        "fallback-frame",
    )
    np.testing.assert_array_equal(resolution.explicit_mask, [True, False, False, False])


def test_resolve_fallback_unit_mismatch_fails() -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [0.0113],
            DEV_UNIT_COLUMN: ["A"],
        }
    )
    fallback = SampleFluxFrame(dev_unit="V", flux_int=0.0, flux_period=0.05)
    with pytest.raises(SampleTableV2Error) as exc:
        resolve_sample_flux(samples, fallback_frame=fallback)
    _assert_reason(exc.value, "unresolved_flux")
    assert exc.value.data["indexes"] == [0]


def test_resolve_unresolved_rows_fail_with_indexes() -> None:
    samples = pd.DataFrame(
        {
            FLUX_COLUMN: [np.nan, 0.5, np.nan],
            DEV_VALUE_COLUMN: [1.0, 2.0, 3.0],
            DEV_UNIT_COLUMN: ["A", "A", "A"],
        }
    )
    with pytest.raises(SampleTableV2Error) as exc:
        resolve_sample_flux(samples)
    _assert_reason(exc.value, "unresolved_flux")
    assert exc.value.data["indexes"] == [0, 2]


def test_resolve_mixed_units_with_matching_fallback() -> None:
    samples = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [1.0, 1.0],
            DEV_UNIT_COLUMN: ["A", "V"],
        }
    )
    fallback = SampleFluxFrame(dev_unit="A", flux_int=0.0, flux_period=2.0)
    # Row 1 (V) has no matching fallback -> unresolved.
    with pytest.raises(SampleTableV2Error) as exc:
        resolve_sample_flux(samples, fallback_frame=fallback)
    assert exc.value.data["indexes"] == [1]
    samples_with_frames = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: [1.0, 1.0],
            DEV_UNIT_COLUMN: ["A", "V"],
            FLUX_INT_COLUMN: [0.0, 0.0],
            FLUX_PERIOD_COLUMN: [2.0, 5.0],
        }
    )
    resolution = resolve_sample_flux(samples_with_frames, fallback_frame=fallback)
    np.testing.assert_allclose(resolution.values, [0.5, 0.2])
    assert resolution.sources == ("row-frame", "row-frame")


def test_resolve_is_deterministic() -> None:
    samples = pd.DataFrame(
        {
            FLUX_COLUMN: [0.5, np.nan],
            DEV_VALUE_COLUMN: [0.0, 1.0],
            DEV_UNIT_COLUMN: ["A", "A"],
            FLUX_INT_COLUMN: [np.nan, 0.0],
            FLUX_PERIOD_COLUMN: [np.nan, 2.0],
        }
    )
    first = resolve_sample_flux(samples)
    second = resolve_sample_flux(samples)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.sources == second.sources
    np.testing.assert_array_equal(first.explicit_mask, [True, False])
    np.testing.assert_array_equal(
        first.explicit_mask,
        np.asarray([s == "explicit" for s in first.sources], dtype=np.bool_),
    )


def test_resolve_empty_table() -> None:
    resolution = resolve_sample_flux(
        pd.DataFrame(columns=[DEV_VALUE_COLUMN, DEV_UNIT_COLUMN])
    )
    assert len(resolution.values) == 0
    assert resolution.sources == ()
    assert len(resolution.explicit_mask) == 0


# ---------------------------------------------------------------------------
# Migration (A3)


@pytest.mark.parametrize(
    ("unit", "scale"),
    [("A", 1.0), ("mA", 1.0e-3), ("V", 1.0), ("mV", 1.0e-3)],
)
def test_migrate_converts_declared_unit_to_base(
    unit: LegacyDeviceValueUnit, scale: float
) -> None:
    samples = pd.DataFrame({"calibrated mA": [1.0, -2.5, 0.0]})
    migrated = migrate_sample_table_v2(
        samples, dev_value_column="calibrated mA", dev_value_unit=unit
    )
    expected_base = "A" if unit in ("A", "mA") else "V"
    np.testing.assert_allclose(
        migrated[DEV_VALUE_COLUMN], np.array([1.0, -2.5, 0.0]) * scale
    )
    assert list(migrated[DEV_UNIT_COLUMN]) == [expected_base] * 3
    validate_sample_table_v2(migrated)


def test_migrate_preserves_measurement_columns_and_row_order() -> None:
    samples = pd.DataFrame(
        {
            "T1 (us)": [50.0, 60.0],
            "calibrated mA": [0.001, 0.002],
            "Freq (MHz)": [4500.0, 4510.0],
        }
    )
    migrated = migrate_sample_table_v2(
        samples, dev_value_column="calibrated mA", dev_value_unit="mA"
    )
    assert list(migrated.columns) == [
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
        "T1 (us)",
        "Freq (MHz)",
    ]
    assert list(migrated["T1 (us)"]) == [50.0, 60.0]
    assert list(migrated["Freq (MHz)"]) == [4500.0, 4510.0]
    np.testing.assert_allclose(migrated[DEV_VALUE_COLUMN], [1.0e-6, 2.0e-6])


def test_migrate_requires_caller_declared_source_column() -> None:
    samples = pd.DataFrame({"other": [1.0]})
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            samples, dev_value_column="calibrated mA", dev_value_unit="A"
        )
    _assert_reason(exc.value, "missing_source_column")


def test_migrate_rejects_source_with_v2_coordinate_columns() -> None:
    samples = pd.DataFrame(
        {DEV_VALUE_COLUMN: [1.0], DEV_UNIT_COLUMN: ["A"], "T1 (us)": [50.0]}
    )
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            samples, dev_value_column=DEV_VALUE_COLUMN, dev_value_unit="A"
        )
    _assert_reason(exc.value, "source_has_v2_columns")


def test_migrate_single_declared_unit_applies_to_whole_column() -> None:
    samples = pd.DataFrame({"calibrated mA": [0.0, 4.46, -1.2]})
    migrated = migrate_sample_table_v2(
        samples, dev_value_column="calibrated mA", dev_value_unit="mA"
    )
    np.testing.assert_allclose(migrated[DEV_VALUE_COLUMN], [0.0, 0.00446, -0.0012])


def test_migrate_maps_flux_column_only_when_declared() -> None:
    samples = pd.DataFrame(
        {"Flux": [0.0, np.nan, 0.5], "calibrated mA": [1.0, 2.0, 3.0]}
    )
    # Without an explicit --flux-column mapping the stale "Flux" alias would
    # remain in the output, so output validation refuses the migration.
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            samples, dev_value_column="calibrated mA", dev_value_unit="mA"
        )
    _assert_reason(exc.value, "legacy_coordinate_column")
    mapped = migrate_sample_table_v2(
        samples,
        dev_value_column="calibrated mA",
        dev_value_unit="mA",
        flux_column="Flux",
    )
    assert list(mapped.columns) == [
        FLUX_COLUMN,
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
    ]
    np.testing.assert_allclose(mapped[FLUX_COLUMN], [0.0, np.nan, 0.5], equal_nan=True)
    validate_sample_table_v2(mapped)


def test_migrate_frame_normalizes_declared_unit() -> None:
    samples = pd.DataFrame({"calibrated mA": [0.0, 1.0]})
    frame = LegacySampleFluxFrame(dev_unit="mA", flux_int=-11.1, flux_period=24.5)
    migrated = migrate_sample_table_v2(
        samples,
        dev_value_column="calibrated mA",
        dev_value_unit="mA",
        flux_frame=frame,
    )
    assert list(migrated.columns) == [
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
        FLUX_INT_COLUMN,
        FLUX_PERIOD_COLUMN,
    ]
    np.testing.assert_allclose(migrated[FLUX_INT_COLUMN], [-0.0111, -0.0111])
    np.testing.assert_allclose(migrated[FLUX_PERIOD_COLUMN], [0.0245, 0.0245])
    validate_sample_table_v2(migrated)
    resolution = resolve_sample_flux(migrated)
    assert resolution.sources == ("row-frame", "row-frame")


def test_migrate_frame_physical_kind_mismatch_fails() -> None:
    samples = pd.DataFrame({"calibrated mA": [1.0]})
    frame = LegacySampleFluxFrame(dev_unit="mV", flux_int=0.0, flux_period=5.0)
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            samples,
            dev_value_column="calibrated mA",
            dev_value_unit="mA",
            flux_frame=frame,
        )
    _assert_reason(exc.value, "frame_kind_mismatch")


def test_migrate_does_not_mutate_input() -> None:
    samples = pd.DataFrame({"calibrated mA": [1.0, 2.0]})
    original = samples.copy(deep=True)
    migrate_sample_table_v2(
        samples, dev_value_column="calibrated mA", dev_value_unit="A"
    )
    pd.testing.assert_frame_equal(samples, original)


def test_migrate_rejects_null_and_nonfinite_source_values() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            pd.DataFrame({"calibrated mA": [1.0, np.nan]}),
            dev_value_column="calibrated mA",
            dev_value_unit="A",
        )
    _assert_reason(exc.value, "null_required_value")
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            pd.DataFrame({"calibrated mA": [np.inf]}),
            dev_value_column="calibrated mA",
            dev_value_unit="A",
        )
    _assert_reason(exc.value, "non_finite_value")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(pd.Timestamp("2020-01-01"), id="timestamp"),
        pytest.param(pd.Timedelta(days=1), id="timedelta"),
        pytest.param(True, id="boolean"),
        pytest.param(1 + 2j, id="complex"),
    ],
)
def test_migrate_rejects_non_real_numeric_coordinate(value: object) -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            pd.DataFrame({"legacy value": [value]}),
            dev_value_column="legacy value",
            dev_value_unit="A",
        )
    _assert_reason(exc.value, "non_numeric_coordinate")


def test_migrate_rejects_invalid_source_unit() -> None:
    with pytest.raises(SampleTableV2Error) as exc:
        migrate_sample_table_v2(
            pd.DataFrame({"calibrated mA": [1.0]}),
            dev_value_column="calibrated mA",
            dev_value_unit="kA",  # type: ignore[arg-type]
        )
    _assert_reason(exc.value, "invalid_unit")


def test_migrate_keeps_flux_and_frame_columns_in_canonical_order() -> None:
    samples = pd.DataFrame({"T1 (us)": [50.0], "calibrated mA": [1.0], "Flux": [0.25]})
    frame = LegacySampleFluxFrame(dev_unit="mA", flux_int=0.0, flux_period=10.0)
    migrated = migrate_sample_table_v2(
        samples,
        dev_value_column="calibrated mA",
        dev_value_unit="mA",
        flux_column="Flux",
        flux_frame=frame,
    )
    assert list(migrated.columns) == [
        FLUX_COLUMN,
        DEV_VALUE_COLUMN,
        DEV_UNIT_COLUMN,
        FLUX_INT_COLUMN,
        FLUX_PERIOD_COLUMN,
        "T1 (us)",
    ]
    assert list(migrated.columns) == list(SAMPLE_COORDINATE_COLUMNS) + ["T1 (us)"]


# ---------------------------------------------------------------------------
# Audited representative fixtures (A4)

# From research/sample-current-unit-audit.md: result/2DQ12/Q4/samples.csv stores
# A-valued data under the "calibrated mA" column name (values [-0.007, 0.002]).
# From the same audit: result/Q12_2D/Q4/samples.csv stores true mA values under
# "calibrated mA" (values [0, 4.46]). Both migrate equivalently to A-valued
# dev_value under explicit caller-declared units only.


def test_migrate_audited_a_valued_fixture() -> None:
    samples = pd.DataFrame({"calibrated mA": [-0.007, 0.002], "T1 (us)": [50.0, 60.0]})
    migrated = migrate_sample_table_v2(
        samples, dev_value_column="calibrated mA", dev_value_unit="A"
    )
    np.testing.assert_allclose(migrated[DEV_VALUE_COLUMN], [-0.007, 0.002])
    assert list(migrated[DEV_UNIT_COLUMN]) == ["A", "A"]
    validate_sample_table_v2(migrated)


def test_migrate_audited_ma_valued_fixture() -> None:
    samples = pd.DataFrame({"calibrated mA": [0.0, 4.46], "T1 (us)": [50.0, 60.0]})
    migrated = migrate_sample_table_v2(
        samples, dev_value_column="calibrated mA", dev_value_unit="mA"
    )
    np.testing.assert_allclose(migrated[DEV_VALUE_COLUMN], [0.0, 0.00446])
    assert list(migrated[DEV_UNIT_COLUMN]) == ["A", "A"]
    validate_sample_table_v2(migrated)


def test_audited_fixtures_never_infer_unit_from_name_or_magnitude() -> None:
    # Same column name and overlapping value ranges must not change behavior:
    # the declared unit alone drives conversion.
    a_valued = migrate_sample_table_v2(
        pd.DataFrame({"calibrated mA": [-0.007, 0.002]}),
        dev_value_column="calibrated mA",
        dev_value_unit="A",
    )
    ma_valued = migrate_sample_table_v2(
        pd.DataFrame({"calibrated mA": [-0.007, 0.002]}),
        dev_value_column="calibrated mA",
        dev_value_unit="mA",
    )
    np.testing.assert_allclose(a_valued[DEV_VALUE_COLUMN], [-0.007, 0.002])
    np.testing.assert_allclose(ma_valued[DEV_VALUE_COLUMN], [-7.0e-6, 2.0e-6])
