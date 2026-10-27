"""Flux coordinates from self-contained nested records and dotted CSV rows."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
from zcu_tools.resources.record_table import flux_from_frame


def make_table(rows: Sequence[Mapping[str, object]], *, flatten: bool) -> pd.DataFrame:
    records = [dict(row) for row in rows]
    return pd.json_normalize(records) if flatten else pd.DataFrame(records)


@pytest.mark.parametrize("flatten", [False, True])
def test_flux_uses_explicit_then_offset_period_frame_then_fallback(
    flatten: bool,
) -> None:
    rows = [
        {
            "flux": 0.8,
            "flux_frame": {"offset": "unused"},
            "devices": {"bias": {"value": "unused"}},
        },
        {
            "flux_frame": {"offset": 0.5, "period": {"bias": 4, "coil": -2}},
            "devices": {"bias": {"value": 3}, "coil": {"value": 0.5}},
        },
        {"devices": {"bias": {"value": 4, "value_unit": "V"}}},
    ]
    table = make_table(rows, flatten=flatten)
    table.index = ["explicit", "row", "fallback"]
    original = table.copy(deep=True)
    actual = flux_from_frame(
        table, fallback_frame=FluxFrame(-0.25, {"bias": DevicePeriod(8, "V")})
    )
    np.testing.assert_allclose(actual, [0.8, 1.0, 0.25])
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("flatten", [False, True])
def test_missing_offset_names_row_and_field_without_falling_back(flatten: bool) -> None:
    rows = [
        {"flux_frame": {"period": {"coil": -2}}, "devices": {"coil": {"value": 0.5}}}
    ]
    table = make_table(rows, flatten=flatten)
    table.index = ["partial"]
    with pytest.raises(ValueError, match=r"partial.*flux_frame.offset"):
        flux_from_frame(table, fallback_frame=FluxFrame(0, {}))


@pytest.mark.parametrize("flatten", [False, True])
def test_absent_device_values_contribute_zero(flatten: bool) -> None:
    rows = [
        {
            "flux_frame": {"offset": 0.5, "period": {"bias": 4, "coil": -2}},
            "devices": {"bias": {"value": 3, "value_unit": "V"}},
        }
    ]
    table = make_table(rows, flatten=flatten)
    np.testing.assert_allclose(flux_from_frame(table), [1.25])
    fallback_table = table.drop(
        columns=[c for c in table.columns if str(c).startswith("flux_frame")]
    )
    np.testing.assert_allclose(
        flux_from_frame(
            fallback_table,
            fallback_frame=FluxFrame(
                0.5, {"bias": DevicePeriod(4, "V"), "coil": DevicePeriod(-2, "A")}
            ),
        ),
        [1.25],
    )


@pytest.mark.parametrize("unit", ["V", None])
@pytest.mark.parametrize("flatten", [False, True])
def test_fallback_checks_unit_for_present_values(
    unit: str | None, flatten: bool
) -> None:
    rows = [{"devices": {"coil": {"value": 3, "value_unit": unit}}}]
    table = make_table(rows, flatten=flatten)
    table.index = ["old"]
    with pytest.raises(ValueError, match=r"old.*devices.coil.value_unit"):
        flux_from_frame(
            table, fallback_frame=FluxFrame(0.5, {"coil": DevicePeriod(-4, "A")})
        )


@pytest.mark.parametrize("name", ["bias.local", "", " \t "])
@pytest.mark.parametrize("flatten", [False, True])
def test_invalid_used_frame_device_names_are_rejected(name: str, flatten: bool) -> None:
    rows = [{"flux_frame": {"offset": 0.5, "period": {name: 4}}}]
    table = make_table(rows, flatten=flatten)
    table.index = ["sample"]
    with pytest.raises(ValueError, match="sample.*(?i:device name)"):
        flux_from_frame(table)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -float("inf"), "1"])
def test_present_invalid_nested_device_value_names_row_and_column(bad: float) -> None:
    table = pd.DataFrame(
        [
            {
                "flux_frame": {"offset": 0.5, "period": {"bias": 4}},
                "devices": {"bias": {"value": bad}},
            }
        ],
        index=["invalid"],
    )
    with pytest.raises(ValueError, match=r"invalid.*devices.bias.value"):
        flux_from_frame(table)


def test_dotted_union_treats_nan_value_and_period_cells_as_absent() -> None:
    rows = [
        {
            "flux_frame": {"offset": 0.5, "period": {"bias": 4, "coil": -2}},
            "devices": {"bias": {"value": 3}},
        },
        {
            "flux_frame": {"offset": 0.5, "period": {"coil": -2}},
            "devices": {"coil": {"value": 0.5}},
        },
    ]
    table = pd.json_normalize(rows)
    np.testing.assert_allclose(flux_from_frame(table), [1.25, 0.25])
    values_only = table.drop(
        columns=[c for c in table.columns if c.startswith("flux_frame")]
    )
    # Absent value cells need no unit, while present values must match fallback.
    values_only["devices.bias.value_unit"] = ["V", None]
    values_only["devices.coil.value_unit"] = [None, "A"]
    np.testing.assert_allclose(
        flux_from_frame(
            values_only,
            fallback_frame=FluxFrame(
                0.5, {"bias": DevicePeriod(4, "V"), "coil": DevicePeriod(-2, "A")}
            ),
        ),
        [1.25, 0.25],
    )


@pytest.mark.parametrize("field", ["flux_frame.period.bias", "devices.bias.value"])
@pytest.mark.parametrize("bad", [True, float("inf"), -float("inf")])
def test_dotted_invalid_non_nan_cells_still_report_row_and_field(
    field: str, bad: float
) -> None:
    table = pd.DataFrame(
        {
            "flux_frame.offset": [0.5],
            "flux_frame.period.bias": [4],
            "devices.bias.value": [3],
        },
        index=["invalid"],
        dtype=object,
    )
    table.loc["invalid", field] = bad
    with pytest.raises(ValueError, match="invalid.*" + field):
        flux_from_frame(table)


@pytest.mark.parametrize("flatten", [False, True])
def test_nan_period_is_only_absent_after_flattening(flatten: bool) -> None:
    rows = [
        {
            "flux_frame": {"offset": 0.5, "period": {"bias": float("nan")}},
            "devices": {"bias": {"value": 3}},
        }
    ]
    table = make_table(rows, flatten=flatten)
    if flatten:
        np.testing.assert_allclose(flux_from_frame(table), [0.5])
    else:
        with pytest.raises(ValueError, match=r"flux_frame.period.bias"):
            flux_from_frame(table)


@pytest.mark.parametrize("raw", [{}, {"offset": None}, {"period": {}}])
def test_nested_missing_offset_never_uses_fallback(raw: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"partial.*flux_frame.offset"):
        flux_from_frame(
            pd.DataFrame([{"flux_frame": raw}], index=["partial"]),
            fallback_frame=FluxFrame(0.5, {}),
        )


@pytest.mark.parametrize("flatten", [False, True])
def test_zero_period_reports_row_and_field(flatten: bool) -> None:
    rows = [{"flux_frame": {"offset": 0.5, "period": {"bias": 0}}}]
    table = make_table(rows, flatten=flatten)
    table.index = ["bad-period"]
    with pytest.raises(ValueError, match=r"bad-period.*flux_frame.period.bias"):
        flux_from_frame(table)


def test_no_frame_reports_row_and_empty_table_returns_empty_array() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        flux_from_frame(pd.DataFrame([{"devices": {}}], index=["unresolved"]))
    assert flux_from_frame(pd.DataFrame()).shape == (0,)
