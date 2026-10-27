"""Author-selected device coordinates from record tables and their dotted CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from zcu_tools.analysis.flux import DeviceFluxFrame, FluxFrame
from zcu_tools.resources.record_table import (
    JsonlRecordTable,
    RecordFluxFrame,
    flux_from_frame,
)


@pytest.mark.parametrize("name", ["bias.local", "", " \t "])
def test_flux_rejects_invalid_selected_device_name_even_with_explicit_flux(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="(?i)device.name"):
        flux_from_frame(pd.DataFrame({"flux": [0.5]}), device_name=name)


@pytest.mark.parametrize("name", ["bias.local", "", " \t "])
@pytest.mark.parametrize("flatten", [False, True])
def test_flux_rejects_invalid_used_row_frame_names(name: str, flatten: bool) -> None:
    rows = [
        {
            "devices": {name: {"value": 3}},
            "flux_frame": {name: {"integer": 1, "period": 4}},
        }
    ]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    table.index = ["sample"]
    with pytest.raises(ValueError, match="sample.*(?i:device name)"):
        flux_from_frame(table, device_name="bias")


@pytest.mark.parametrize("name", [" Bias ", "bias", "BIAS", "偏壓", "bias．local"])
@pytest.mark.parametrize("flatten", [False, True])
def test_flux_preserves_names_for_row_frame_and_single_or_multi_fallback(
    name: str, flatten: bool
) -> None:
    rows = [
        {
            "devices": {
                name: {"value": 3, "value_unit": "V"},
                "coil": {"value": 3, "value_unit": "A"},
            },
            "flux_frame": {
                name: {"integer": 1, "period": 4},
                "coil": {"integer": 2, "period": -4},
            },
        },
        {
            "devices": {
                name: {"value": 3, "value_unit": "V"},
                "coil": {"value": 2, "value_unit": "A"},
            }
        },
    ]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    multi = FluxFrame(
        {name: DeviceFluxFrame(1, 4, "V"), "coil": DeviceFluxFrame(2, -4, "A")}
    )
    for fallback in (RecordFluxFrame("V", 1, 4), multi):
        np.testing.assert_allclose(
            flux_from_frame(table, device_name=name, fallback_frame=fallback),
            [0.25, 0.5],
        )


@pytest.mark.parametrize("flatten", [False, True])
def test_flux_uses_explicit_then_row_frame_then_named_fallback(flatten: bool) -> None:
    rows = [
        {
            "flux": 0.8,
            "flux_frame": {"bias": {"integer": "unused"}},
            "devices": {"bias": {"value": "unused"}},
        },
        {
            "devices": {
                "bias": {"value": 3, "value_unit": "arbitrary"},
                "other": {"value": 99},
            },
            "flux_frame": {"bias": {"integer": 1, "period": 4}},
        },
        {"devices": {"bias": {"value": 4, "value_unit": "V"}}},
    ]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    table.index = [8, 2, 5]
    original = table.copy(deep=True)

    flux = flux_from_frame(
        table, device_name="bias", fallback_frame=RecordFluxFrame("V", 2, 8)
    )

    np.testing.assert_allclose(flux, [0.8, 0.5, 0.25])
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("flatten", [False, True])
def test_flux_from_record_read_handles_null_and_populated_devices(
    tmp_path: Path, flatten: bool
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"run_id":"explicit","flux":0.7,"devices":null}\n'
        '{"run_id":"frame","devices":{"bias":{"value":0.9,"value_unit":"V"}},'
        '"flux_frame":{"bias":{"integer":0.1,"period":0.8}}}\n',
        encoding="utf-8",
    )
    table = JsonlRecordTable(path).read(flatten=flatten)
    original = table.copy(deep=True)

    np.testing.assert_allclose(flux_from_frame(table, device_name="bias"), [0.7, 1.0])
    pd.testing.assert_frame_equal(table, original)


def test_flux_row_frame_accepts_numeric_dotted_columns_without_units() -> None:
    table = pd.DataFrame(
        {
            "devices.bias.value": [3, 5],
            "flux_frame.bias.integer": [1, 1],
            "flux_frame.bias.period": [4, 4],
        }
    )

    np.testing.assert_allclose(flux_from_frame(table, device_name="bias"), [0.5, 1.0])


@pytest.mark.parametrize(
    ("present", "missing"),
    [
        ("flux_frame.bias.integer", "flux_frame.bias.period"),
        ("flux_frame.bias.period", "flux_frame.bias.integer"),
    ],
)
@pytest.mark.parametrize("null_column", [False, True])
def test_incomplete_used_row_frame_fails_without_fallback(
    present: str, missing: str, null_column: bool
) -> None:
    table = pd.DataFrame(
        {
            "flux": [0.8, np.nan],
            "devices.bias.value": [np.nan, 3.0],
            "devices.bias.value_unit": [None, "V"],
            present: [np.nan, 1.0],
        },
        index=["explicit", "partial"],
    )
    if null_column:
        table[missing] = np.nan

    with pytest.raises(ValueError, match=f"partial.*{missing}|{missing}.*partial"):
        flux_from_frame(
            table, device_name="bias", fallback_frame=RecordFluxFrame("V", 1, 4)
        )


@pytest.mark.parametrize("flatten", [False, True])
def test_flux_sums_every_row_frame_device_with_signed_period(flatten: bool) -> None:
    rows = [
        {
            "devices": {"bias": {"value": 3}, "coil": {"value": 3}},
            "flux_frame": {
                "bias": {"integer": 1, "period": 4},
                "coil": {"integer": 2, "period": -4},
            },
        },
        {
            "devices": {"coil": {"value": 6}},
            "flux_frame": {"coil": {"integer": 2, "period": -4}},
        },
        {"flux": 0.8, "flux_frame": {"missing": {"integer": "unused"}}},
    ]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    table.index = ["combined", "coil-only", "explicit"]
    original = table.copy(deep=True)
    np.testing.assert_allclose(
        flux_from_frame(table, device_name="bias"), [0.25, -1.0, 0.8]
    )
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("missing", ["value", "integer", "period"])
def test_used_frame_reports_missing_contributor_without_falling_back(
    flatten: bool, missing: str
) -> None:
    coil_frame = {"integer": 2, "period": -4}
    coil_value = {"value": 3, "value_unit": "A"}
    if missing == "value":
        coil_value.pop(missing)
    else:
        coil_frame.pop(missing)
    row = {
        "devices": {"bias": {"value": 3}, "coil": coil_value},
        "flux_frame": {"bias": {"integer": 1, "period": 4}, "coil": coil_frame},
    }
    table = pd.json_normalize([row]) if flatten else pd.DataFrame([row])
    table.index = ["partial"]
    with pytest.raises(ValueError, match=f"partial.*coil.*{missing}"):
        flux_from_frame(
            table, device_name="bias", fallback_frame=RecordFluxFrame("V", 0, 2)
        )


@pytest.mark.parametrize("flatten", [False, True])
def test_multiple_device_fallback_uses_caller_units_and_current_row_values(
    flatten: bool,
) -> None:
    rows = [
        {"flux": 0.8},
        {
            "devices": {"bias": {"value": 5}, "coil": {"value": 3}},
            "flux_frame": {
                "bias": {"integer": 1, "period": 4},
                "coil": {"integer": 2, "period": -4},
            },
        },
        {
            "devices": {
                "bias": {"value": 3, "value_unit": "V"},
                "coil": {"value": 3, "value_unit": "A"},
            }
        },
    ]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    fallback = FluxFrame(
        {"bias": DeviceFluxFrame(1, 4, "V"), "coil": DeviceFluxFrame(2, -4, "A")}
    )
    np.testing.assert_allclose(
        flux_from_frame(table, device_name="bias", fallback_frame=fallback),
        [0.8, 0.75, 0.25],
    )


@pytest.mark.parametrize("unit", ["V", None])
def test_multiple_device_fallback_rejects_missing_or_mismatching_unit(
    unit: str | None,
) -> None:
    table = pd.DataFrame(
        {"devices.coil.value": [3], "devices.coil.value_unit": [unit]},
        index=["old"],
    )
    with pytest.raises(ValueError, match="old.*devices.coil.value_unit"):
        flux_from_frame(
            table,
            device_name="coil",
            fallback_frame=FluxFrame({"coil": DeviceFluxFrame(2, -4, "A")}),
        )


def test_dotted_all_null_device_is_absent_but_any_frame_field_declares_it() -> None:
    table = pd.DataFrame(
        {
            "devices.bias.value": [3, 3],
            "devices.bias.value_unit": ["V", "V"],
            "flux_frame.bias.integer": [1, None],
            "flux_frame.bias.period": [4, None],
            "flux_frame.coil.integer": [None, None],
            "flux_frame.coil.period": [None, None],
            "flux_frame.coil.half": [None, None],
        },
        index=["row-frame", "fallback"],
    )
    np.testing.assert_allclose(
        flux_from_frame(
            table, device_name="bias", fallback_frame=RecordFluxFrame("V", 2, 4)
        ),
        [0.5, 0.25],
    )
    table.loc["row-frame", "flux_frame.coil.half"] = 2
    with pytest.raises(ValueError, match="row-frame.*flux_frame.coil.integer"):
        flux_from_frame(table, device_name="bias")


def test_nested_device_key_with_all_null_fields_is_invalid() -> None:
    table = pd.DataFrame(
        [{"flux_frame": {"coil": {"integer": None, "period": None}}}],
        index=["corrupt"],
    )
    with pytest.raises(ValueError, match="corrupt.*flux_frame.coil.integer"):
        flux_from_frame(table, device_name="coil")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("flux_frame.bias.period", 0),
        ("flux_frame.bias.integer", True),
        ("devices.bias.value", True),
    ],
)
def test_invalid_used_numeric_field_names_row_and_field(
    column: str, value: float
) -> None:
    table = pd.DataFrame(
        {
            "flux_frame.bias.integer": [1],
            "flux_frame.bias.period": [4],
            "devices.bias.value": [3],
        },
        index=["invalid"],
        dtype=object,
    )
    table.loc["invalid", column] = value
    with pytest.raises(ValueError, match="invalid.*bias.*" + column.rpartition(".")[2]):
        flux_from_frame(
            table, device_name="bias", fallback_frame=RecordFluxFrame("V", 0, 4)
        )


def test_missing_frame_and_fallback_reports_row() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        flux_from_frame(
            pd.DataFrame({"devices.bias.value": [3]}, index=["unresolved"]),
            device_name="bias",
        )
