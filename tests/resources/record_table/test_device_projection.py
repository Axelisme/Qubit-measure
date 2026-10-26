"""Caller-selected Run snapshot coordinates, independent of adapter definitions."""

import pandas as pd
import pytest
from zcu_tools.resources.record_table import project_device_coordinates


@pytest.mark.parametrize("name", ["bias.local", "", " \t "])
def test_projection_rejects_invalid_device_names_even_with_explicit_flux(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="(?i)device.name"):
        project_device_coordinates(pd.DataFrame({"flux": [0.5]}), device_name=name)


@pytest.mark.parametrize("name", [" Bias ", "bias", "BIAS", "偏壓", "bias．local"])
@pytest.mark.parametrize("flatten", [False, True])
def test_projection_uses_valid_device_names_unchanged(name: str, flatten: bool) -> None:
    rows = [{"devices": {name: {"value": 3, "value_unit": "V"}}}]
    table = pd.json_normalize(rows) if flatten else pd.DataFrame(rows)
    projected = project_device_coordinates(table, device_name=name)
    assert projected["dev_value"].tolist() == [3]
    assert projected["dev_unit"].tolist() == ["V"]


def test_projection_selects_the_named_device_and_preserves_records() -> None:
    records = pd.DataFrame(
        {
            "devices": [
                {
                    "other": {"value": 99, "value_unit": "A"},
                    "bias": {"value": -0.25, "value_unit": "V"},
                },
                {"bias": {"value": 0.75, "value_unit": "V"}},
            ],
            "freq_mhz": [1000.0, 2000.0],
        },
        index=[7, 3],
    )
    projected = project_device_coordinates(records, device_name="bias")
    assert projected["dev_value"].tolist() == [-0.25, 0.75]
    assert projected["dev_unit"].tolist() == ["V", "V"]
    assert projected.index.tolist() == [7, 3]
    assert projected["freq_mhz"].tolist() == [1000.0, 2000.0]
    assert projected["devices"].tolist() == records["devices"].tolist()
    assert records.columns.tolist() == ["devices", "freq_mhz"]


@pytest.mark.parametrize(
    ("record", "expected_value", "expected_unit"),
    [
        ({"flux": 0.5, "devices": None}, None, None),
        (
            {"devices": {"bias": {"value": 1}}, "flux_int": 0, "flux_period": 2},
            1,
            None,
        ),
        ({"devices": {"bias": {"value": 1, "value_unit": "raw"}}}, 1, "raw"),
    ],
)
def test_projection_checks_only_device_facts_needed_for_flux(
    record: dict[str, object], expected_value: float | None, expected_unit: str | None
) -> None:
    projected = project_device_coordinates(pd.DataFrame([record]), device_name="bias")

    if expected_value is None:
        assert pd.isna(projected.loc[0, "dev_value"])
    else:
        assert projected.loc[0, "dev_value"] == expected_value
    assert projected.loc[0, "dev_unit"] == expected_unit


@pytest.mark.parametrize(
    "devices",
    [
        None,
        {},
        {"bias": {}},
        {"bias": {"value": True, "value_unit": "A"}},
        {"bias": {"value": float("nan"), "value_unit": "A"}},
    ],
)
def test_projection_rejects_missing_or_unusable_selected_device(
    devices: object,
) -> None:
    records = pd.DataFrame({"devices": [devices], "dev_value": [99], "dev_unit": ["A"]})
    with pytest.raises(ValueError, match="bias.*row|row.*bias"):
        project_device_coordinates(records, device_name="bias")
