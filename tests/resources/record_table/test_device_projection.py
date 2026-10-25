"""Caller-selected Run snapshot coordinates, independent of adapter definitions."""

import pandas as pd
import pytest
from zcu_tools.resources.record_table import project_device_coordinates


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
    "devices",
    [
        None,
        {},
        {"bias": {}},
        {"bias": {"value": True, "value_unit": "A"}},
        {"bias": {"value": float("nan"), "value_unit": "A"}},
        {"bias": {"value": 1, "value_unit": "mA"}},
        {"bias": {"value": 1}},
    ],
)
def test_projection_rejects_missing_or_unusable_selected_device(
    devices: object,
) -> None:
    records = pd.DataFrame({"devices": [devices], "dev_value": [99], "dev_unit": ["A"]})
    with pytest.raises(ValueError, match="bias.*row|row.*bias"):
        project_device_coordinates(records, device_name="bias")
