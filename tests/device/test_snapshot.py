"""Historical scalar projection uses the public Info value/unit contract."""

from __future__ import annotations

import pytest
from zcu_tools.device import (
    DeviceScalar,
    FakeDeviceInfo,
    YOKOGS200Info,
    snapshot_scalar_values,
)


def test_snapshot_scalars_preserve_fake_yoko_units_and_omit_missing_names():
    snapshot = {
        "fake": FakeDeviceInfo(address="fake", value=2.0).model_dump(mode="json"),
        "coil": YOKOGS200Info(address="yoko", mode="current", value=0.002).model_dump(
            mode="json"
        ),
        "legacy": {"value": -3.0},
        "irrelevant": {"freq_Hz": 1e9},
    }
    assert snapshot_scalar_values(snapshot, ["fake", "coil", "legacy", "missing"]) == {
        "fake": DeviceScalar(2.0, "native"),
        "coil": DeviceScalar(0.002, "A"),
        "legacy": DeviceScalar(-3.0, "native"),
    }


@pytest.mark.parametrize(
    "info",
    [
        {},
        {"value": True},
        {"value": "1"},
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": 1, "value_unit": ""},
        {"value": 1, "value_unit": None},
    ],
)
def test_present_invalid_scalar_fails_naming_the_device(info):
    with pytest.raises(ValueError, match="bad_coil"):
        snapshot_scalar_values({"bad_coil": info}, ["bad_coil"])
