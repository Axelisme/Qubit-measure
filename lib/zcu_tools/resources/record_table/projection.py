"""Explicit device selection for adapter-owned records and flat analysis frames."""

from math import isfinite

import pandas as pd


def project_device_coordinates(
    records: pd.DataFrame, *, device_name: str
) -> pd.DataFrame:
    """Project a named Run device into an isolated flat v2 analysis frame.

    Each record must contain devices[device_name] with numeric value and
    value_unit (A, V or native), as captured by RunMetadata. Missing facts,
    nonfinite values and unsupported units raise ValueError naming the row.
    device_name must be nonempty. No other device or legacy flat coordinate is
    used as fallback. Other record fields and row order/index are preserved.
    """
    if not device_name.strip():
        raise ValueError("device_name must be nonempty")
    if not records.empty and "devices" not in records:
        raise ValueError(
            f"Missing devices for {device_name!r} at row {records.index[0]!r}"
        )
    values: list[int | float] = []
    units: list[str] = []
    snapshots = records["devices"] if "devices" in records else pd.Series(dtype=object)
    for index, devices in snapshots.items():
        if not isinstance(devices, dict):
            raise ValueError(f"Missing device {device_name!r} at row {index!r}")
        device: object = devices.get(device_name)
        if not isinstance(device, dict):
            raise ValueError(f"Missing device {device_name!r} at row {index!r}")
        value: object = device.get("value")
        unit: object = device.get("value_unit")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or not isinstance(unit, str)
            or unit not in ("A", "V", "native")
        ):
            raise ValueError(
                f"Unusable coordinate for {device_name!r} at row {index!r}"
            )
        values.append(value)
        units.append(unit)
    projected = records.copy()
    projected["dev_value"] = values
    projected["dev_unit"] = units
    return projected
