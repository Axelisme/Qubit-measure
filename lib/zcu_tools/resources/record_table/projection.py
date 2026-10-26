"""Opt-in projection of named device facts into analysis coordinates."""

from math import isfinite

import pandas as pd


def project_device_coordinates(
    records: pd.DataFrame, *, device_name: str
) -> pd.DataFrame:
    """Project devices[device_name] into isolated dev_value/dev_unit columns.

    device_name must be nonempty. Rows without explicit finite flux require a
    finite numeric device value; failures name devices[device_name].value and
    the row index. Explicit flux rows may omit device facts, yielding NaN for
    their source dev_value. Units are copied when strings, otherwise None;
    resolve_record_flux checks a unit only when using a fallback frame.
    Other fields, row order and index are preserved, and the input is unchanged.
    No other device or legacy flat coordinate is used as fallback.
    """
    if not device_name.strip():
        raise ValueError("device_name must be nonempty")
    values: list[float] = []
    units: list[str | None] = []
    for index, row in records.iterrows():
        devices: object = row.get("devices")
        device: object = devices.get(device_name) if isinstance(devices, dict) else None
        value: object = device.get("value") if isinstance(device, dict) else None
        unit: object = device.get("value_unit") if isinstance(device, dict) else None
        usable_value = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
        )
        flux: object = row.get("flux")
        explicit_flux = (
            isinstance(flux, (int, float))
            and not isinstance(flux, bool)
            and isfinite(flux)
        )
        if not usable_value and not explicit_flux:
            raise ValueError(
                f"Missing or unusable devices[{device_name!r}].value at row {index!r}"
            )
        values.append(
            float(value)
            if isinstance(value, (int, float)) and usable_value
            else float("nan")
        )
        units.append(unit if isinstance(unit, str) else None)
    projected = records.copy()
    projected["dev_value"] = values
    projected["dev_unit"] = units
    return projected
