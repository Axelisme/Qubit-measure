"""Opt-in projection of named device facts into analysis coordinates."""

from math import isfinite

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .schema import RecordFluxFrame, resolve_record_flux


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
    Dotted columns devices.<device_name>.value/value_unit are accepted for rows
    without a nested devices object (as in a flattened record CSV).
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
        if not isinstance(devices, dict):
            value = row.get(f"devices.{device_name}.value")
            unit = row.get(f"devices.{device_name}.value_unit")
        usable_value = (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and isfinite(value)
        )
        flux: object = row.get("flux")
        explicit_flux = (
            isinstance(flux, (int, float, np.integer, np.floating))
            and not isinstance(flux, bool)
            and isfinite(flux)
        )
        if not usable_value and not explicit_flux:
            raise ValueError(
                f"Missing or unusable devices[{device_name!r}].value at row {index!r}"
            )
        values.append(
            float(value)
            if isinstance(value, (int, float, np.integer, np.floating)) and usable_value
            else float("nan")
        )
        units.append(unit if isinstance(unit, str) else None)
    projected = records.copy()
    projected["dev_value"] = values
    projected["dev_unit"] = units
    return projected


def flux_from_frame(
    table: pd.DataFrame,
    *,
    device_name: str,
    fallback_frame: RecordFluxFrame | None = None,
) -> NDArray[np.float64]:
    """Return normalized flux in table row order without modifying table.

    table is a record table or its dotted-column CSV representation. device_name
    selects devices[name].value/value_unit (CSV: devices.<name>.value/value_unit);
    flat dev_value/dev_unit are not input aliases. Finite explicit flux wins,
    otherwise use the row's flux_int/flux_period, then fallback_frame only when
    both row-frame values are absent. Row frames use the selected device value
    without conversion; fallback requires a matching unit. fallback_frame is
    caller-provided data, never a file path.

    ValueError names missing/unusable device fields and rows, or an empty device
    name. RecordTableV2Error names invalid used coordinates, incomplete row-frame
    fields and rows, or rows unresolved without a matching fallback. Explicit
    flux ignores unused device/frame fields. Empty tables return an empty array.
    """
    projected = project_device_coordinates(table, device_name=device_name)
    return resolve_record_flux(projected, fallback_frame=fallback_frame).values
