"""Opt-in projection of named device facts into analysis coordinates."""

from math import isfinite

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.analysis.flux import (
    DevicePeriod,
    FluxFrame,
    flux_from_device_values,
)
from zcu_tools.cfg_model import validate_device_name

from .schema import resolve_record_flux


def project_device_coordinates(
    records: pd.DataFrame, *, device_name: str
) -> pd.DataFrame:
    """Project devices[device_name] into isolated dev_value/dev_unit columns.

    device_name must be nonblank without ASCII dots; valid names are unchanged.
    Invalid names raise ValueError. Rows without explicit finite flux require a
    finite numeric device value; failures name devices[device_name].value and
    the row index. Explicit flux rows may omit device facts, yielding NaN for
    their source dev_value. Units are copied when strings, otherwise None;
    resolve_record_flux checks a unit only when using a fallback frame.
    Other fields, row order and index are preserved, and the input is unchanged.
    Dotted columns devices.<device_name>.value/value_unit are accepted for rows
    without a nested devices object (as in a flattened record CSV).
    No other device or legacy flat coordinate is used as fallback.
    """
    validate_device_name(device_name)
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
    fallback_frame: FluxFrame | None = None,
) -> NDArray[np.float64]:
    """Return each record's normalized flux, preserving row order and input.

    Each nested or dotted row uses explicit finite flux, then flux_frame.offset
    plus devices.<name>.value / flux_frame.period.<name>, then caller fallback
    only when no row frame exists. Dotted membership uses non-null period cells.
    Missing device values contribute zero. Dotted NaN value/period cells are
    absent: flattening loses the distinction between a missing key and an
    explicit NaN. Only nested input guarantees rejection of explicit NaN.
    Present inf, boolean and other invalid values still raise.
    Row frames use native calibration units without checking unit labels.
    Fallback checks value_unit against DevicePeriod.unit for each present value.
    No unit conversion or file reads occur. Flat flux_int/flux_period columns
    are not row frames. Errors name the row and used field; invalid frames do
    not fall back. Explicit flux ignores unused fields. Inputs remain unchanged
    and empty tables return an empty float array.
    """
    values = np.empty(len(table), dtype=np.float64)
    for position, (index, row) in enumerate(table.iterrows()):
        single = table.iloc[position : position + 1]
        if not _absent(row.get("flux")):
            values[position] = resolve_record_flux(
                single.reindex(columns=["flux"])
            ).values[0]
            continue
        try:
            frame = _row_flux_frame(row)
            if frame is not None:
                values[position] = _flux_for_record(row, frame, check_units=False)
            elif isinstance(fallback_frame, FluxFrame):
                values[position] = _flux_for_record(
                    row, fallback_frame, check_units=True
                )
            else:
                raise ValueError("missing flux_frame and fallback frame")
        except ValueError as exc:
            raise ValueError(f"row {index!r}: {exc}") from exc
    return values


def _absent(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))
    )


def _row_flux_frame(row: pd.Series) -> FluxFrame | None:
    raw: object = row.get("flux_frame")
    if isinstance(raw, dict):
        offset: object = raw.get("offset")
        periods: object = raw.get("period")
    else:
        if not _absent(raw):
            raise ValueError("flux_frame must contain offset and period")
        periods = {
            column.removeprefix("flux_frame.period."): value
            for column, value in row.items()
            if isinstance(column, str)
            and column.startswith("flux_frame.period.")
            and not _absent(value)
        }
        offset = row.get("flux_frame.offset")
        if not periods and _absent(offset):
            return None
    if _absent(offset):
        raise ValueError("missing flux_frame.offset")
    if not isinstance(periods, dict):
        raise ValueError("flux_frame.period must be a device mapping")
    return FluxFrame(
        _record_number(offset, "flux_frame.offset"),
        {name: _device_frame(name, period) for name, period in periods.items()},
    )


def _device_frame(name: str, period: object) -> DevicePeriod:
    field = f"flux_frame.period.{name}"
    period_value = _record_number(period, field)
    try:
        return DevicePeriod(period_value)
    except ValueError as exc:
        raise ValueError(f"{field}: {exc}") from exc


def _record_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _flux_for_record(row: pd.Series, frame: FluxFrame, *, check_units: bool) -> float:
    values: dict[str, float] = {}
    for name, device in frame.devices.items():
        devices: object = row.get("devices")
        if isinstance(devices, dict):
            entry: object = devices.get(name)
            if entry is None:
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"devices.{name} must be a value mapping")
            if "value" not in entry:
                continue
            value: object = entry["value"]
            unit: object = entry.get("value_unit")
        else:
            column = f"devices.{name}.value"
            if column not in row:
                continue
            value = row[column]
            if _absent(value):
                continue
            unit = row.get(f"devices.{name}.value_unit")
        values[name] = _record_number(value, f"devices.{name}.value")
        if check_units and unit != device.unit:
            raise ValueError(
                f"devices.{name}.value_unit must match fallback unit {device.unit!r}"
            )
    return flux_from_device_values(frame, values)
