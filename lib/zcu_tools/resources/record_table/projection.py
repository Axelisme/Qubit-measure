"""Opt-in projection of named device facts into analysis coordinates."""

from math import isfinite

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.analysis.flux import (
    DeviceFluxFrame,
    FluxFrame,
    flux_from_device_values,
)
from zcu_tools.cfg_model import validate_device_name

from .schema import RecordFluxFrame, resolve_record_flux


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
    device_name: str,
    fallback_frame: RecordFluxFrame | FluxFrame | None = None,
) -> NDArray[np.float64]:
    """Return each record's normalized flux, preserving row order and input.

    table contains nested records or their dotted-column CSV representation.
    Each row uses explicit finite flux, then its flux_frame device mapping
    (integer/period and devices.<name>.value), then caller fallback only if no
    row frame exists. All row-frame devices contribute with signed periods.
    Row frames do not require units: values already use the calibration units.
    Nested membership follows frame keys. Dotted membership uses any non-null
    flux_frame.<device>.* cell in that row; all-null device fields mean absent.
    Flattening cannot distinguish absent devices from originally all-null ones.

    device_name must be nonblank without ASCII dots and is preserved unchanged.
    It selects ONLY the device used by a RecordFluxFrame fallback.
    A FluxFrame fallback uses all its named devices
    and requires each recorded value_unit to equal its declared unit. Neither
    fallback converts units or reads files. Flat flux_int/flux_period columns
    are not row frames; resolve_record_flux retains its independent v2 API.

    Invalid device_name raises ValueError even for empty/explicit-only tables.
    ValueError names the row and missing/invalid used field, including partial
    frames and missing device values; invalid used frames never fall back.
    Valid explicit flux ignores unused frame/device fields. Empty tables return
    an empty float array. No input table, JSONL or calibration is changed.
    """
    validate_device_name(device_name)
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
                values[position] = _flux_for_record(single, frame, check_units=False)
            elif isinstance(fallback_frame, FluxFrame):
                values[position] = _flux_for_record(
                    single, fallback_frame, check_units=True
                )
            else:
                projected = project_device_coordinates(single, device_name=device_name)
                values[position] = resolve_record_flux(
                    projected.reindex(columns=["dev_value", "dev_unit"]),
                    fallback_frame=fallback_frame,
                ).values[0]
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
        devices = {
            name: _device_frame(name, parameters) for name, parameters in raw.items()
        }
        return FluxFrame(devices)
    if not _absent(raw):
        raise ValueError("flux_frame must be a device mapping")
    # CSV columns are the union of all rows: all-null device fields in this
    # row do not declare a contributor, but any populated field declares one.
    names = {
        column.removeprefix("flux_frame.").rpartition(".")[0]
        for column, value in row.items()
        if isinstance(column, str)
        and column.startswith("flux_frame.")
        and not _absent(value)
    }
    if not names:
        return None
    return FluxFrame(
        {
            name: _device_frame(
                name,
                {
                    "integer": row.get(f"flux_frame.{name}.integer"),
                    "period": row.get(f"flux_frame.{name}.period"),
                },
            )
            for name in sorted(names)
        }
    )


def _device_frame(name: str, parameters: object) -> DeviceFluxFrame:
    if not isinstance(parameters, dict):
        raise ValueError(f"flux_frame.{name} must contain integer and period")
    integer: object = parameters.get("integer")
    period: object = parameters.get("period")
    for field, value in (("integer", integer), ("period", period)):
        if _absent(value):
            raise ValueError(f"missing flux_frame.{name}.{field}")
    integer_value = _record_number(integer, f"flux_frame.{name}.integer")
    period_value = _record_number(period, f"flux_frame.{name}.period")
    # The numerical owner validates finite references and signed periods.
    # Give its diagnostic the record column's full path.
    try:
        return DeviceFluxFrame(integer=integer_value, period=period_value)
    except ValueError as exc:
        raise ValueError(f"flux_frame.{name}.{exc}") from exc


def _record_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a real number")
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc


def _flux_for_record(
    row: pd.DataFrame, frame: FluxFrame, *, check_units: bool
) -> float:
    values: dict[str, float] = {}
    for name, device in frame.devices.items():
        projected = project_device_coordinates(row, device_name=name)
        values[name] = float(projected["dev_value"].iloc[0])
        if check_units and projected["dev_unit"].iloc[0] != device.unit:
            raise ValueError(
                f"devices.{name}.value_unit must match fallback unit {device.unit!r}"
            )
    return flux_from_device_values(frame, values)
