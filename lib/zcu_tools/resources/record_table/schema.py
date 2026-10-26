"""Flat RecordTable v2 coordinate contract: schema, validation, resolution, migration.

Producers and consumers that opt into the v2 contract share the coordinate
names, affine ``RecordFluxFrame``,
``validate_record_table_v2`` and ``resolve_record_flux`` from this module. The
explicit legacy-unit migration seam (``migrate_record_table_v2`` with
``LegacyRecordFluxFrame``) is the only supported path for legacy A/mA/V/mV tables;
there is no legacy aliasing, magnitude inference or per-row source-unit guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DeviceValueUnit = Literal["A", "V", "native"]
LegacyDeviceValueUnit = Literal["A", "mA", "V", "mV"]

FLUX_COLUMN = "flux"
DEV_VALUE_COLUMN = "dev_value"
DEV_UNIT_COLUMN = "dev_unit"
FLUX_INT_COLUMN = "flux_int"
FLUX_PERIOD_COLUMN = "flux_period"
RECORD_COORDINATE_COLUMNS = (
    FLUX_COLUMN,
    DEV_VALUE_COLUMN,
    DEV_UNIT_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
)

_DEVICE_VALUE_UNITS = frozenset({"A", "V", "native"})
_LEGACY_DEVICE_VALUE_UNITS = frozenset({"A", "mA", "V", "mV"})
# Legacy coordinate aliases are not accepted v2 columns: a table that still
# carries them is stale and must fail before mutation/analysis.
_LEGACY_COORDINATE_COLUMN_ALIASES = frozenset(
    {"calibrated mA", "calibrated A", "Flux", "flux_bias"}
)
_LEGACY_UNIT_SCALE = {"A": 1.0, "mA": 1.0e-3, "V": 1.0, "mV": 1.0e-3}
_NON_REAL_COORDINATE_TYPES = (
    bool,
    complex,
    date,
    timedelta,
    np.bool_,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
)


def _is_non_real_coordinate(value: object) -> bool:
    return isinstance(value, _NON_REAL_COORDINATE_TYPES)


class RecordTableV2Error(ValueError):
    """Expected RecordTable v2 contract violation."""

    def __init__(
        self, message: str, *, reason: str, data: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.data = data or {}


def _physical_kind(unit: str) -> str:
    return "current" if unit in ("A", "mA") else "voltage"


def _base_unit_of(unit: str) -> str:
    return "A" if _physical_kind(unit) == "current" else "V"


def _require_real_frame_value(value: object, *, field: str) -> None:
    if _is_non_real_coordinate(value):
        raise RecordTableV2Error(
            f"{field} must be a real numeric value, got {value!r}",
            reason="non_numeric_coordinate",
            data={"field": field},
        )


def _coerce_real_numeric_input(value: object, *, field: str) -> NDArray[np.float64]:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecordTableV2Error(
            f"{field} must contain real numeric values: {exc}",
            reason="non_numeric_coordinate",
            data={"field": field},
        ) from exc

    non_real_indexes = [
        index for index, item in enumerate(source.flat) if _is_non_real_coordinate(item)
    ]
    if non_real_indexes:
        raise RecordTableV2Error(
            f"{field} must contain real numeric values; found non-real coordinate "
            f"type at flattened index(es) {non_real_indexes}",
            reason="non_numeric_coordinate",
            data={"field": field, "indexes": non_real_indexes},
        )

    try:
        return np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecordTableV2Error(
            f"{field} must contain real numeric values: {exc}",
            reason="non_numeric_coordinate",
            data={"field": field},
        ) from exc


@dataclass(frozen=True, slots=True)
class RecordFluxFrame:
    """Device-value affine frame mapping A/V/native ``dev_value`` to flux.

    dev_unit declares the raw value unit (A, V or native); flux_int is the
    zero-flux device value and flux_period is the positive device-value span
    per flux quantum. Native values have no implied physical unit conversion."""

    dev_unit: DeviceValueUnit
    flux_int: float
    flux_period: float

    def __post_init__(self) -> None:
        if self.dev_unit not in _DEVICE_VALUE_UNITS:
            raise RecordTableV2Error(
                f"unsupported device unit {self.dev_unit!r}; expected 'A', 'V' or 'native'",
                reason="invalid_unit",
                data={"unit": self.dev_unit},
            )
        _require_real_frame_value(self.flux_int, field="flux_int")
        _require_real_frame_value(self.flux_period, field="flux_period")
        if not np.isfinite(self.flux_int):
            raise RecordTableV2Error(
                f"flux_int must be finite, got {self.flux_int!r}",
                reason="non_finite_value",
                data={"field": "flux_int"},
            )
        if not np.isfinite(self.flux_period) or self.flux_period <= 0:
            raise RecordTableV2Error(
                f"flux_period must be finite and > 0, got {self.flux_period!r}",
                reason="invalid_period",
                data={"field": "flux_period"},
            )

    @overload
    def flux_from_dev_value(self, value: float) -> float: ...

    @overload
    def flux_from_dev_value(
        self, value: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def flux_from_dev_value(
        self, value: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        arr = _coerce_real_numeric_input(value, field="value")
        result = (arr - self.flux_int) / self.flux_period
        if arr.ndim == 0:
            return float(result)
        return result

    @overload
    def dev_value_from_flux(self, flux: float) -> float: ...

    @overload
    def dev_value_from_flux(self, flux: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def dev_value_from_flux(
        self, flux: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        arr = _coerce_real_numeric_input(flux, field="flux")
        result = arr * self.flux_period + self.flux_int
        if arr.ndim == 0:
            return float(result)
        return result


RecordFluxSource = Literal["explicit", "row-frame", "fallback-frame"]


@dataclass(frozen=True, slots=True)
class RecordFluxResolution:
    """Per-row resolved flux with closed provenance (single source of truth)."""

    values: NDArray[np.float64]
    sources: tuple[RecordFluxSource, ...]

    @property
    def explicit_mask(self) -> NDArray[np.bool_]:
        return np.asarray(
            [source == "explicit" for source in self.sources], dtype=np.bool_
        )


def _coerce_numeric_column(records: pd.DataFrame, column: str) -> NDArray[np.float64]:
    source = records[column]
    non_real_indexes = [
        index
        for index, value in enumerate(source.array)
        if _is_non_real_coordinate(value)
    ]
    if non_real_indexes:
        raise RecordTableV2Error(
            f"column {column!r} must contain real numeric values; found non-real "
            f"coordinate type at row(s) {non_real_indexes}",
            reason="non_numeric_coordinate",
            data={"column": column, "indexes": non_real_indexes},
        )
    try:
        # pandas' untyped converter also accepts scalars and arrays; this boundary
        # always consumes a column and retains its Series representation.
        series = pd.Series(pd.to_numeric(source, errors="raise"))
        return series.to_numpy(dtype=np.dtype(np.float64))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecordTableV2Error(
            f"column {column!r} must be numeric: {exc}",
            reason="non_numeric_coordinate",
            data={"column": column},
        ) from exc


def _require_finite_non_null(values: NDArray[np.float64], *, column: str) -> None:
    null_indexes = np.flatnonzero(np.isnan(values)).tolist()
    if null_indexes:
        raise RecordTableV2Error(
            f"column {column!r} must be non-null for row(s) {null_indexes}",
            reason="null_required_value",
            data={"column": column, "indexes": null_indexes},
        )
    nonfinite_indexes = np.flatnonzero(~np.isfinite(values)).tolist()
    if nonfinite_indexes:
        raise RecordTableV2Error(
            f"column {column!r} must be finite for row(s) {nonfinite_indexes}",
            reason="non_finite_value",
            data={"column": column, "indexes": nonfinite_indexes},
        )


def _require_finite_where_present(values: NDArray[np.float64], *, column: str) -> None:
    nonfinite_indexes = np.flatnonzero(
        ~np.isfinite(values) & ~np.isnan(values)
    ).tolist()
    if nonfinite_indexes:
        raise RecordTableV2Error(
            f"column {column!r} must be finite or null for row(s) {nonfinite_indexes}",
            reason="non_finite_value",
            data={"column": column, "indexes": nonfinite_indexes},
        )


def validate_record_table_v2(
    records: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> None:
    """Fast-fail validation of the flat v2 coordinate contract.

    ``dev_value`` / ``dev_unit`` are required; ``flux``, ``flux_int`` and
    ``flux_period`` are optional. ``allow_empty=True`` accepts only a completely
    empty table or a zero-row table with valid v2 headers; tables that still
    carry legacy/stale coordinate columns always fail.
    """
    # tolerated: Reject non-DataFrame runtime inputs with invalid_input before column checks.
    if not isinstance(records, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise RecordTableV2Error(
            "records must be a pandas DataFrame", reason="invalid_input"
        )
    if records.columns.duplicated().any():
        dupes = sorted(records.columns[records.columns.duplicated()].unique().tolist())
        raise RecordTableV2Error(
            f"duplicate column name(s): {dupes}",
            reason="duplicate_columns",
            data={"columns": dupes},
        )
    stale = sorted(set(records.columns) & _LEGACY_COORDINATE_COLUMN_ALIASES)
    if stale:
        raise RecordTableV2Error(
            f"stale legacy coordinate column(s) not allowed in v2: {stale}",
            reason="legacy_coordinate_column",
            data={"columns": stale},
        )
    if len(records.index) == 0 and len(records.columns) == 0:
        if allow_empty:
            return
        raise RecordTableV2Error(
            "record table is empty; pass allow_empty=True to accept an empty v2 table",
            reason="empty_table",
        )
    missing = [
        column
        for column in (DEV_VALUE_COLUMN, DEV_UNIT_COLUMN)
        if column not in records.columns
    ]
    if missing:
        raise RecordTableV2Error(
            f"missing required v2 coordinate column(s): {missing}",
            reason="missing_required_column",
            data={"columns": missing},
        )
    has_int = FLUX_INT_COLUMN in records.columns
    has_period = FLUX_PERIOD_COLUMN in records.columns
    if has_int != has_period:
        orphan = FLUX_INT_COLUMN if has_int else FLUX_PERIOD_COLUMN
        raise RecordTableV2Error(
            f"orphan frame column {orphan!r}: flux_int and flux_period must "
            "appear together",
            reason="orphan_frame_column",
            data={"column": orphan},
        )
    if records.empty:
        if allow_empty:
            return
        raise RecordTableV2Error(
            "record table is empty (zero rows); pass allow_empty=True to accept "
            "a header-only v2 table",
            reason="empty_table",
        )
    dev_values = _coerce_numeric_column(records, DEV_VALUE_COLUMN)
    _require_finite_non_null(dev_values, column=DEV_VALUE_COLUMN)
    _require_device_units(records)
    if FLUX_COLUMN in records.columns:
        _require_finite_where_present(
            _coerce_numeric_column(records, FLUX_COLUMN), column=FLUX_COLUMN
        )
    if has_int:
        _require_row_frames(records)


def _require_device_units(records: pd.DataFrame) -> None:
    """Require a non-null supported unit for every device-value row."""
    dev_units = records[DEV_UNIT_COLUMN]
    null_unit_indexes = np.flatnonzero(dev_units.isna().to_numpy()).tolist()
    if null_unit_indexes:
        raise RecordTableV2Error(
            f"column {DEV_UNIT_COLUMN!r} must be non-null for row(s) "
            f"{null_unit_indexes}",
            reason="null_required_value",
            data={"column": DEV_UNIT_COLUMN, "indexes": null_unit_indexes},
        )
    invalid_units = sorted(
        {str(unit) for unit in dev_units.unique()} - _DEVICE_VALUE_UNITS
    )
    if invalid_units:
        raise RecordTableV2Error(
            f"dev_unit must be 'A', 'V' or 'native', found: {invalid_units}",
            reason="invalid_unit",
            data={"units": invalid_units},
        )


def _require_row_frames(records: pd.DataFrame) -> None:
    """Require paired finite-or-null frame values and positive present periods."""
    int_values = _coerce_numeric_column(records, FLUX_INT_COLUMN)
    period_values = _coerce_numeric_column(records, FLUX_PERIOD_COLUMN)
    _require_finite_where_present(int_values, column=FLUX_INT_COLUMN)
    _require_finite_where_present(period_values, column=FLUX_PERIOD_COLUMN)
    int_null = np.isnan(int_values)
    period_null = np.isnan(period_values)
    mismatched = np.flatnonzero(int_null != period_null).tolist()
    if mismatched:
        raise RecordTableV2Error(
            f"row(s) {mismatched} must have flux_int and flux_period either "
            "both finite or both null",
            reason="orphan_frame_column",
            data={"indexes": mismatched},
        )
    nonpositive = np.flatnonzero(
        ~np.isnan(period_values) & (period_values <= 0)
    ).tolist()
    if nonpositive:
        raise RecordTableV2Error(
            f"flux_period must be > 0 for row(s) {nonpositive}",
            reason="invalid_period",
            data={"indexes": nonpositive},
        )


def _optional_row_coordinate(row: pd.DataFrame, column: str) -> float | None:
    """Read one numeric cell only when its coordinate branch needs it."""
    if column not in row:
        return None
    value = float(_coerce_numeric_column(row, column)[0])
    if np.isnan(value):
        return None
    if not np.isfinite(value):
        raise RecordTableV2Error(
            f"{column} must be finite at row(s) {row.index.tolist()}",
            reason="non_finite_coordinate",
            data={"column": column},
        )
    return value


def resolve_record_flux(
    records: pd.DataFrame,
    *,
    fallback_frame: RecordFluxFrame | None = None,
) -> RecordFluxResolution:
    """Resolve flux using explicit -> row-frame -> fallback-frame precedence.

    Only coordinates used by each row are checked. Explicit flux needs no
    device facts or row frame. A complete row frame needs finite dev_value and
    positive flux_period, but no unit conversion. Only one non-null row-frame
    value raises an error naming the row and missing field; it cannot use
    fallback. Fallback needs dev_value and dev_unit matching fallback_frame. Unknown and legacy fields are ignored,
    never aliased. Missing/unusable coordinates raise RecordTableV2Error naming
    fields and row indexes. The input table is not modified; empty tables resolve
    to empty values and provenance.
    """
    values = np.empty(len(records), dtype=np.float64)
    sources: list[RecordFluxSource] = []
    unresolved: list[int] = []
    for index in range(len(records)):
        row = records.iloc[index : index + 1]
        flux = _optional_row_coordinate(row, FLUX_COLUMN)
        if flux is not None:
            values[index] = flux
            sources.append("explicit")
            continue
        has_int = FLUX_INT_COLUMN in row and not row[FLUX_INT_COLUMN].isna().iloc[0]
        has_period = (
            FLUX_PERIOD_COLUMN in row and not row[FLUX_PERIOD_COLUMN].isna().iloc[0]
        )
        if has_int != has_period:
            missing = FLUX_PERIOD_COLUMN if has_int else FLUX_INT_COLUMN
            raise RecordTableV2Error(
                f"missing {missing} in row frame at row(s) {row.index.tolist()}",
                reason="incomplete_row_frame",
                data={"column": missing, "indexes": row.index.tolist()},
            )
        flux_int = _optional_row_coordinate(row, FLUX_INT_COLUMN)
        flux_period = _optional_row_coordinate(row, FLUX_PERIOD_COLUMN)
        dev_value = _optional_row_coordinate(row, DEV_VALUE_COLUMN)
        if flux_int is not None and flux_period is not None and dev_value is not None:
            if flux_period <= 0:
                raise RecordTableV2Error(
                    f"flux_period must be positive at row(s) {row.index.tolist()}",
                    reason="invalid_flux_period",
                )
            values[index] = (dev_value - flux_int) / flux_period
            sources.append("row-frame")
            continue
        if (
            fallback_frame is not None
            and DEV_UNIT_COLUMN in row
            and row[DEV_UNIT_COLUMN].iloc[0] == fallback_frame.dev_unit
            and dev_value is not None
        ):
            values[index] = fallback_frame.flux_from_dev_value(dev_value)
            sources.append("fallback-frame")
            continue
        unresolved.append(index)
    if unresolved:
        raise RecordTableV2Error(
            f"unable to resolve flux for row(s) {unresolved}: need flux, or "
            "dev_value with flux_int/flux_period, or dev_value/dev_unit "
            "with a matching fallback frame",
            reason="unresolved_flux",
            data={"indexes": unresolved},
        )
    return RecordFluxResolution(values=values, sources=tuple(sources))


@dataclass(frozen=True, slots=True)
class LegacyRecordFluxFrame:
    """Legacy migration frame: values use the declared legacy unit."""

    dev_unit: LegacyDeviceValueUnit
    flux_int: float
    flux_period: float

    def __post_init__(self) -> None:
        if self.dev_unit not in _LEGACY_DEVICE_VALUE_UNITS:
            raise RecordTableV2Error(
                f"unsupported legacy unit {self.dev_unit!r}; expected one of "
                f"{sorted(_LEGACY_DEVICE_VALUE_UNITS)}",
                reason="invalid_unit",
                data={"unit": self.dev_unit},
            )
        _require_real_frame_value(self.flux_int, field="flux_int")
        _require_real_frame_value(self.flux_period, field="flux_period")
        if not np.isfinite(self.flux_int):
            raise RecordTableV2Error(
                f"flux_int must be finite, got {self.flux_int!r}",
                reason="non_finite_value",
                data={"field": "flux_int"},
            )
        if not np.isfinite(self.flux_period) or self.flux_period <= 0:
            raise RecordTableV2Error(
                f"flux_period must be finite and > 0, got {self.flux_period!r}",
                reason="invalid_period",
                data={"field": "flux_period"},
            )


def _require_matching_physical_kind(dev_value_unit: str, frame_unit: str) -> None:
    if _physical_kind(dev_value_unit) != _physical_kind(frame_unit):
        raise RecordTableV2Error(
            f"flux frame unit {frame_unit!r} does not match device value "
            f"physical kind of {dev_value_unit!r}",
            reason="frame_kind_mismatch",
            data={"dev_unit": dev_value_unit, "frame_unit": frame_unit},
        )


def migrate_record_table_v2(
    records: pd.DataFrame,
    *,
    dev_value_column: str,
    dev_value_unit: LegacyDeviceValueUnit,
    flux_column: str | None = None,
    flux_frame: LegacyRecordFluxFrame | None = None,
) -> pd.DataFrame:
    """Convert a legacy user CSV to a flat v2 table under one declared source unit.

    The single scalar ``dev_value_unit`` applies to the whole ``dev_value_column``:
    A/mA/V/mV closed literals only, A/V pass through and mA/mV divide by 1000.
    Frame values normalize to base A/V units; the frame physical kind must match
    the device value kind. The input DataFrame is never mutated.
    """
    # tolerated: Reject non-DataFrame runtime inputs with invalid_input before unit or column checks.
    if not isinstance(records, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise RecordTableV2Error(
            "records must be a pandas DataFrame", reason="invalid_input"
        )
    if dev_value_unit not in _LEGACY_DEVICE_VALUE_UNITS:
        raise RecordTableV2Error(
            f"unsupported source unit {dev_value_unit!r}; expected one of "
            f"{sorted(_LEGACY_DEVICE_VALUE_UNITS)}",
            reason="invalid_unit",
            data={"unit": dev_value_unit},
        )
    present_v2 = [c for c in RECORD_COORDINATE_COLUMNS if c in records.columns]
    if present_v2:
        raise RecordTableV2Error(
            f"source already contains v2 coordinate column(s): {present_v2}",
            reason="source_has_v2_columns",
            data={"columns": present_v2},
        )
    if dev_value_column not in records.columns:
        raise RecordTableV2Error(
            f"source column {dev_value_column!r} not found",
            reason="missing_source_column",
            data={"column": dev_value_column},
        )
    if flux_column is not None and flux_column not in records.columns:
        raise RecordTableV2Error(
            f"source column {flux_column!r} not found",
            reason="missing_source_column",
            data={"column": flux_column},
        )
    if flux_column == dev_value_column:
        raise RecordTableV2Error(
            "flux_column must differ from dev_value_column",
            reason="invalid_mapping",
            data={"column": dev_value_column},
        )
    if flux_frame is not None:
        _require_matching_physical_kind(dev_value_unit, flux_frame.dev_unit)
    scale = _LEGACY_UNIT_SCALE[dev_value_unit]
    base_unit = _base_unit_of(dev_value_unit)
    dev_values = _coerce_numeric_column(records, dev_value_column)
    _require_finite_non_null(dev_values, column=dev_value_column)
    output = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: dev_values * scale,
            DEV_UNIT_COLUMN: base_unit,
        }
    )
    if flux_column is not None:
        output[FLUX_COLUMN] = records[flux_column].to_numpy()
    if flux_frame is not None:
        frame_scale = _LEGACY_UNIT_SCALE[flux_frame.dev_unit]
        output[FLUX_INT_COLUMN] = flux_frame.flux_int * frame_scale
        output[FLUX_PERIOD_COLUMN] = flux_frame.flux_period * frame_scale
    mapped = {dev_value_column, flux_column}
    for column in records.columns:
        if column in mapped:
            continue
        output[column] = records[column].to_numpy()
    coordinate = [c for c in RECORD_COORDINATE_COLUMNS if c in output.columns]
    measurement = [c for c in records.columns if c not in mapped]
    output = output.reindex(columns=coordinate + measurement)
    validate_record_table_v2(output, allow_empty=True)
    return output
