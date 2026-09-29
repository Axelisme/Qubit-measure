"""Flat SampleTable v2 coordinate contract: schema, validation, resolution, migration.

``SampleTable`` (``table.py``) stays schema-free; producers and consumers that opt
into the v2 contract share the coordinate names, affine ``SampleFluxFrame``,
``validate_sample_table_v2`` and ``resolve_sample_flux`` from this module. The
explicit legacy-unit migration seam (``migrate_sample_table_v2`` with
``LegacySampleFluxFrame``) is the only supported path for legacy A/mA/V/mV tables;
there is no legacy aliasing, magnitude inference or per-row source-unit guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

DeviceValueUnit = Literal["A", "V"]
LegacyDeviceValueUnit = Literal["A", "mA", "V", "mV"]

FLUX_COLUMN = "flux"
DEV_VALUE_COLUMN = "dev_value"
DEV_UNIT_COLUMN = "dev_unit"
FLUX_INT_COLUMN = "flux_int"
FLUX_PERIOD_COLUMN = "flux_period"
SAMPLE_COORDINATE_COLUMNS = (
    FLUX_COLUMN,
    DEV_VALUE_COLUMN,
    DEV_UNIT_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
)

_DEVICE_VALUE_UNITS = frozenset({"A", "V"})
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


class SampleTableV2Error(ValueError):
    """Expected SampleTable v2 contract violation."""

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
        raise SampleTableV2Error(
            f"{field} must be a real numeric value, got {value!r}",
            reason="non_numeric_coordinate",
            data={"field": field},
        )


def _coerce_real_numeric_input(value: object, *, field: str) -> NDArray[np.float64]:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SampleTableV2Error(
            f"{field} must contain real numeric values: {exc}",
            reason="non_numeric_coordinate",
            data={"field": field},
        ) from exc

    non_real_indexes = [
        index for index, item in enumerate(source.flat) if _is_non_real_coordinate(item)
    ]
    if non_real_indexes:
        raise SampleTableV2Error(
            f"{field} must contain real numeric values; found non-real coordinate "
            f"type at flattened index(es) {non_real_indexes}",
            reason="non_numeric_coordinate",
            data={"field": field, "indexes": non_real_indexes},
        )

    try:
        return np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SampleTableV2Error(
            f"{field} must contain real numeric values: {exc}",
            reason="non_numeric_coordinate",
            data={"field": field},
        ) from exc


@dataclass(frozen=True, slots=True)
class SampleFluxFrame:
    """Device-value affine frame mapping base-SI ``dev_value`` to dimensionless flux."""

    dev_unit: DeviceValueUnit
    flux_int: float
    flux_period: float

    def __post_init__(self) -> None:
        if self.dev_unit not in _DEVICE_VALUE_UNITS:
            raise SampleTableV2Error(
                f"unsupported device unit {self.dev_unit!r}; expected 'A' or 'V'",
                reason="invalid_unit",
                data={"unit": self.dev_unit},
            )
        _require_real_frame_value(self.flux_int, field="flux_int")
        _require_real_frame_value(self.flux_period, field="flux_period")
        if not np.isfinite(self.flux_int):
            raise SampleTableV2Error(
                f"flux_int must be finite, got {self.flux_int!r}",
                reason="non_finite_value",
                data={"field": "flux_int"},
            )
        if not np.isfinite(self.flux_period) or self.flux_period <= 0:
            raise SampleTableV2Error(
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


SampleFluxSource = Literal["explicit", "row-frame", "fallback-frame"]


@dataclass(frozen=True, slots=True)
class SampleFluxResolution:
    """Per-row resolved flux with closed provenance (single source of truth)."""

    values: NDArray[np.float64]
    sources: tuple[SampleFluxSource, ...]

    @property
    def explicit_mask(self) -> NDArray[np.bool_]:
        return np.asarray(
            [source == "explicit" for source in self.sources], dtype=np.bool_
        )


def _coerce_numeric_column(samples: pd.DataFrame, column: str) -> NDArray[np.float64]:
    source = samples[column]
    non_real_indexes = [
        index
        for index, value in enumerate(source.array)
        if _is_non_real_coordinate(value)
    ]
    if non_real_indexes:
        raise SampleTableV2Error(
            f"column {column!r} must contain real numeric values; found non-real "
            f"coordinate type at row(s) {non_real_indexes}",
            reason="non_numeric_coordinate",
            data={"column": column, "indexes": non_real_indexes},
        )
    try:
        series = pd.to_numeric(source, errors="raise")
        return series.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SampleTableV2Error(
            f"column {column!r} must be numeric: {exc}",
            reason="non_numeric_coordinate",
            data={"column": column},
        ) from exc


def _require_finite_non_null(values: NDArray[np.float64], *, column: str) -> None:
    null_indexes = np.flatnonzero(np.isnan(values)).tolist()
    if null_indexes:
        raise SampleTableV2Error(
            f"column {column!r} must be non-null for row(s) {null_indexes}",
            reason="null_required_value",
            data={"column": column, "indexes": null_indexes},
        )
    nonfinite_indexes = np.flatnonzero(~np.isfinite(values)).tolist()
    if nonfinite_indexes:
        raise SampleTableV2Error(
            f"column {column!r} must be finite for row(s) {nonfinite_indexes}",
            reason="non_finite_value",
            data={"column": column, "indexes": nonfinite_indexes},
        )


def _require_finite_where_present(values: NDArray[np.float64], *, column: str) -> None:
    nonfinite_indexes = np.flatnonzero(
        ~np.isfinite(values) & ~np.isnan(values)
    ).tolist()
    if nonfinite_indexes:
        raise SampleTableV2Error(
            f"column {column!r} must be finite or null for row(s) {nonfinite_indexes}",
            reason="non_finite_value",
            data={"column": column, "indexes": nonfinite_indexes},
        )


def validate_sample_table_v2(
    samples: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> None:
    """Fast-fail validation of the flat v2 coordinate contract.

    ``dev_value`` / ``dev_unit`` are required; ``flux``, ``flux_int`` and
    ``flux_period`` are optional. ``allow_empty=True`` accepts only a completely
    empty table or a zero-row table with valid v2 headers; tables that still
    carry legacy/stale coordinate columns always fail.
    """
    if not isinstance(samples, pd.DataFrame):
        raise SampleTableV2Error(
            "samples must be a pandas DataFrame", reason="invalid_input"
        )
    if samples.columns.duplicated().any():
        dupes = sorted(samples.columns[samples.columns.duplicated()].unique().tolist())
        raise SampleTableV2Error(
            f"duplicate column name(s): {dupes}",
            reason="duplicate_columns",
            data={"columns": dupes},
        )
    stale = sorted(set(samples.columns) & _LEGACY_COORDINATE_COLUMN_ALIASES)
    if stale:
        raise SampleTableV2Error(
            f"stale legacy coordinate column(s) not allowed in v2: {stale}",
            reason="legacy_coordinate_column",
            data={"columns": stale},
        )
    if len(samples.index) == 0 and len(samples.columns) == 0:
        if allow_empty:
            return
        raise SampleTableV2Error(
            "sample table is empty; pass allow_empty=True to accept an empty v2 table",
            reason="empty_table",
        )
    missing = [
        column
        for column in (DEV_VALUE_COLUMN, DEV_UNIT_COLUMN)
        if column not in samples.columns
    ]
    if missing:
        raise SampleTableV2Error(
            f"missing required v2 coordinate column(s): {missing}",
            reason="missing_required_column",
            data={"columns": missing},
        )
    has_int = FLUX_INT_COLUMN in samples.columns
    has_period = FLUX_PERIOD_COLUMN in samples.columns
    if has_int != has_period:
        orphan = FLUX_INT_COLUMN if has_int else FLUX_PERIOD_COLUMN
        raise SampleTableV2Error(
            f"orphan frame column {orphan!r}: flux_int and flux_period must "
            "appear together",
            reason="orphan_frame_column",
            data={"column": orphan},
        )
    if samples.empty:
        if allow_empty:
            return
        raise SampleTableV2Error(
            "sample table is empty (zero rows); pass allow_empty=True to accept "
            "a header-only v2 table",
            reason="empty_table",
        )
    dev_values = _coerce_numeric_column(samples, DEV_VALUE_COLUMN)
    _require_finite_non_null(dev_values, column=DEV_VALUE_COLUMN)
    dev_units = samples[DEV_UNIT_COLUMN]
    null_unit_indexes = np.flatnonzero(dev_units.isna().to_numpy()).tolist()
    if null_unit_indexes:
        raise SampleTableV2Error(
            f"column {DEV_UNIT_COLUMN!r} must be non-null for row(s) "
            f"{null_unit_indexes}",
            reason="null_required_value",
            data={"column": DEV_UNIT_COLUMN, "indexes": null_unit_indexes},
        )
    invalid_units = sorted(
        {str(unit) for unit in dev_units.unique()} - _DEVICE_VALUE_UNITS
    )
    if invalid_units:
        raise SampleTableV2Error(
            f"dev_unit must be 'A' or 'V', found: {invalid_units}",
            reason="invalid_unit",
            data={"units": invalid_units},
        )
    if FLUX_COLUMN in samples.columns:
        _require_finite_where_present(
            _coerce_numeric_column(samples, FLUX_COLUMN), column=FLUX_COLUMN
        )
    if has_int:
        int_values = _coerce_numeric_column(samples, FLUX_INT_COLUMN)
        period_values = _coerce_numeric_column(samples, FLUX_PERIOD_COLUMN)
        _require_finite_where_present(int_values, column=FLUX_INT_COLUMN)
        _require_finite_where_present(period_values, column=FLUX_PERIOD_COLUMN)
        int_null = np.isnan(int_values)
        period_null = np.isnan(period_values)
        mismatched = np.flatnonzero(int_null != period_null).tolist()
        if mismatched:
            raise SampleTableV2Error(
                f"row(s) {mismatched} must have flux_int and flux_period either "
                "both finite or both null",
                reason="orphan_frame_column",
                data={"indexes": mismatched},
            )
        nonpositive = np.flatnonzero(
            ~np.isnan(period_values) & (period_values <= 0)
        ).tolist()
        if nonpositive:
            raise SampleTableV2Error(
                f"flux_period must be > 0 for row(s) {nonpositive}",
                reason="invalid_period",
                data={"indexes": nonpositive},
            )


def resolve_sample_flux(
    samples: pd.DataFrame,
    *,
    fallback_frame: SampleFluxFrame | None = None,
) -> SampleFluxResolution:
    """Resolve per-row flux with ``explicit -> row-frame -> fallback-frame`` precedence.

    ``sources`` is the closed per-row provenance; ``explicit_mask`` is derived
    only from it. Rows that cannot be resolved (and fallback rows whose unit does
    not match the row ``dev_unit``) fail fast with their indexes.
    """
    validate_sample_table_v2(samples, allow_empty=True)
    if samples.empty:
        return SampleFluxResolution(values=np.empty(0, dtype=np.float64), sources=())
    dev_values = _coerce_numeric_column(samples, DEV_VALUE_COLUMN)
    dev_units = samples[DEV_UNIT_COLUMN].astype(str).to_numpy()
    flux_values = (
        _coerce_numeric_column(samples, FLUX_COLUMN)
        if FLUX_COLUMN in samples.columns
        else None
    )
    int_values = (
        _coerce_numeric_column(samples, FLUX_INT_COLUMN)
        if FLUX_INT_COLUMN in samples.columns
        else None
    )
    period_values = (
        _coerce_numeric_column(samples, FLUX_PERIOD_COLUMN)
        if FLUX_PERIOD_COLUMN in samples.columns
        else None
    )
    row_count = len(samples)
    values = np.empty(row_count, dtype=np.float64)
    sources: list[SampleFluxSource] = []
    unresolved: list[int] = []
    for index in range(row_count):
        if flux_values is not None and not np.isnan(flux_values[index]):
            values[index] = flux_values[index]
            sources.append("explicit")
            continue
        if (
            int_values is not None
            and period_values is not None
            and not np.isnan(int_values[index])
            and not np.isnan(period_values[index])
        ):
            values[index] = (dev_values[index] - int_values[index]) / period_values[
                index
            ]
            sources.append("row-frame")
            continue
        if fallback_frame is not None and fallback_frame.dev_unit == dev_units[index]:
            values[index] = (
                dev_values[index] - fallback_frame.flux_int
            ) / fallback_frame.flux_period
            sources.append("fallback-frame")
            continue
        unresolved.append(index)
    if unresolved:
        raise SampleTableV2Error(
            f"unable to resolve flux for row(s) {unresolved}: no explicit flux, "
            "no row frame, and no matching fallback frame",
            reason="unresolved_flux",
            data={"indexes": unresolved},
        )
    return SampleFluxResolution(values=values, sources=tuple(sources))


@dataclass(frozen=True, slots=True)
class LegacySampleFluxFrame:
    """Legacy migration frame: values use the declared legacy unit."""

    dev_unit: LegacyDeviceValueUnit
    flux_int: float
    flux_period: float

    def __post_init__(self) -> None:
        if self.dev_unit not in _LEGACY_DEVICE_VALUE_UNITS:
            raise SampleTableV2Error(
                f"unsupported legacy unit {self.dev_unit!r}; expected one of "
                f"{sorted(_LEGACY_DEVICE_VALUE_UNITS)}",
                reason="invalid_unit",
                data={"unit": self.dev_unit},
            )
        _require_real_frame_value(self.flux_int, field="flux_int")
        _require_real_frame_value(self.flux_period, field="flux_period")
        if not np.isfinite(self.flux_int):
            raise SampleTableV2Error(
                f"flux_int must be finite, got {self.flux_int!r}",
                reason="non_finite_value",
                data={"field": "flux_int"},
            )
        if not np.isfinite(self.flux_period) or self.flux_period <= 0:
            raise SampleTableV2Error(
                f"flux_period must be finite and > 0, got {self.flux_period!r}",
                reason="invalid_period",
                data={"field": "flux_period"},
            )


def _require_matching_physical_kind(dev_value_unit: str, frame_unit: str) -> None:
    if _physical_kind(dev_value_unit) != _physical_kind(frame_unit):
        raise SampleTableV2Error(
            f"flux frame unit {frame_unit!r} does not match device value "
            f"physical kind of {dev_value_unit!r}",
            reason="frame_kind_mismatch",
            data={"dev_unit": dev_value_unit, "frame_unit": frame_unit},
        )


def migrate_sample_table_v2(
    samples: pd.DataFrame,
    *,
    dev_value_column: str,
    dev_value_unit: LegacyDeviceValueUnit,
    flux_column: str | None = None,
    flux_frame: LegacySampleFluxFrame | None = None,
) -> pd.DataFrame:
    """Convert a legacy user CSV to a flat v2 table under one declared source unit.

    The single scalar ``dev_value_unit`` applies to the whole ``dev_value_column``:
    A/mA/V/mV closed literals only, A/V pass through and mA/mV divide by 1000.
    Frame values normalize to base A/V units; the frame physical kind must match
    the device value kind. The input DataFrame is never mutated.
    """
    if not isinstance(samples, pd.DataFrame):
        raise SampleTableV2Error(
            "samples must be a pandas DataFrame", reason="invalid_input"
        )
    if dev_value_unit not in _LEGACY_DEVICE_VALUE_UNITS:
        raise SampleTableV2Error(
            f"unsupported source unit {dev_value_unit!r}; expected one of "
            f"{sorted(_LEGACY_DEVICE_VALUE_UNITS)}",
            reason="invalid_unit",
            data={"unit": dev_value_unit},
        )
    present_v2 = [c for c in SAMPLE_COORDINATE_COLUMNS if c in samples.columns]
    if present_v2:
        raise SampleTableV2Error(
            f"source already contains v2 coordinate column(s): {present_v2}",
            reason="source_has_v2_columns",
            data={"columns": present_v2},
        )
    if dev_value_column not in samples.columns:
        raise SampleTableV2Error(
            f"source column {dev_value_column!r} not found",
            reason="missing_source_column",
            data={"column": dev_value_column},
        )
    if flux_column is not None and flux_column not in samples.columns:
        raise SampleTableV2Error(
            f"source column {flux_column!r} not found",
            reason="missing_source_column",
            data={"column": flux_column},
        )
    if flux_column == dev_value_column:
        raise SampleTableV2Error(
            "flux_column must differ from dev_value_column",
            reason="invalid_mapping",
            data={"column": dev_value_column},
        )
    if flux_frame is not None:
        _require_matching_physical_kind(dev_value_unit, flux_frame.dev_unit)
    scale = _LEGACY_UNIT_SCALE[dev_value_unit]
    base_unit = _base_unit_of(dev_value_unit)
    dev_values = _coerce_numeric_column(samples, dev_value_column)
    _require_finite_non_null(dev_values, column=dev_value_column)
    output = pd.DataFrame(
        {
            DEV_VALUE_COLUMN: dev_values * scale,
            DEV_UNIT_COLUMN: base_unit,
        }
    )
    if flux_column is not None:
        output[FLUX_COLUMN] = samples[flux_column].to_numpy()
    if flux_frame is not None:
        frame_scale = _LEGACY_UNIT_SCALE[flux_frame.dev_unit]
        output[FLUX_INT_COLUMN] = flux_frame.flux_int * frame_scale
        output[FLUX_PERIOD_COLUMN] = flux_frame.flux_period * frame_scale
    mapped = {dev_value_column, flux_column}
    for column in samples.columns:
        if column in mapped:
            continue
        output[column] = samples[column].to_numpy()
    coordinate = [c for c in SAMPLE_COORDINATE_COLUMNS if c in output.columns]
    measurement = [c for c in samples.columns if c not in mapped]
    output = output[coordinate + measurement]
    validate_sample_table_v2(output, allow_empty=True)
    return output
