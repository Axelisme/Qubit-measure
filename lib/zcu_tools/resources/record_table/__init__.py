"""Append-only JSON records and opt-in flat v2 analysis coordinates."""

from .jsonl import JsonlRecordTable
from .projection import flux_from_frame, project_device_coordinates
from .schema import (
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    FLUX_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
    RECORD_COORDINATE_COLUMNS,
    DeviceValueUnit,
    LegacyDeviceValueUnit,
    LegacyRecordFluxFrame,
    RecordFluxFrame,
    RecordFluxResolution,
    RecordFluxSource,
    RecordTableV2Error,
    migrate_record_table_v2,
    resolve_record_flux,
    validate_record_table_v2,
)

__all__ = [
    "JsonlRecordTable",
    "project_device_coordinates",
    "flux_from_frame",
    "DEV_UNIT_COLUMN",
    "DEV_VALUE_COLUMN",
    "FLUX_COLUMN",
    "FLUX_INT_COLUMN",
    "FLUX_PERIOD_COLUMN",
    "RECORD_COORDINATE_COLUMNS",
    "DeviceValueUnit",
    "LegacyDeviceValueUnit",
    "LegacyRecordFluxFrame",
    "RecordFluxFrame",
    "RecordFluxResolution",
    "RecordFluxSource",
    "RecordTableV2Error",
    "migrate_record_table_v2",
    "resolve_record_flux",
    "validate_record_table_v2",
]
