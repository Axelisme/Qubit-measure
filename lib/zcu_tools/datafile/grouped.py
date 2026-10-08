"""Grouped Labber dataset persistence."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import h5py
import numpy as np

from .labber import (
    _all_log_refs,
    _decode,
    _read_log_label,
    _read_single_log,
    _read_tags,
    _read_uniform_multi_channel_log,
    _resolve_path,
    _str_array,
    _write_uniform_multi_channel_log_group,
)
from .models import (
    Axis,
    DataVariable,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
)
from .paths import format_ext

GROUPED_DATASET_VERSION = 2
GROUPED_VERSION_ATTR = "zcu_tools.grouped_dataset_version"
DATA_VARIABLES_ATTR = "zcu_tools.dataset_roles"
DATA_VARIABLE_CHANNELS_ATTR = "zcu_tools.dataset_role_channels"
DATA_VARIABLE_ATTR = "zcu_tools.dataset_role"
_STREAMING_VERSION_ATTR = "zcu_tools.streaming_grouped_dataset_version"
_STREAMING_GROUPED_DATASET_VERSION = 1


def save_grouped_labber_data(
    path: str,
    variables: Mapping[str | DataVariable, LabberPayload],
    *,
    metadata: LabberMetadata | None = None,
) -> str:
    """Save named common-grid variables as parallel channels in one Labber log.

    path is the destination; its filename is normalized to the Labber extension.
    variables maps snake_case identities to payloads sharing axes, shape and
    timestamps. metadata is shared, or empty when omitted. Return the written
    path. Invalid variables/grid raise ValueError before creating the file;
    invalid payload/metadata types raise TypeError. Existing paths raise FileExistsError.
    """
    return write_grouped_labber_data_file(
        format_ext(path), GroupedLabberData(variables, metadata=metadata)
    )


def write_grouped_labber_data_file(path: str, grouped: GroupedLabberData) -> str:
    """Write common-grid grouped channels at the exact path and return it.

    grouped carries ordered stable variable identities and shared metadata.
    Payloads must share axes, shape and timestamps. Invalid grids/labels raise
    ValueError before creating the destination. Existing paths raise
    FileExistsError and I/O errors propagate. No path normalization is applied.
    A failed I/O operation may leave a partial new file.
    """
    _validate_v2_payloads(grouped.variables)

    raw_metadata = grouped.metadata
    creation_time = (
        time.time()
        if raw_metadata.creation_time is None
        else float(raw_metadata.creation_time)
    )
    effective_metadata = LabberMetadata(
        comment=raw_metadata.comment,
        tags=raw_metadata.tags,
        project=raw_metadata.project,
        user=raw_metadata.user,
        creation_time=creation_time,
    )
    log_name = os.path.splitext(os.path.basename(path))[0]
    variable_items = list(grouped.variables.items())
    variable_names = [str(variable) for variable, _payload in variable_items]
    channel_names = [payload.data.name for _variable, payload in variable_items]

    with h5py.File(path, "x") as f:
        _write_uniform_multi_channel_log_group(
            f,
            [payload for _variable, payload in variable_items],
            effective_metadata,
            log_name=log_name,
            creation_time=creation_time,
        )
        f.attrs[GROUPED_VERSION_ATTR] = GROUPED_DATASET_VERSION
        f.attrs[DATA_VARIABLES_ATTR] = _str_array(variable_names)
        f.attrs[DATA_VARIABLE_CHANNELS_ATTR] = _str_array(channel_names)

    return path


def load_grouped_labber_data(
    path: str,
    *,
    required_variables: Sequence[str | DataVariable] | None = None,
) -> GroupedLabberData:
    """Load root-only grouped v2 or marker-qualified streaming grouped v1.

    path identifies the local Labber file; file resolution adds its extension
    when omitted. required_variables, when supplied, must match the stored
    identities exactly; None accepts all stored variables for inspection.
    Return named payloads and shared metadata without modifying the input file.
    Invalid names, schema/version, mapping or variable set raise ValueError;
    unreadable files propagate their I/O error. Unmarked grouped v1 is rejected.
    """
    path = _resolve_path(path)
    with h5py.File(path, "r") as f:
        raw_version = f.attrs.get(GROUPED_VERSION_ATTR)
        if raw_version is None:
            raise ValueError("file is not a grouped Labber dataset")
        version = _read_exact_version(raw_version, "grouped dataset")

        raw_streaming_version = f.attrs.get(_STREAMING_VERSION_ATTR)
        if version == GROUPED_DATASET_VERSION:
            if raw_streaming_version is not None:
                raise ValueError(
                    "unsupported grouped/streaming dataset version combination "
                    f"{version!r}/{raw_streaming_version!r}"
                )
            return _load_grouped_v2(f, required_variables)

        if version == _STREAMING_GROUPED_DATASET_VERSION:
            if raw_streaming_version is None:
                raise ValueError(
                    "unmarked grouped dataset version 1 is unsupported; "
                    "load a canonical grouped v2 file"
                )
            streaming_version = _read_exact_version(
                raw_streaming_version, "streaming grouped dataset"
            )
            if streaming_version != _STREAMING_GROUPED_DATASET_VERSION:
                raise ValueError(
                    "unsupported grouped/streaming dataset version combination "
                    f"{version!r}/{streaming_version!r}"
                )
            return _load_streaming_grouped_v1(f, required_variables)

        raise ValueError(f"unsupported grouped dataset version {raw_version!r}")


def _read_exact_version(raw_version: Any, label: str) -> int:
    value = np.asarray(raw_version)
    if value.ndim != 0:
        raise ValueError(f"invalid {label} version {raw_version!r}")
    scalar = value.item()
    if isinstance(scalar, bool) or not isinstance(scalar, Integral):
        raise ValueError(f"invalid {label} version {raw_version!r}")
    return int(scalar)


def _load_grouped_v2(
    f: h5py.File,
    required_variables: Sequence[str | DataVariable] | None,
) -> GroupedLabberData:
    if DATA_VARIABLE_ATTR in f.attrs:
        raise ValueError("grouped v2 must not declare a singular root data variable")
    declared_variables = _read_declared_variables(f)
    declared_channels = _read_declared_channels(f)
    if len(declared_variables) != len(declared_channels):
        raise ValueError("grouped v2 variable-to-channel mapping lengths do not match")
    if len(set(declared_channels)) != len(declared_channels):
        raise ValueError("grouped v2 variable channel labels must be unique")
    if any(not channel for channel in declared_channels):
        raise ValueError("grouped v2 variable channel labels must be non-empty")

    logs = _all_log_refs(f)
    if len(logs) != 1 or any(
        isinstance(name, str) and name.startswith("Log_") for name in f
    ):
        raise ValueError("grouped v2 must contain exactly one root Labber log")

    channel_values, axes, relative_timestamps = _read_uniform_multi_channel_log(f, f)
    if list(channel_values) != declared_channels:
        raise ValueError(
            "grouped v2 variable-to-channel mapping does not match actual Labber channels"
        )

    metadata = _read_metadata(f)
    timestamps = (
        None
        if relative_timestamps is None
        else metadata.creation_time + np.asarray(relative_timestamps)
    )
    payloads = {
        variable: LabberPayload(
            Axis(channel, channel_values[channel][0], channel_values[channel][1]),
            [Axis(name, unit, values) for name, unit, values in axes],
            timestamps=timestamps,
        )
        for variable, channel in zip(
            declared_variables, declared_channels, strict=False
        )
    }
    _validate_v2_payloads(payloads)
    if required_variables is not None:
        _validate_required_variables(payloads, required_variables)
    return GroupedLabberData(
        {str(variable): payload for variable, payload in payloads.items()},
        metadata=metadata,
    )


def _load_streaming_grouped_v1(
    f: h5py.File,
    required_variables: Sequence[str | DataVariable] | None,
) -> GroupedLabberData:
    declared_variables = _read_declared_variables(f)
    logs = _all_log_refs(f)
    if len(declared_variables) != len(logs):
        raise ValueError("grouped data variable list does not match log group count")

    metadata = _read_metadata(f)
    payloads: dict[DataVariable, LabberPayload] = {}
    seen_from_logs: list[DataVariable] = []
    for log in logs:
        variable = _read_log_variable(log)
        if variable in payloads:
            raise ValueError(f"duplicate data variable {variable!r}")
        z, axes, relative_timestamps = _read_single_log(f, log)
        z_name, z_unit = _read_log_label(f, log)
        timestamps = (
            None
            if relative_timestamps is None
            else metadata.creation_time + np.asarray(relative_timestamps)
        )
        payloads[variable] = LabberPayload(
            Axis(z_name, z_unit, z),
            [Axis(name, unit, values) for name, unit, values in axes],
            timestamps=timestamps,
        )
        seen_from_logs.append(variable)

    if declared_variables != seen_from_logs:
        raise ValueError("grouped data variable list does not match log variables")
    if required_variables is not None:
        _validate_required_variables(payloads, required_variables)
    return GroupedLabberData(
        {str(variable): payload for variable, payload in payloads.items()},
        metadata=metadata,
    )


@dataclass(frozen=True)
class _V2VariableGrid:
    shape: tuple[int, ...]
    axes: list[tuple[str, str, np.ndarray]]
    timestamps: np.ndarray | None


def _normalize_v2_variable_axes(
    variable: DataVariable, payload: LabberPayload, shape: tuple[int, ...]
) -> list[tuple[str, str, np.ndarray]]:
    normalized_axes: list[tuple[str, str, np.ndarray]] = []
    for index, axis in enumerate(payload.axes):
        if not isinstance(axis.name, str) or not axis.name:
            raise ValueError("grouped v2 physical channel labels must be non-empty")
        if not isinstance(axis.unit, str):
            raise ValueError("grouped v2 channel units must be strings")
        axis_values = np.asarray(axis.values)
        if (
            axis_values.ndim != 1
            or axis_values.dtype == object
            or not np.issubdtype(axis_values.dtype, np.number)
        ):
            raise ValueError(
                f"grouped v2 axis {axis.name!r} values must be one-dimensional numeric data"
            )
        expected_length = shape[-1 - index]
        if len(axis_values) != expected_length:
            raise ValueError(
                f"grouped v2 variable {variable!r} shape {shape} does not match "
                f"axis {axis.name!r} length {len(axis_values)}"
            )
        normalized_axes.append((axis.name, axis.unit, axis_values))
    return normalized_axes


def _normalize_v2_variable_grid(
    variable: DataVariable, payload: LabberPayload
) -> _V2VariableGrid:
    try:
        values = np.asarray(payload.data.values)
    except ValueError as exc:
        raise ValueError(
            f"grouped v2 variable {variable!r} has ragged or vector-valued data"
        ) from exc
    if values.dtype == object or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"grouped v2 variable {variable!r} data must be numeric")
    if values.ndim < 1 or values.size == 0:
        raise ValueError(
            f"grouped v2 variable {variable!r} data must have at least one dimension "
            "and one value"
        )
    if not payload.axes:
        raise ValueError("grouped v2 requires at least one step axis")
    if len(payload.axes) != values.ndim:
        raise ValueError(
            f"grouped v2 variable {variable!r} shape {values.shape} requires "
            f"{values.ndim} axes, got {len(payload.axes)}"
        )
    if not isinstance(payload.data.name, str) or not payload.data.name:
        raise ValueError("grouped v2 physical channel labels must be non-empty")
    if not isinstance(payload.data.unit, str):
        raise ValueError("grouped v2 channel units must be strings")

    normalized_axes = _normalize_v2_variable_axes(variable, payload, values.shape)
    expected_timestamps = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
    timestamps: np.ndarray | None
    if payload.timestamps is None:
        timestamps = None
    else:
        timestamps = np.asarray(payload.timestamps, dtype=float)
        if timestamps.ndim != 1 or len(timestamps) != expected_timestamps:
            raise ValueError(
                f"grouped v2 variable {variable!r} timestamps must be a flat array of "
                f"length {expected_timestamps}"
            )
    return _V2VariableGrid(values.shape, normalized_axes, timestamps)


def _validate_v2_common_grid(
    reference: _V2VariableGrid, actual: _V2VariableGrid
) -> None:
    if actual.shape != reference.shape or len(actual.axes) != len(reference.axes):
        raise ValueError("grouped v2 variables must share one common grid and shape")
    for expected_axis, actual_axis in zip(reference.axes, actual.axes, strict=True):
        if expected_axis[:2] != actual_axis[:2] or not np.array_equal(
            expected_axis[2], actual_axis[2], equal_nan=True
        ):
            raise ValueError("grouped v2 variables must share one common grid")
    if (reference.timestamps is None) != (actual.timestamps is None) or (
        reference.timestamps is not None
        and actual.timestamps is not None
        and not np.array_equal(reference.timestamps, actual.timestamps, equal_nan=True)
    ):
        raise ValueError("grouped v2 variables must have identical timestamps")


def _validate_v2_payloads(
    payloads: Mapping[DataVariable, LabberPayload],
) -> None:
    reference_grid: _V2VariableGrid | None = None
    physical_labels: list[str] = []

    for variable, payload in payloads.items():
        # Finish each variable before comparing it or moving to the next variable, so
        # malformed payloads retain their first-error ordering.
        variable_grid = _normalize_v2_variable_grid(variable, payload)
        if reference_grid is None:
            reference_grid = variable_grid
            physical_labels.extend(axis.name for axis in payload.axes)
        else:
            _validate_v2_common_grid(reference_grid, variable_grid)
        physical_labels.append(payload.data.name)

    if len(set(physical_labels)) != len(physical_labels):
        raise ValueError("grouped v2 physical channel labels must be globally unique")


def _read_metadata(f: h5py.File) -> LabberMetadata:
    comment = _decode(f.attrs.get("comment", "")) or ""
    tags, project, user = _read_tags(f)
    creation_time = float(f.attrs.get("creation_time", 0.0) or 0.0)
    return LabberMetadata(
        comment=comment,
        tags=tags,
        project=project,
        user=user,
        creation_time=creation_time,
    )


def _read_declared_variables(f: h5py.File) -> list[DataVariable]:
    raw = _decode(f.attrs.get(DATA_VARIABLES_ATTR))
    if raw is None:
        raise ValueError("grouped dataset is missing data variable list")
    values: list[Any] = [raw] if isinstance(raw, str) else list(raw)

    variables: list[DataVariable] = []
    seen: set[DataVariable] = set()
    for value in values:
        variable = DataVariable(value)
        if variable in seen:
            raise ValueError(f"duplicate data variable {variable!r}")
        seen.add(variable)
        variables.append(variable)
    return variables


def _read_declared_channels(f: h5py.File) -> list[str]:
    raw = _decode(f.attrs.get(DATA_VARIABLE_CHANNELS_ATTR))
    if raw is None:
        raise ValueError("grouped v2 is missing data variable channel mapping")
    values = [raw] if isinstance(raw, str) else list(raw)
    channels: list[str] = []
    for value in values:
        decoded = _decode(value)
        if not isinstance(decoded, str):
            raise ValueError("grouped v2 variable channel mapping must contain strings")
        channels.append(decoded)
    return channels


def _read_log_variable(log: h5py.File | h5py.Group) -> DataVariable:
    raw = _decode(log.attrs.get(DATA_VARIABLE_ATTR))
    if raw is None:
        raise ValueError("grouped log is missing data variable")
    return DataVariable(raw)


def _validate_required_variables(
    payloads: Mapping[DataVariable, LabberPayload],
    required_variables: Sequence[str | DataVariable],
) -> None:
    required: set[DataVariable] = set()
    for raw_variable in required_variables:
        variable = DataVariable(raw_variable)
        if variable in required:
            raise ValueError(f"duplicate required data variable {variable!r}")
        required.add(variable)

    present = set(payloads)
    missing = required - present
    unknown = present - required
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing required data variable(s): {names}")
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown data variable(s): {names}")
