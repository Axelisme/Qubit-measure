"""Grouped Labber dataset persistence."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
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
    DatasetRole,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
)
from .paths import format_ext

GROUPED_DATASET_VERSION = 2
GROUPED_VERSION_ATTR = "zcu_tools.grouped_dataset_version"
DATASET_ROLES_ATTR = "zcu_tools.dataset_roles"
DATASET_ROLE_CHANNELS_ATTR = "zcu_tools.dataset_role_channels"
DATASET_ROLE_ATTR = "zcu_tools.dataset_role"
_STREAMING_VERSION_ATTR = "zcu_tools.streaming_grouped_dataset_version"
_STREAMING_GROUPED_DATASET_VERSION = 1
_GROUPED_V1_MIGRATION_COMMAND = """.venv/bin/python script/migrate_experiment_data.py \\
  --experiment grouped/v1 \\
  --input INPUT.hdf5 \\
  --output OUTPUT.hdf5"""


def save_grouped_labber_data(
    path: str,
    roles: Mapping[str | DatasetRole, LabberPayload],
    *,
    metadata: LabberMetadata | None = None,
) -> str:
    """Save common-grid role payloads as parallel channels in one Labber log."""
    grouped = GroupedLabberData(roles, metadata=metadata)
    _validate_v2_payloads(grouped.roles)

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
    path = format_ext(path)
    log_name = os.path.splitext(os.path.basename(path))[0]
    role_items = list(grouped.roles.items())
    role_names = [str(role) for role, _payload in role_items]
    channel_names = [payload.data.name for _role, payload in role_items]

    with h5py.File(path, "x") as f:
        _write_uniform_multi_channel_log_group(
            f,
            [payload for _role, payload in role_items],
            effective_metadata,
            log_name=log_name,
            creation_time=creation_time,
        )
        f.attrs[GROUPED_VERSION_ATTR] = GROUPED_DATASET_VERSION
        f.attrs[DATASET_ROLES_ATTR] = _str_array(role_names)
        f.attrs[DATASET_ROLE_CHANNELS_ATTR] = _str_array(channel_names)

    return path


def load_grouped_labber_data(
    path: str,
    *,
    required_roles: Sequence[str | DatasetRole] | None = None,
) -> GroupedLabberData:
    """Load root-only grouped v2 or marker-qualified streaming grouped v1."""
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
            return _load_grouped_v2(f, required_roles)

        if version == _STREAMING_GROUPED_DATASET_VERSION:
            if raw_streaming_version is None:
                raise ValueError(
                    "grouped dataset version 1 requires manual migration:\n"
                    f"{_GROUPED_V1_MIGRATION_COMMAND}"
                )
            streaming_version = _read_exact_version(
                raw_streaming_version, "streaming grouped dataset"
            )
            if streaming_version != _STREAMING_GROUPED_DATASET_VERSION:
                raise ValueError(
                    "unsupported grouped/streaming dataset version combination "
                    f"{version!r}/{streaming_version!r}"
                )
            return _load_streaming_grouped_v1(f, required_roles)

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
    required_roles: Sequence[str | DatasetRole] | None,
) -> GroupedLabberData:
    if DATASET_ROLE_ATTR in f.attrs:
        raise ValueError("grouped v2 must not declare a singular root dataset role")
    declared_roles = _read_declared_roles(f)
    declared_channels = _read_declared_channels(f)
    if len(declared_roles) != len(declared_channels):
        raise ValueError("grouped v2 role-to-channel mapping lengths do not match")
    if len(set(declared_channels)) != len(declared_channels):
        raise ValueError("grouped v2 role channel labels must be unique")
    if any(not channel for channel in declared_channels):
        raise ValueError("grouped v2 role channel labels must be non-empty")

    logs = _all_log_refs(f)
    if len(logs) != 1 or any(
        isinstance(name, str) and name.startswith("Log_") for name in f
    ):
        raise ValueError("grouped v2 must contain exactly one root Labber log")

    channel_values, axes, relative_timestamps = _read_uniform_multi_channel_log(f, f)
    if list(channel_values) != declared_channels:
        raise ValueError(
            "grouped v2 role-to-channel mapping does not match actual Labber channels"
        )

    metadata = _read_metadata(f)
    timestamps = (
        None
        if relative_timestamps is None
        else metadata.creation_time + np.asarray(relative_timestamps)
    )
    payloads = {
        role: LabberPayload(
            Axis(channel, channel_values[channel][0], channel_values[channel][1]),
            [Axis(name, unit, values) for name, unit, values in axes],
            timestamps=timestamps,
        )
        for role, channel in zip(declared_roles, declared_channels)
    }
    _validate_v2_payloads(payloads)
    if required_roles is not None:
        _validate_required_roles(payloads, required_roles)
    return GroupedLabberData(
        {str(role): payload for role, payload in payloads.items()}, metadata=metadata
    )


def _load_streaming_grouped_v1(
    f: h5py.File,
    required_roles: Sequence[str | DatasetRole] | None,
) -> GroupedLabberData:
    declared_roles = _read_declared_roles(f)
    logs = _all_log_refs(f)
    if len(declared_roles) != len(logs):
        raise ValueError("grouped dataset role list does not match log group count")

    metadata = _read_metadata(f)
    payloads: dict[DatasetRole, LabberPayload] = {}
    seen_from_logs: list[DatasetRole] = []
    for log in logs:
        role = _read_log_role(log)
        if role in payloads:
            raise ValueError(f"duplicate dataset role {role!r}")
        z, axes, relative_timestamps = _read_single_log(f, log)
        z_name, z_unit = _read_log_label(f, log)
        timestamps = (
            None
            if relative_timestamps is None
            else metadata.creation_time + np.asarray(relative_timestamps)
        )
        payloads[role] = LabberPayload(
            Axis(z_name, z_unit, z),
            [Axis(name, unit, values) for name, unit, values in axes],
            timestamps=timestamps,
        )
        seen_from_logs.append(role)

    if declared_roles != seen_from_logs:
        raise ValueError("grouped dataset role list does not match log roles")
    if required_roles is not None:
        _validate_required_roles(payloads, required_roles)
    return GroupedLabberData(
        {str(role): payload for role, payload in payloads.items()}, metadata=metadata
    )


def _validate_v2_payloads(
    payloads: Mapping[DatasetRole, LabberPayload],
) -> None:
    reference_shape: tuple[int, ...] | None = None
    reference_axes: list[tuple[str, str, np.ndarray]] | None = None
    reference_timestamps: np.ndarray | None = None
    physical_labels: list[str] = []

    for role, payload in payloads.items():
        try:
            values = np.asarray(payload.data.values)
        except ValueError as exc:
            raise ValueError(
                f"grouped v2 role {role!r} has ragged or vector-valued data"
            ) from exc
        if values.dtype == object or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"grouped v2 role {role!r} data must be numeric")
        if values.ndim < 1 or values.size == 0:
            raise ValueError(
                f"grouped v2 role {role!r} data must have at least one dimension "
                "and one value"
            )
        if not payload.axes:
            raise ValueError("grouped v2 requires at least one step axis")
        if len(payload.axes) != values.ndim:
            raise ValueError(
                f"grouped v2 role {role!r} shape {values.shape} requires "
                f"{values.ndim} axes, got {len(payload.axes)}"
            )
        if not isinstance(payload.data.name, str) or not payload.data.name:
            raise ValueError("grouped v2 physical channel labels must be non-empty")
        if not isinstance(payload.data.unit, str):
            raise ValueError("grouped v2 channel units must be strings")

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
            expected_length = values.shape[-1 - index]
            if len(axis_values) != expected_length:
                raise ValueError(
                    f"grouped v2 role {role!r} shape {values.shape} does not match "
                    f"axis {axis.name!r} length {len(axis_values)}"
                )
            normalized_axes.append((axis.name, axis.unit, axis_values))

        expected_timestamps = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
        timestamps: np.ndarray | None
        if payload.timestamps is None:
            timestamps = None
        else:
            timestamps = np.asarray(payload.timestamps, dtype=float)
            if timestamps.ndim != 1 or len(timestamps) != expected_timestamps:
                raise ValueError(
                    f"grouped v2 role {role!r} timestamps must be a flat array of "
                    f"length {expected_timestamps}"
                )

        if reference_shape is None:
            reference_shape = values.shape
            reference_axes = normalized_axes
            reference_timestamps = timestamps
            physical_labels.extend(axis.name for axis in payload.axes)
        else:
            assert reference_axes is not None
            if values.shape != reference_shape or len(normalized_axes) != len(
                reference_axes
            ):
                raise ValueError(
                    "grouped v2 roles must share one common grid and shape"
                )
            for expected, actual in zip(reference_axes, normalized_axes):
                if expected[:2] != actual[:2] or not np.array_equal(
                    expected[2], actual[2], equal_nan=True
                ):
                    raise ValueError("grouped v2 roles must share one common grid")
            if (reference_timestamps is None) != (timestamps is None) or (
                reference_timestamps is not None
                and timestamps is not None
                and not np.array_equal(reference_timestamps, timestamps, equal_nan=True)
            ):
                raise ValueError("grouped v2 roles must have identical timestamps")

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


def _read_declared_roles(f: h5py.File) -> list[DatasetRole]:
    raw = _decode(f.attrs.get(DATASET_ROLES_ATTR))
    if raw is None:
        raise ValueError("grouped dataset is missing dataset role list")
    values: list[Any] = [raw] if isinstance(raw, str) else list(raw)

    roles: list[DatasetRole] = []
    seen: set[DatasetRole] = set()
    for value in values:
        role = DatasetRole(value)
        if role in seen:
            raise ValueError(f"duplicate dataset role {role!r}")
        seen.add(role)
        roles.append(role)
    return roles


def _read_declared_channels(f: h5py.File) -> list[str]:
    raw = _decode(f.attrs.get(DATASET_ROLE_CHANNELS_ATTR))
    if raw is None:
        raise ValueError("grouped v2 is missing dataset role channel mapping")
    values = [raw] if isinstance(raw, str) else list(raw)
    channels: list[str] = []
    for value in values:
        decoded = _decode(value)
        if not isinstance(decoded, str):
            raise ValueError("grouped v2 role channel mapping must contain strings")
        channels.append(decoded)
    return channels


def _read_log_role(log: h5py.File | h5py.Group) -> DatasetRole:
    raw = _decode(log.attrs.get(DATASET_ROLE_ATTR))
    if raw is None:
        raise ValueError("grouped log is missing dataset role")
    return DatasetRole(raw)


def _validate_required_roles(
    payloads: Mapping[DatasetRole, LabberPayload],
    required_roles: Sequence[str | DatasetRole],
) -> None:
    required: set[DatasetRole] = set()
    for raw_role in required_roles:
        role = DatasetRole(raw_role)
        if role in required:
            raise ValueError(f"duplicate required dataset role {role!r}")
        required.add(role)

    present = set(payloads)
    missing = required - present
    unknown = present - required
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing required dataset role(s): {names}")
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown dataset role(s): {names}")
