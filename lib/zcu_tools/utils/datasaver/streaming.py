"""Streaming grouped Labber dataset writer.

This module is the partial-write counterpart to the one-shot grouped writer.
It keeps the same Labber-compatible on-disk layout, but preallocates each role
with NaNs and lets callers commit the outer workflow row one slice at a time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from .grouped import (
    DATASET_ROLE_ATTR,
    DATASET_ROLES_ATTR,
    GROUPED_DATASET_VERSION,
    GROUPED_VERSION_ATTR,
)
from .labber import _str_array, _write_payload_to_log
from .models import Axis, DatasetRole, LabberMetadata, LabberPayload, as_axis
from .paths import format_ext

STREAMING_DATASET_VERSION = 1
STREAMING_VERSION_ATTR = "zcu_tools.streaming_grouped_dataset_version"
STREAMING_FINALIZED_ATTR = "zcu_tools.streaming_finalized"


@dataclass(frozen=True, slots=True)
class StreamingLabberRoleSpec:
    """Schema for one streamable grouped Labber role."""

    role: DatasetRole
    data_name: str
    data_unit: str
    axes: tuple[Axis, ...]
    shape: tuple[int, ...]
    fill_value: complex | float = np.nan
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        role: str | DatasetRole,
        data_name: str,
        data_unit: str,
        axes: Sequence[Axis | tuple[str, str, Any]],
        shape: Sequence[int],
        *,
        fill_value: complex | float = np.nan,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_axes = tuple(
            as_axis(axis, f"axes[{i}]") for i, axis in enumerate(axes)
        )
        normalized_shape = tuple(int(dim) for dim in shape)
        if not normalized_shape:
            raise ValueError("streaming role shape must have at least one dimension")
        if len(normalized_axes) != len(normalized_shape):
            raise ValueError(
                f"role {role!r} axes count {len(normalized_axes)} must match "
                f"shape rank {len(normalized_shape)}"
            )
        for axis, expected in zip(reversed(normalized_axes), normalized_shape):
            actual = int(np.asarray(axis.values, dtype=float).ravel().shape[0])
            if actual != expected:
                raise ValueError(
                    f"role {role!r} axis {axis.name!r} length {actual} != "
                    f"shape dimension {expected}"
                )

        object.__setattr__(self, "role", DatasetRole(role))
        object.__setattr__(self, "data_name", str(data_name))
        object.__setattr__(self, "data_unit", str(data_unit))
        object.__setattr__(self, "axes", normalized_axes)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "fill_value", fill_value)
        object.__setattr__(self, "attrs", dict(attrs or {}))


@dataclass(slots=True)
class _RoleHandle:
    spec: StreamingLabberRoleSpec
    target: h5py.File | h5py.Group
    data: h5py.Dataset
    timestamps: h5py.Dataset


class StreamingGroupedLabberWriter:
    """Open writer for a partial grouped Labber dataset."""

    def __init__(
        self,
        path: str,
        roles: Sequence[StreamingLabberRoleSpec],
        *,
        metadata: LabberMetadata | None = None,
    ) -> None:
        if not roles:
            raise ValueError("StreamingGroupedLabberWriter requires at least one role")
        self.path = format_ext(path)
        self._closed = False
        self._handles: dict[DatasetRole, _RoleHandle] = {}

        role_names: list[str] = []
        seen: set[DatasetRole] = set()
        for spec in roles:
            if spec.role in seen:
                raise ValueError(f"duplicate streaming dataset role {spec.role!r}")
            seen.add(spec.role)
            role_names.append(str(spec.role))

        raw_metadata = metadata if metadata is not None else LabberMetadata()
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
        log_name = os.path.splitext(os.path.basename(self.path))[0]

        self._file = h5py.File(self.path, "x")
        try:
            for index, spec in enumerate(roles):
                target: h5py.File | h5py.Group
                if index == 0:
                    target = self._file
                else:
                    target = self._file.create_group(f"Log_{index + 1}")
                payload = LabberPayload(
                    Axis(
                        spec.data_name,
                        spec.data_unit,
                        np.full(spec.shape, spec.fill_value, dtype=complex),
                    ),
                    spec.axes,
                    timestamps=np.full(_entry_count(spec.shape), creation_time),
                )
                _write_payload_to_log(
                    target,
                    payload,
                    effective_metadata,
                    log_name=log_name,
                    creation_time=creation_time,
                    write_tags=index == 0,
                )
                target.attrs[DATASET_ROLE_ATTR] = str(spec.role)
                for key, value in spec.attrs.items():
                    target.attrs[str(key)] = _attr_value(value)
                data_group = _require_group(target, "Data")
                self._handles[spec.role] = _RoleHandle(
                    spec=spec,
                    target=target,
                    data=_require_dataset(data_group, "Data"),
                    timestamps=_require_dataset(data_group, "Time stamp"),
                )

            self._file.attrs[GROUPED_VERSION_ATTR] = GROUPED_DATASET_VERSION
            self._file.attrs[DATASET_ROLES_ATTR] = _str_array(role_names)
            self._file.attrs[STREAMING_VERSION_ATTR] = STREAMING_DATASET_VERSION
            self._file.attrs[STREAMING_FINALIZED_ATTR] = False
            self.flush()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> StreamingGroupedLabberWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def write_outer_slice(
        self,
        role: str | DatasetRole,
        outer_index: int,
        values: Any,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Write one workflow row along the first data dimension."""
        self._ensure_open()
        role_key = DatasetRole(role)
        _write_outer_slice(
            self._handles[role_key], role_key, outer_index, values, timestamp
        )

    def flush(self) -> None:
        self._ensure_open()
        self._file.flush()

    def finalize(self) -> None:
        self._ensure_open()
        self._file.attrs[STREAMING_FINALIZED_ATTR] = True
        self.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._file.close()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming Labber writer is closed")

    @staticmethod
    def _creation_time(handle: _RoleHandle) -> float:
        return float(handle.target.attrs.get("creation_time", 0.0) or 0.0)


class StreamingLabberWriter:
    """Open writer for a partial single-log Labber dataset."""

    def __init__(
        self,
        path: str,
        spec: StreamingLabberRoleSpec,
        *,
        metadata: LabberMetadata | None = None,
    ) -> None:
        self.path = format_ext(path)
        self._closed = False
        raw_metadata = metadata if metadata is not None else LabberMetadata()
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
        log_name = os.path.splitext(os.path.basename(self.path))[0]

        self._file = h5py.File(self.path, "x")
        try:
            payload = LabberPayload(
                Axis(
                    spec.data_name,
                    spec.data_unit,
                    np.full(spec.shape, spec.fill_value, dtype=complex),
                ),
                spec.axes,
                timestamps=np.full(_entry_count(spec.shape), creation_time),
            )
            _write_payload_to_log(
                self._file,
                payload,
                effective_metadata,
                log_name=log_name,
                creation_time=creation_time,
                write_tags=True,
            )
            for key, value in spec.attrs.items():
                self._file.attrs[str(key)] = _attr_value(value)
            data_group = _require_group(self._file, "Data")
            self._handle = _RoleHandle(
                spec=spec,
                target=self._file,
                data=_require_dataset(data_group, "Data"),
                timestamps=_require_dataset(data_group, "Time stamp"),
            )
            self._file.attrs[STREAMING_VERSION_ATTR] = STREAMING_DATASET_VERSION
            self._file.attrs[STREAMING_FINALIZED_ATTR] = False
            self.flush()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> StreamingLabberWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def write_outer_slice(
        self,
        outer_index: int,
        values: Any,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Write one workflow row along the first data dimension."""
        self._ensure_open()
        _write_outer_slice(
            self._handle,
            self._handle.spec.role,
            outer_index,
            values,
            timestamp,
        )

    def flush(self) -> None:
        self._ensure_open()
        self._file.flush()

    def finalize(self) -> None:
        self._ensure_open()
        self._file.attrs[STREAMING_FINALIZED_ATTR] = True
        self.flush()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._file.close()
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming Labber writer is closed")


def open_streaming_grouped_labber_data(
    path: str,
    roles: Sequence[StreamingLabberRoleSpec],
    *,
    metadata: LabberMetadata | None = None,
) -> StreamingGroupedLabberWriter:
    """Open a new exact-path streaming grouped Labber writer."""
    return StreamingGroupedLabberWriter(path, roles, metadata=metadata)


def open_streaming_labber_data(
    path: str,
    spec: StreamingLabberRoleSpec,
    *,
    metadata: LabberMetadata | None = None,
) -> StreamingLabberWriter:
    """Open a new exact-path streaming single-log Labber writer."""
    return StreamingLabberWriter(path, spec, metadata=metadata)


def _write_outer_slice(
    handle: _RoleHandle,
    role: DatasetRole,
    outer_index: int,
    values: Any,
    timestamp: float | None,
) -> None:
    spec = handle.spec
    idx = int(outer_index)
    if idx < 0 or idx >= spec.shape[0]:
        raise IndexError(
            f"outer_index {idx} out of range for role {role!r} "
            f"with {spec.shape[0]} row(s)"
        )

    if len(spec.shape) == 1:
        arr = np.asarray(values, dtype=complex)
        if arr.shape not in {(), (1,)}:
            raise ValueError(
                f"role {role!r} expects a scalar row, got shape {arr.shape}"
            )
        scalar = complex(arr.reshape(-1)[0])
        handle.data[idx, -2, 0] = scalar.real
        handle.data[idx, -1, 0] = scalar.imag
        if timestamp is not None:
            handle.timestamps[0] = float(timestamp) - _creation_time(handle)
        return

    expected_shape = spec.shape[1:]
    arr = np.asarray(values, dtype=complex)
    if arr.shape != expected_shape:
        raise ValueError(
            f"role {role!r} row shape {arr.shape} != expected {expected_shape}"
        )

    n_x = spec.shape[-1]
    block_entries = int(np.prod(spec.shape[1:-1])) if len(spec.shape) > 2 else 1
    start = idx * block_entries
    stop = start + block_entries
    zf = arr.reshape(block_entries, n_x)
    handle.data[:, -2, start:stop] = np.real(zf).T
    handle.data[:, -1, start:stop] = np.imag(zf).T
    if timestamp is not None:
        handle.timestamps[start:stop] = float(timestamp) - _creation_time(handle)


def _creation_time(handle: _RoleHandle) -> float:
    return float(handle.target.attrs.get("creation_time", 0.0) or 0.0)


def _entry_count(shape: tuple[int, ...]) -> int:
    return int(np.prod(shape[:-1])) if len(shape) > 1 else 1


def _require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    child = parent[name]
    if not isinstance(child, h5py.Group):
        raise TypeError(f"expected HDF5 group at {child.name!r}")
    return child


def _require_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    child = parent[name]
    if not isinstance(child, h5py.Dataset):
        raise TypeError(f"expected HDF5 dataset at {child.name!r}")
    return child


def _attr_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _str_array([str(item) for item in value])
    return str(value)


__all__ = [
    "STREAMING_DATASET_VERSION",
    "STREAMING_VERSION_ATTR",
    "STREAMING_FINALIZED_ATTR",
    "StreamingGroupedLabberWriter",
    "StreamingLabberWriter",
    "StreamingLabberRoleSpec",
    "open_streaming_labber_data",
    "open_streaming_grouped_labber_data",
]
