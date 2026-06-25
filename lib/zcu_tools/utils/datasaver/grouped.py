"""Grouped Labber dataset persistence."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np

from .labber import (
    _all_log_refs,
    _decode,
    _read_log_label,
    _read_single_log,
    _read_tags,
    _resolve_path,
    _str_array,
    _write_payload_to_log,
)
from .models import (
    Axis,
    DatasetRole,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
)
from .paths import format_ext

GROUPED_DATASET_VERSION = 1
GROUPED_VERSION_ATTR = "zcu_tools.grouped_dataset_version"
DATASET_ROLES_ATTR = "zcu_tools.dataset_roles"
DATASET_ROLE_ATTR = "zcu_tools.dataset_role"


def save_grouped_labber_data(
    path: str,
    roles: Mapping[str | DatasetRole, LabberPayload],
    *,
    metadata: LabberMetadata | None = None,
) -> str:
    """Save role payloads into one Labber-compatible HDF5 file."""
    grouped = GroupedLabberData(roles, metadata=metadata)
    metadata = grouped.metadata
    creation_time = (
        time.time() if metadata.creation_time is None else float(metadata.creation_time)
    )
    effective_metadata = LabberMetadata(
        comment=metadata.comment,
        tags=metadata.tags,
        project=metadata.project,
        user=metadata.user,
        creation_time=creation_time,
    )

    path = format_ext(path)
    log_name = os.path.splitext(os.path.basename(path))[0]
    role_items = list(grouped.roles.items())
    role_names = [str(role) for role, _payload in role_items]

    with h5py.File(path, "x") as f:
        for index, (role, payload) in enumerate(role_items):
            target: h5py.File | h5py.Group
            if index == 0:
                target = f
            else:
                target = f.create_group(f"Log_{index + 1}")

            _write_payload_to_log(
                target,
                payload,
                effective_metadata,
                log_name=log_name,
                creation_time=creation_time,
                write_tags=index == 0,
            )
            target.attrs[DATASET_ROLE_ATTR] = str(role)

        f.attrs[GROUPED_VERSION_ATTR] = GROUPED_DATASET_VERSION
        f.attrs[DATASET_ROLES_ATTR] = _str_array(role_names)

    return path


def load_grouped_labber_data(
    path: str,
    *,
    required_roles: Sequence[str | DatasetRole] | None = None,
) -> GroupedLabberData:
    """Load a grouped Labber dataset.

    Passing ``required_roles`` enables strict mode: missing and unknown roles
    raise. Omitting it is diagnostic mode and returns every role after validating
    role format and duplicates.
    """
    path = _resolve_path(path)

    with h5py.File(path, "r") as f:
        version = f.attrs.get(GROUPED_VERSION_ATTR)
        if version is None:
            raise ValueError("file is not a grouped Labber dataset")
        if int(version) != GROUPED_DATASET_VERSION:
            raise ValueError(f"unsupported grouped dataset version {version!r}")

        declared_roles = _read_declared_roles(f)
        logs = _all_log_refs(f)
        if len(declared_roles) != len(logs):
            raise ValueError("grouped dataset role list does not match log group count")

        comment = _decode(f.attrs.get("comment", "")) or ""
        tags, project, user = _read_tags(f)
        creation_time = float(f.attrs.get("creation_time", 0.0) or 0.0)
        metadata = LabberMetadata(
            comment=comment,
            tags=tags,
            project=project,
            user=user,
            creation_time=creation_time,
        )

        payloads: dict[DatasetRole, LabberPayload] = {}
        seen_from_logs: list[DatasetRole] = []
        for log in logs:
            role = _read_log_role(log)
            if role in payloads:
                raise ValueError(f"duplicate dataset role {role!r}")
            z, axes, ts_rel = _read_single_log(f, log)
            z_name, z_unit = _read_log_label(f, log)
            timestamps = None if ts_rel is None else creation_time + np.asarray(ts_rel)
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
        {str(role): payload for role, payload in payloads.items()},
        metadata=metadata,
    )


def _read_declared_roles(f: h5py.File) -> list[DatasetRole]:
    raw = _decode(f.attrs.get(DATASET_ROLES_ATTR))
    if raw is None:
        raise ValueError("grouped dataset is missing dataset role list")
    if isinstance(raw, str):
        values: list[Any] = [raw]
    else:
        values = list(raw)

    roles: list[DatasetRole] = []
    seen: set[DatasetRole] = set()
    for value in values:
        role = DatasetRole(value)
        if role in seen:
            raise ValueError(f"duplicate dataset role {role!r}")
        seen.add(role)
        roles.append(role)
    return roles


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
