from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path

import h5py
from zcu_tools.utils.datasaver import (
    DatasetRole,
    GroupedLabberData,
    LabberMetadata,
    LabberPayload,
)
from zcu_tools.utils.datasaver.labber import _str_array, _write_payload_to_log


def write_grouped_v1(
    path: Path,
    roles: Mapping[str | DatasetRole, LabberPayload],
    *,
    metadata: LabberMetadata | None = None,
) -> Path:
    """Write the retired root/Log_N grouped-v1 wire format for migration tests."""
    grouped = GroupedLabberData(roles, metadata=metadata)
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
    log_name = os.path.splitext(path.name)[0]
    role_items = list(grouped.roles.items())

    with h5py.File(path, "x") as file:
        for index, (role, payload) in enumerate(role_items):
            target = file if index == 0 else file.create_group(f"Log_{index + 1}")
            _write_payload_to_log(
                target,
                payload,
                effective_metadata,
                log_name=log_name,
                creation_time=creation_time,
                write_tags=index == 0,
            )
            target.attrs["zcu_tools.dataset_role"] = str(role)

        file.attrs["zcu_tools.grouped_dataset_version"] = 1
        file.attrs["zcu_tools.dataset_roles"] = _str_array(
            [str(role) for role, _payload in role_items]
        )

    return path
