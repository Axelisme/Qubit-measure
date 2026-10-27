"""Synthetic native containers for standalone analysis input contracts."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
from zcu_tools.datafile import (
    Axis,
    Container,
    ContainerAttrs,
    JsonObject,
    Variable,
    save,
)
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance


def native_spectrum_file(path: Path, snapshot: dict[str, JsonObject] | None) -> Path:
    """Save a 2-D complex native sweep with typed acquisition metadata."""
    metadata = RunMetadata(
        run_id="20261027T120000Z-abc123",
        experiment="synthetic",
        started_at=None,
        finished_at=None,
        completion="unknown",
        snapshot=RunSnapshot(entry_id=UUID(int=1), point=None, roles={}),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
        device_snapshot=snapshot,
    )
    save(
        path,
        Container(
            vars={
                "signal": Variable(
                    np.array([[1j, 2j, 3j], [4j, 5j, 6j]]),
                    (
                        Axis("bias", "A", np.array([0.001, 0.002])),
                        Axis("frequency", "MHz", np.array([4000.0, 5000.0, 6000.0])),
                    ),
                    "a.u.",
                )
            },
            docs={"metadata": metadata.model_dump(mode="json")},
            groups={},
            attrs=ContainerAttrs("3.0", "test", None, {}),
        ),
    )
    return path
