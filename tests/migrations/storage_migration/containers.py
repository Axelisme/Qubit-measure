"""Typed projections of migration-owned JSON documents for contract assertions."""

from dataclasses import dataclass
from pathlib import Path

import h5py
from pydantic import TypeAdapter
from zcu_tools.datafile import Container, JsonObject, load
from zcu_tools.experiment.metadata import CfgSnapshot, RunSnapshot


@dataclass(frozen=True)
class MigrationMetadata:
    """Historical fields asserted by offline migration consumers."""

    run_id: str
    experiment: str
    started_at: str | None
    finished_at: str | None
    completion: str
    snapshot: RunSnapshot
    device_snapshot: dict[str, JsonObject] | None


@dataclass(frozen=True)
class MigratedRun:
    """One generic container plus consumer projections of cfg and metadata docs."""

    container: Container
    cfg: CfgSnapshot
    metadata: MigrationMetadata


def read_migrated(path: Path) -> MigratedRun:
    container = load(path)
    metadata = TypeAdapter(MigrationMetadata).validate_json(
        TypeAdapter(JsonObject).dump_json(container.docs["metadata"])
    )
    cfg = TypeAdapter(CfgSnapshot).validate_json(
        TypeAdapter(JsonObject).dump_json(container.docs["cfg"])
    )
    return MigratedRun(container, cfg, metadata)


def hdf_group(file: h5py.File, name: str) -> h5py.Group:
    node = file[name]
    if not isinstance(node, h5py.Group):
        raise TypeError(f"{name}: expected group")
    return node
