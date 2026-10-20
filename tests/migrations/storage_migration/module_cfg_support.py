"""Synthetic module libraries and typed storage-migration boundaries."""

import json
from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import ResultEntry

from migrations.storage_migration import (
    MigrationManifest,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
)


def library(meta: Path, values: YamlMap) -> Path:
    """Write a legacy module library beside a synthetic metadata file."""
    source = meta.parent / "module_cfg.yaml"
    source.write_text(json.dumps(values), encoding="utf-8")
    return source


def pulse(waveform: str = "flat") -> YamlMap:
    """Build a supported legacy pulse referencing the given waveform key."""
    return {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 123.5,
        "gain": 0.25,
        "waveform": waveform,
    }


def payload() -> YamlMap:
    """Build a library with pulse, readout and nested waveform references."""
    return {
        "modules": {
            "same": pulse(),
            "readout": {
                "type": "readout/pulse",
                "gen_ch": 0,
                "nqz": 1,
                "freq": 123.5,
                "gain": 0.25,
                "waveform": "flat",
                "ro_ch": 0,
                "ro_length": 2.5,
            },
        },
        "waveforms": {
            "same": {"style": "const", "length": 0.125},
            "ramp": {"style": "gauss", "length": 0.05, "sigma": 0.01},
            "flat": {"style": "flat_top", "length": 0.25, "raise_waveform": "ramp"},
        },
    }


def convert(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    """Run storage migration with the synthetic config and native validators."""
    return migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def entry(request: MigrationRequest) -> ResultEntry:
    """Open the synthetic destination with explicit result and database roots."""
    return ResultEntry.open(
        request.name,
        result_root=request.results_root,
        database_root=request.database_root,
    )


def manifest(request: MigrationRequest) -> MigrationManifest:
    """Read the durable manifest through its typed public contract."""
    return TypeAdapter(MigrationManifest).validate_json(
        manifest_path(request).read_bytes()
    )
