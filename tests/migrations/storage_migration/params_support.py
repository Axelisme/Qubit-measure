"""Synthetic analysis schema and typed storage-migration builders."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import ComponentSchema, ResultEntry
from zcu_tools.resources.predictor_file import PredictorFile, load_predictor_file

from migrations.storage_migration import (
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import noop_validation, validate_cfg


class AnalysisQubit(ComponentSchema):
    """Synthetic qubit with a meta-mapped energy and independent extension data."""

    model_config = ConfigDict(extra="forbid")
    EJ: float | None = None
    freq: float | None = None
    ext: YamlMap = Field(default_factory=dict)
    wiring: dict[str, int] = Field(default_factory=dict)
    module: dict[str, str] = Field(default_factory=dict)
    flux_unit: Literal["A", "V"] | None = None
    flux_half: float | None = None
    flux_period: float | None = None
    flux_int: float | None = None
    flux: float | None = None


def write_params(request: MigrationRequest, values: JsonObject) -> Path:
    """Write one synthetic legacy handoff in the fixture source tree."""
    path = request.result_root / "chip" / "qubit" / "params.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def migrate(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    """Run only parameter migration through its public composition seam."""
    return migrate_storage(
        replace(request, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def open_entry(request: MigrationRequest) -> ResultEntry:
    """Open the synthetic destination with explicit roots."""
    return ResultEntry.open(
        request.name,
        result_root=request.results_root,
        database_root=request.database_root,
    )


def read_predictor(request: MigrationRequest) -> JsonObject:
    """Read the published predictor at the migration's public file boundary."""
    path = request.results_root / request.name / "predictors" / "Qlegacy.json"
    assert path.is_file(), "migrate_storage must publish the entry predictor"
    return TypeAdapter(JsonObject).validate_json(path.read_text(encoding="utf-8"))


def load_migrated_predictor(request: MigrationRequest) -> PredictorFile:
    """Read the migration output through the predictor format owner."""
    return load_predictor_file(
        request.results_root / request.name / "predictors" / "Qlegacy.json"
    )


def backup_path(request: MigrationRequest) -> Path:
    """Locate the explicit synthetic backup; never use production defaults."""
    if request.backup_parent is None:
        raise ValueError("Tests must supply a synthetic backup_parent")
    return request.backup_parent / "repo-backup"


def predictor_params(request: MigrationRequest) -> JsonObject:
    """Return the public predictor parameters as a typed JSON object."""
    return TypeAdapter(JsonObject).validate_python(read_predictor(request)["params"])


def declare_request(
    request: MigrationRequest, **declarations: JsonValue
) -> MigrationRequest:
    """Apply operator input through the request's typed boundary."""
    adapter = TypeAdapter(MigrationRequest)
    return adapter.validate_python({**adapter.dump_python(request), **declarations})
