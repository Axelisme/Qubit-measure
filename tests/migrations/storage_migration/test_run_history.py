"""Historical identity and unknown lifecycle at the public migration/native seams."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import DataVariable, JsonObject, RunMetadata, load_run_data
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    ExpCfgModel,
    ZSpec,
    load_run_with_metadata,
)

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    load_run_evidence,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    manifest_path,
    validate_cfg,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


class SyntheticCfg(ExpCfgModel):
    frequency: float


@dataclass
class HistoricalResult:
    x: np.ndarray
    signal: np.ndarray


HISTORY_SPEC = AxesSpec(
    axes=(Axis("x", "x", "s"),),
    z=ZSpec("signal", "signal", "V"),
    result_type=HistoricalResult,
    cfg_type=SyntheticCfg,
    tag="synthetic_scan",
    data_variable=DataVariable("signal"),
)


def history_request(
    request: MigrationRequest,
    changes: JsonObject,
    *,
    missing: tuple[str, ...] = (),
) -> MigrationRequest:
    """Replace selected raw evidence fields, preserving the synthetic source/hash."""
    assert request.run_evidence is not None
    raw = deepcopy(request.run_evidence.raw)
    entries = raw["entries"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    entries[0].update(changes)
    for key in missing:
        entries[0].pop(key, None)
    path = request.result_root.parent / "history-evidence.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        evidence = load_run_evidence(path)
    except MigrationInputError as error:
        pytest.fail(
            f"Missing historical time/ID must not block native migration: {error}"
        )
    return replace(request, run_evidence=evidence)


def read_history(path: Path, tag: str, cfg_type: str) -> RunMetadata:
    """Read through generic and typed owners, checking data and historical identity."""
    stored = load_run_data(path)
    record, metadata = load_run_with_metadata(path, spec=HISTORY_SPEC)
    assert metadata == stored.metadata
    assert metadata.experiment == tag
    assert stored.cfg.cfg_type == cfg_type
    assert record.cfg == SyntheticCfg(frequency=12.5)
    np.testing.assert_array_equal(record.result.x, [0.0, 1.0])
    np.testing.assert_array_equal(record.result.signal, [1 + 2j, 3 + 4j])
    return metadata


@pytest.mark.parametrize("missing", [(), ("started_at", "run_id")])
def test_unknown_start_still_publishes_native_and_reports_assigned_identity(
    request_data: MigrationRequest, mapping: MigrationMapping, missing: tuple[str, ...]
) -> None:
    request, source = write_run(request_data)
    before = source.read_bytes()
    request = history_request(
        request,
        {"started_at": None, "run_id": None, "finished_at": "2025-10-01T00:00:01Z"},
        missing=missing,
    )
    observed: list[RunMetadata] = []

    def validate(path: Path, tag: str, cfg_type: str) -> None:
        assert source.read_bytes() == before
        observed.append(read_history(path, tag, cfg_type))

    report = migrate_storage(
        request, mapping=mapping, validate_cfg=validate_cfg, validate_native=validate
    )
    assert len(report.converted_files) == len(report.moved_labber_files) == 1
    assert len(observed) == 1
    metadata = observed[0]
    assert (metadata.started_at, metadata.completion, metadata.finished_at) == (
        None,
        "unknown",
        None,
    )
    assert metadata.snapshot.entry_id == report.entry_id
    assert metadata.snapshot.entry_name == "historical"
    assert metadata.snapshot.point == "point_at_acquisition"
    assert any(
        item.old_key == "run_id" and "遷移賦值" in item.reason
        for item in report.key_mappings
    )
    assert any(
        item.old_key == "started_at" and "unknown" in item.reason
        for item in report.key_mappings
    )
    assert report.moved_labber_files[0].destination.read_bytes() == before
    assert report.backup_path is not None
    backup_source = report.backup_path / "Database" / "2025" / "10" / "scan.hdf5"
    assert backup_source.read_bytes() == before
    manifest = json.loads(manifest_path(request).read_text())
    assert request.run_evidence is not None
    assert manifest["evidence"] == request.run_evidence.raw
    assert manifest["runs"][0]["run_id"] == metadata.run_id
    assert not source.exists()


def test_trusted_historical_identity_and_times_survive_native_conversion(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, _source = write_run(request_data)
    request = history_request(
        request,
        {
            "run_id": "20251001T000000Z-abc123",
            "started_at": "2025-10-01T00:00:00Z",
            "finished_at": "2025-10-01T00:00:01Z",
            "completion": "complete",
        },
    )
    report = migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=lambda path, tag, cfg_type: (
            read_history(path, tag, cfg_type) and None
        ),
    )
    metadata = load_run_data(report.converted_files[0].destination).metadata
    assert metadata.run_id == "20251001T000000Z-abc123"
    assert metadata.started_at == "2025-10-01T00:00:00Z"
    assert metadata.finished_at == "2025-10-01T00:00:01Z"
    assert metadata.completion == "complete"


def test_unknown_history_resume_reuses_assigned_id_and_published_bytes(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    request, source = write_run(request_data)
    request = history_request(request, {"started_at": None, "completion": "unknown"})
    paths: list[Path] = []

    def interrupt(path: Path, tag: str, cfg_type: str) -> None:
        read_history(path, tag, cfg_type)
        paths.append(path)
        raise RuntimeError("validation interrupted")

    with pytest.raises(RuntimeError, match="validation interrupted"):
        migrate_storage(
            request,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=interrupt,
        )
    before = json.loads(manifest_path(request).read_text())
    native_bytes = paths[0].read_bytes()
    assert source.is_file()

    def validate(path: Path, tag: str, cfg_type: str) -> None:
        assert path == paths[0]
        assert path.read_bytes() == native_bytes
        read_history(path, tag, cfg_type)

    report = migrate_storage(
        replace(request, resume=True, run_evidence=None),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=validate,
    )
    after = json.loads(manifest_path(request).read_text())
    assert after["runs"] == before["runs"]
    assert report.converted_files[0].destination == paths[0]
    assert paths[0].read_bytes() == native_bytes
    assert len(report.moved_labber_files) == 1
    assert not report.failures
    assert not source.exists()


@pytest.mark.parametrize(
    "gap", ["cfg_identity", "cfg_values", "schema", "axis_unit", "signal_name"]
)
def test_unknown_history_does_not_relax_other_required_evidence(
    request_data: MigrationRequest, mapping: MigrationMapping, gap: str
) -> None:
    request, source = write_run(replace(request_data, part="data"))
    changes: JsonObject = {"started_at": None}
    if gap == "cfg_identity":
        changes["cfg"] = {"values": {"frequency": 12.5}, "schema_version": "1.0"}
    elif gap == "cfg_values":
        changes["cfg"] = {
            "values": {},
            "cfg_type": "SyntheticCfg",
            "schema_version": "1.0",
        }
    elif gap == "schema":
        mapping = replace(mapping, data_schemas={})
    else:
        schema = mapping.data_schemas[("synthetic_scan", "SyntheticCfg")][0]
        if gap == "axis_unit":
            schema = replace(schema, axes=(replace(schema.axes[0], unit="Hz"),))
        else:
            schema = replace(schema, signal_name="unproven_signal")
        mapping = replace(
            mapping, data_schemas={("synthetic_scan", "SyntheticCfg"): (schema,)}
        )
    request = history_request(request, changes)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def forbidden(path: Path, tag: str, cfg_type: str) -> None:
        pytest.fail("Incomplete evidence must stay pending before native publication")

    report = migrate_storage(
        request, mapping=mapping, validate_cfg=validate_cfg, validate_native=forbidden
    )
    assert report.converted_files == report.moved_labber_files == ()
    assert any(item.source == source for item in report.pending)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert not tuple((request.database_root / request.name).rglob("data.h5"))
