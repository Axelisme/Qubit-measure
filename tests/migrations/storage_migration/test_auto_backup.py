"""Offline backup and cleanup promises through the public migration seam."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import noop_validation, validate_cfg

pytestmark = pytest.mark.usefixtures("registry_guard")


def migrate(request: MigrationRequest, mapping: MigrationMapping) -> None:
    """Run the public seam using the synthetic registration and validators."""
    migrate_storage(
        request, mapping=mapping, validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def backup_path(request: MigrationRequest) -> Path:
    """Locate the fixture's external backup, never the real checkout parent."""
    return request.result_root.parent.parent / f"{request.name}-backup"


def test_complete_backup_precedes_cleanup_and_resume_needs_no_sources(
    request_data: MigrationRequest, mapping: MigrationMapping,
) -> None:
    old = request_data.result_root / "chip/qubit"
    database = request_data.database_root / "chip/qubit"
    (old / "unknown.bin").write_bytes(b"legacy artifact")
    (database / "notes.txt").write_text("unconverted", encoding="utf-8")
    (old / "empty").mkdir()
    migrate(request_data, mapping)
    backup = backup_path(request_data)
    assert (backup / "result/unknown.bin").read_bytes() == b"legacy artifact"
    assert (backup / "Database/notes.txt").read_text() == "unconverted"
    assert (backup / "result/empty").is_dir()
    hashes = json.loads((backup / "sha256.json").read_text())["hashes"]
    assert hashes["result/unknown.bin"] == hashlib.sha256(b"legacy artifact").hexdigest()
    assert not old.exists()
    assert not database.exists()
    report = json.loads((request_data.results_root / request_data.name / "records/migration-report.json").read_text())
    assert any(row["source"] == str(backup / "result/unknown.bin") for row in report["pending"])
    migrate(replace(request_data, resume=True), mapping)
    assert (backup / "result/unknown.bin").read_bytes() == b"legacy artifact"


def test_dry_run_does_not_backup_or_remove_sources(
    request_data: MigrationRequest, mapping: MigrationMapping,
) -> None:
    migrate(replace(request_data, dry_run=True), mapping)
    assert not backup_path(request_data).exists()
    assert not request_data.results_root.exists()
    assert (request_data.result_root / "chip/qubit").is_dir()
    assert (request_data.database_root / "chip/qubit").is_dir()


def test_parameters_only_keeps_database_and_resume_verifies_backup(
    request_data: MigrationRequest, mapping: MigrationMapping,
) -> None:
    old = request_data.result_root / "chip/qubit/unknown.bin"
    old.write_bytes(b"original")
    migrate(replace(request_data, part="parameters"), mapping)
    backup = backup_path(request_data)
    assert (backup / "result/unknown.bin").read_bytes() == b"original"
    assert old.exists()
    assert (request_data.database_root / "chip/qubit").exists()
    (backup / "result/unknown.bin").write_bytes(b"corrupt")
    with pytest.raises(MigrationInputError, match="backup"):
        migrate(replace(request_data, part="data", resume=True), mapping)
    assert old.exists()


def test_existing_backup_blocks_all_migration_writes(
    request_data: MigrationRequest, mapping: MigrationMapping,
) -> None:
    backup_path(request_data).mkdir()
    with pytest.raises(FileExistsError):
        migrate(request_data, mapping)
    assert not request_data.results_root.exists()


def test_backup_failure_leaves_migration_and_sources_untouched(
    request_data: MigrationRequest, mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = request_data.result_root / "chip/qubit/unknown.bin"
    old.write_bytes(b"original")

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("backup unavailable")

    monkeypatch.setattr(shutil, "copy2", unavailable)
    with pytest.raises(OSError, match="backup unavailable"):
        migrate(request_data, mapping)
    assert not request_data.results_root.exists()
    assert old.read_bytes() == b"original"


@pytest.mark.parametrize("change", ["new", "changed"])
def test_source_changes_since_backup_prevent_cleanup(
    request_data: MigrationRequest, mapping: MigrationMapping, change: str,
) -> None:
    old = request_data.result_root / "chip/qubit/unknown.bin"
    old.write_bytes(b"original")
    migrate(replace(request_data, part="parameters"), mapping)
    assert (backup_path(request_data) / "result/unknown.bin").is_file()
    changed = old if change == "changed" else old.with_name("new.bin")
    changed.write_bytes(b"new data")
    with pytest.raises(MigrationInputError):
        migrate(replace(request_data, part="data", resume=True), mapping)
    assert changed.read_bytes() == b"new data"
    assert (request_data.database_root / "chip/qubit").exists()
