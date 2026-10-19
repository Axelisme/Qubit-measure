"""Offline backup and cleanup promises through the public migration seam."""

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone, tzinfo
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
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def backup_path(request: MigrationRequest) -> Path:
    """Locate the fixture's external backup, never the real checkout parent."""
    if request.backup_parent is None or request.repo_root is None:
        raise ValueError("Tests must provide synthetic repo_root and backup_parent")
    return request.backup_parent / f"{request.repo_root.name}-backup"


def test_complete_backup_precedes_cleanup_and_resume_needs_no_sources(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    old = request_data.result_root / "chip/qubit"
    database = request_data.database_root / "chip/qubit"
    (old / "unknown.bin").write_bytes(b"legacy artifact")
    (database / "notes.txt").write_text("unconverted", encoding="utf-8")
    (old / "empty").mkdir()
    migrate(request_data, mapping)
    backup = backup_path(request_data)
    assert backup.is_dir(), "backup root must use the checkout name, not entry name"
    assert (backup / "result/chip/qubit/unknown.bin").read_bytes() == b"legacy artifact"
    assert (backup / "Database/chip/qubit/notes.txt").read_text() == "unconverted"
    assert (backup / "result/chip/qubit/empty").is_dir()
    records = backup / "migration-records/chip/qubit"
    hashes = json.loads((records / "result.sha256").read_text())["hashes"]
    assert (records / "Database.sha256").is_file()
    assert (
        hashes["result/chip/qubit/unknown.bin"]
        == hashlib.sha256(b"legacy artifact").hexdigest()
    )
    assert not old.exists()
    assert not database.exists()
    date = datetime.now(timezone.utc).date().isoformat()
    report_path = records / f"migration-report-{date}.json"
    report = json.loads(report_path.read_text())
    entry_records = request_data.results_root / request_data.name / "records"
    assert json.loads((entry_records / report_path.name).read_text()) == report
    manifest = json.loads((entry_records / "migration-state.json").read_text())
    assert manifest["report_path"] == str(report_path)
    preserved = next(
        row
        for row in report["preserved_files"]
        if row["source"] == str(old / "unknown.bin")
    )
    target = request_data.results_root / request_data.name / "unknown.bin"
    assert preserved["destination"] == str(target)
    assert (
        preserved["source_hash"]
        == preserved["destination_hash"]
        == hashes["result/chip/qubit/unknown.bin"]
    )
    assert target.read_bytes() == b"legacy artifact"
    migrate(replace(request_data, resume=True), mapping)
    assert (backup / "result/chip/qubit/unknown.bin").read_bytes() == b"legacy artifact"


def test_dry_run_does_not_backup_or_remove_sources(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    migrate(replace(request_data, dry_run=True), mapping)
    assert not backup_path(request_data).exists()
    assert not request_data.results_root.exists()
    assert (request_data.result_root / "chip/qubit").is_dir()
    assert (request_data.database_root / "chip/qubit").is_dir()


def test_parameters_only_keeps_database_and_resume_verifies_backup(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    old = request_data.result_root / "chip/qubit/unknown.bin"
    old.write_bytes(b"original")
    migrate(replace(request_data, part="parameters"), mapping)
    backup = backup_path(request_data)
    assert (backup / "result/chip/qubit/unknown.bin").read_bytes() == b"original"
    assert old.exists()
    assert (request_data.database_root / "chip/qubit").exists()
    (backup / "result/chip/qubit/unknown.bin").write_bytes(b"corrupt")
    with pytest.raises(MigrationInputError, match="backup"):
        migrate(replace(request_data, part="data", resume=True), mapping)
    assert old.exists()


@pytest.mark.parametrize("label", ["result", "Database"])
def test_existing_source_destination_blocks_all_migration_writes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    label: str,
) -> None:
    destination = backup_path(request_data) / label / "chip/qubit"
    destination.mkdir(parents=True)
    sentinel = destination / "keep.bin"
    sentinel.write_bytes(b"previous backup")
    with pytest.raises(FileExistsError):
        migrate(request_data, mapping)
    assert sentinel.read_bytes() == b"previous backup"
    assert sorted(destination.iterdir()) == [sentinel]
    assert not request_data.results_root.exists()


def test_copy_verification_failure_does_not_create_entry(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = request_data.result_root / "chip/qubit/unknown.bin"
    old.write_bytes(b"original")
    copy = shutil.copy2

    def corrupt_copy(source: Path, destination: Path) -> Path:
        copied = Path(copy(source, destination))
        copied.write_bytes(b"corrupt copy")
        return copied

    monkeypatch.setattr(shutil, "copy2", corrupt_copy)
    with pytest.raises(MigrationInputError, match="backup verification"):
        migrate(request_data, mapping)
    assert not request_data.results_root.exists()
    assert old.read_bytes() == b"original"
    assert not (request_data.database_root / request_data.name).exists()


def test_backup_failure_leaves_migration_and_sources_untouched(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
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
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    change: str,
) -> None:
    old = request_data.result_root / "chip/qubit/unknown.bin"
    old.write_bytes(b"original")
    migrate(replace(request_data, part="parameters"), mapping)
    assert (backup_path(request_data) / "result/chip/qubit/unknown.bin").is_file()
    changed = old if change == "changed" else old.with_name("new.bin")
    changed.write_bytes(b"new data")
    with pytest.raises(MigrationInputError):
        migrate(replace(request_data, part="data", resume=True), mapping)
    assert changed.read_bytes() == b"new data"
    assert (request_data.database_root / "chip/qubit").exists()


def test_backup_inside_repo_is_refused_before_writes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    request = replace(request_data, backup_parent=request_data.result_root.parent)
    with pytest.raises(MigrationInputError, match="outside repo"):
        migrate(request, mapping)
    assert not request.results_root.exists()
    assert not (request.result_root.parent / "repo-backup").exists()


def test_backup_enclosing_source_is_refused(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    backup = backup_path(request_data)
    request = replace(request_data, result_root=backup / "result")
    old = request.result_root / "chip/qubit"
    old.mkdir(parents=True)
    (old / "data.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(MigrationInputError, match="source/destination"):
        migrate(request, mapping)
    assert (old / "data.txt").read_text() == "keep"
    assert not request.results_root.exists()


def test_cleanup_interruption_resumes_only_after_backup_verification(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = request_data.result_root / "chip/qubit"
    first, second = old / "a.bin", old / "b.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    unlink = Path.unlink

    def interrupted(path: Path, missing_ok: bool = False) -> None:
        if path == second:
            raise OSError("cleanup interrupted")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(OSError, match="cleanup interrupted"):
            migrate(request_data, mapping)
    assert not first.exists()
    assert second.read_bytes() == b"second"
    backup = backup_path(request_data)
    (backup / "result/chip/qubit/a.bin").write_bytes(b"bad backup")
    with pytest.raises(MigrationInputError, match="backup"):
        migrate(replace(request_data, resume=True), mapping)
    assert second.read_bytes() == b"second"
    (backup / "result/chip/qubit/a.bin").write_bytes(b"first")
    migrate(replace(request_data, resume=True), mapping)
    assert not old.exists()
    assert not (request_data.database_root / "chip/qubit").exists()


def test_shared_backup_accumulates_independent_sources_and_resumes_each(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    requests = [request_data]
    for chip, qubit, name in (
        ("chip", "second", "second-entry"),
        ("other-chip", "qubit", "third-entry"),
    ):
        request = replace(request_data, source_chip=chip, source_qubit=qubit, name=name)
        for root in (request.result_root, request.database_root):
            (root / chip / qubit).mkdir(parents=True)
        requests.append(request)
    snapshots: dict[Path, bytes] = {}
    for request in requests:
        source = request.result_root / request.source_chip / request.source_qubit
        (source / "original.bin").write_bytes(request.name.encode())
        migrate(request, mapping)
        backup = backup_path(request)
        assert backup.is_dir(), "independent migrations share the checkout backup root"
        for path in backup.rglob("*"):
            if path.is_file() and not path.name.startswith("migration-report-"):
                snapshots.setdefault(path, path.read_bytes())
        for path, original in snapshots.items():
            assert path.read_bytes() == original
        assert (
            backup
            / "result"
            / request.source_chip
            / request.source_qubit
            / "original.bin"
        ).read_bytes() == request.name.encode()
    for request in requests:
        migrate(replace(request, resume=True), mapping)
    assert all(path.read_bytes() == original for path, original in snapshots.items())


@pytest.mark.parametrize("label", ["result", "Database"])
def test_resume_rejects_changed_hash_list_without_replacing_it(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    label: str,
) -> None:
    migrate(replace(request_data, part="parameters"), mapping)
    inventory = (
        backup_path(request_data) / "migration-records/chip/qubit" / f"{label}.sha256"
    )
    assert inventory.is_file(), "each mirror requires its bound hash list"
    changed = inventory.read_bytes() + b"\n"
    inventory.write_bytes(changed)
    with pytest.raises(MigrationInputError, match="inventory"):
        migrate(replace(request_data, part="data", resume=True), mapping)
    assert inventory.read_bytes() == changed
    assert (request_data.result_root / "chip/qubit").is_dir()


def test_report_path_is_fixed_on_resume_and_copied_only_after_completion(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migrate(replace(request_data, part="parameters"), mapping)
    records = backup_path(request_data) / "migration-records/chip/qubit"
    reports = list(records.glob("migration-report-*.json"))
    assert len(reports) == 1, "the automatic report belongs beside the hash lists"
    path = reports[0]
    entry_records = request_data.results_root / request_data.name / "records"
    assert list(entry_records.glob("migration-report-*.json")) == []

    class LaterDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return super().now(tz).replace(year=2040, month=1, day=1)

    monkeypatch.setattr(
        "migrations.storage_migration.bootstrap.datetime", LaterDatetime
    )
    migrate(replace(request_data, part="data", resume=True), mapping)
    assert list(records.glob("migration-report-*.json")) == [path]
    entry_copy = entry_records / path.name
    assert entry_copy.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("label", ["result", "Database"])
def test_explicit_report_cannot_write_inside_backup_mirror(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    label: str,
) -> None:
    parent = backup_path(request_data) / label / "unrelated"
    parent.mkdir(parents=True)
    request = replace(request_data, report_path=parent / "report.json")
    with pytest.raises(MigrationInputError, match="report"):
        migrate(request, mapping)
    assert list(parent.iterdir()) == []
    assert not request.results_root.exists()


def test_explicit_external_report_is_published_and_preserved_on_resume(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    path = request_data.result_root.parent.parent / "chosen-report.json"
    request = replace(request_data, report_path=path)
    migrate(request, mapping)
    first = path.read_bytes()
    assert json.loads(first)["backup_path"] == str(backup_path(request))
    migrate(replace(request, resume=True), mapping)
    assert path.read_bytes() == first


@pytest.mark.parametrize("label", ["result", "Database", "migration-records"])
def test_shared_backup_symlink_ancestor_is_refused_without_escape_writes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    label: str,
) -> None:
    backup = backup_path(request_data)
    backup.mkdir()
    outside = backup.parent / "unowned"
    outside.mkdir()
    (backup / label).symlink_to(outside, target_is_directory=True)
    with pytest.raises(MigrationInputError, match="symlink"):
        migrate(request_data, mapping)
    assert list(outside.iterdir()) == []
    assert not request_data.results_root.exists()


def test_custom_source_roots_keep_logical_mirrors(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
) -> None:
    request = replace(
        request_data,
        result_root=request_data.result_root.parent.parent / "custom-results",
        database_root=request_data.database_root.parent.parent / "custom-database",
    )
    for root in (request.result_root, request.database_root):
        source = root / "chip/qubit"
        source.mkdir(parents=True)
        (source / "empty").mkdir()
        (source / "source.bin").write_bytes(root.name.encode())
    migrate(request, mapping)
    for label, root in (
        ("result", request.result_root),
        ("Database", request.database_root),
    ):
        mirror = backup_path(request) / label / "chip/qubit"
        assert mirror.is_dir(), "custom roots use the same logical mirror paths"
        assert (mirror / "source.bin").read_bytes() == root.name.encode()
        assert (mirror / "empty").is_dir()
