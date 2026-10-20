"""Offline backup and cleanup promises through the public migration seam."""

import hashlib
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone, tzinfo
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.entry import ResultEntry, snapshot

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    MigrationSource,
    migrate_storage,
)
from migrations.storage_migration.backup import create_backup
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
    with pytest.raises(MigrationInputError, match="keep.bin"):
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


@pytest.fixture
def backup_source(request_data: MigrationRequest) -> MigrationSource:
    """Two synthetic source roots, including a file and an empty directory."""
    source = MigrationSource(
        result_path=request_data.result_root / "chip/qubit",
        database_path=request_data.database_root / "chip/qubit",
    )
    (source.result_path / "original.bin").write_bytes(b"original")
    (source.result_path / "empty").mkdir()
    (source.database_path / "database.bin").write_bytes(b"database")
    return source


def test_identical_complete_backup_is_adopted_without_rewriting(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    backup_source: MigrationSource,
) -> None:
    path = backup_path(request_data)
    create_backup(path, backup_source, dry_run=False)
    before = {
        item: (item.read_bytes(), item.stat().st_mtime_ns)
        for item in path.rglob("*")
        if item.is_file()
    }
    try:
        migrate(request_data, mapping)
    except FileExistsError as exc:
        pytest.fail(f"identical verified backup must be adopted: {exc}")
    assert all(
        (item.read_bytes(), item.stat().st_mtime_ns) == state
        for item, state in before.items()
    )
    assert not backup_source.result_path.exists()
    assert not backup_source.database_path.exists()


@pytest.mark.parametrize("with_inventory", [False, True])
def test_partial_backup_fills_only_missing_files_and_directories(
    request_data: MigrationRequest,
    backup_source: MigrationSource,
    with_inventory: bool,
) -> None:
    path = backup_path(request_data)
    if with_inventory:
        create_backup(path, backup_source, dry_run=False)
        (path / "Database/chip/qubit/database.bin").unlink()
        (path / "result/chip/qubit/empty").rmdir()
    else:
        mirror = path / "result/chip/qubit"
        mirror.mkdir(parents=True)
        (mirror / "original.bin").write_bytes(b"original")
    kept = path / "result/chip/qubit/original.bin"
    before = kept.stat().st_mtime_ns
    try:
        backup = create_backup(path, backup_source, dry_run=False)
    except FileExistsError as exc:
        pytest.fail(f"partial matching backup must be completed: {exc}")
    assert backup is not None
    assert kept.stat().st_mtime_ns == before
    assert (path / "Database/chip/qubit/database.bin").read_bytes() == b"database"
    assert (path / "result/chip/qubit/empty").is_dir()
    assert set(backup.inventory_hashes) == {"result", "Database"}
    assert backup_source.result_path.exists()


@pytest.mark.parametrize(
    "conflict",
    [
        "bytes",
        "extra",
        "directory",
        "inventory",
        "invalid-inventory",
        "inventory-directory",
    ],
)
def test_backup_conflict_is_located_and_preflight_prevents_partial_writes(
    request_data: MigrationRequest,
    backup_source: MigrationSource,
    conflict: str,
) -> None:
    path = backup_path(request_data)
    create_backup(path, backup_source, dry_run=False)
    missing = path / "Database/chip/qubit/database.bin"
    missing.unlink()
    target = path / "result/chip/qubit/original.bin"
    if conflict == "bytes":
        target.write_bytes(b"different")
    elif conflict == "extra":
        target = target.with_name("extra.bin")
        target.write_bytes(b"extra")
    elif conflict == "directory":
        target.unlink()
        target.mkdir()
    else:
        target = path / "migration-records/chip/qubit/result.sha256"
        raw = json.loads(target.read_text())
        raw["hashes"]["result/chip/qubit/original.bin"] = "0" * 64
        target.write_text(json.dumps(raw))
        if conflict == "invalid-inventory":
            target.write_bytes(b"incomplete JSON {")
        elif conflict == "inventory-directory":
            target.unlink()
            target.mkdir()
    before = {item: item.read_bytes() for item in path.rglob("*") if item.is_file()}
    with pytest.raises(
        (MigrationInputError, FileExistsError, IsADirectoryError)
    ) as caught:
        create_backup(path, backup_source, dry_run=False)
    assert isinstance(caught.value, MigrationInputError), "must report a data conflict"
    assert caught.value.differences, "conflicts expose typed located comparisons"
    difference = caught.value.differences[0]
    assert difference.destination == target
    assert difference.expected != difference.actual
    assert str(target) in str(caught.value)
    assert not missing.exists(), "all existing items are checked before filling gaps"
    assert {
        item: item.read_bytes() for item in path.rglob("*") if item.is_file()
    } == before


def test_partial_backup_dry_run_validates_without_filling_gaps(
    request_data: MigrationRequest,
    backup_source: MigrationSource,
) -> None:
    path = backup_path(request_data)
    mirror = path / "result/chip/qubit"
    mirror.mkdir(parents=True)
    kept = mirror / "original.bin"
    kept.write_bytes(b"original")
    try:
        result = create_backup(path, backup_source, dry_run=True)
    except FileExistsError as exc:
        pytest.fail(f"dry-run must validate matching partial backup: {exc}")
    assert result is None
    assert sorted(path.rglob("*.bin")) == [kept]
    assert not (path / "Database").exists()
    assert not (path / "migration-records").exists()


@pytest.fixture
def interrupted_initialization(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Interrupt external manifest I/O after publishing the initial entry."""
    target = (
        request_data.results_root / request_data.name / "records/migration-state.json"
    )
    replace_file = os.replace

    def fail_manifest(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == target:
            raise OSError("initial manifest interrupted")
        replace_file(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_manifest)
        with pytest.raises(OSError, match="initial manifest interrupted"):
            migrate(request_data, mapping)
    return (
        backup_path(request_data)
        / "migration-records/chip/qubit/migration-initialization.json"
    )


def test_initial_manifest_interruption_reruns_with_fixed_identity_and_future_fields(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    interrupted_initialization: Path,
) -> None:
    assert interrupted_initialization.is_file(), "ownership must precede entry writes"
    raw = json.loads(interrupted_initialization.read_text())
    identity = raw["manifest"]["identity"]["entry_id"]
    created_at = raw["created_at"]
    report_path = raw["manifest"]["report_path"]
    raw["future"] = {"opaque": [1, "retained"]}
    interrupted_initialization.write_text(json.dumps(raw))
    migrate(request_data, mapping)
    entry = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    assert entry.entry_id == identity
    assert snapshot(entry.setup).general.created_at == created_at
    manifest = json.loads(
        (entry.result_path / "records/migration-state.json").read_text()
    )
    assert manifest["report_path"] == report_path
    assert json.loads(interrupted_initialization.read_text())["future"] == raw["future"]


@pytest.mark.parametrize("window", ["before-entry", "result-rename", "database-rename"])
def test_initialization_windows_resume_without_an_explicit_resume_flag(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    mkdir = Path.mkdir
    rename = Path.rename
    result_target = request_data.results_root / request_data.name
    database_target = request_data.database_root / request_data.name

    def fail_mkdir(
        path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        if window == "before-entry" and path.name.startswith(".migration-init-"):
            raise OSError("initialization interrupted")
        mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    def fail_rename(path: Path, target: str | Path) -> Path:
        if (window == "result-rename" and Path(target) == result_target) or (
            window == "database-rename" and Path(target) == database_target
        ):
            raise OSError("initialization interrupted")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", fail_mkdir)
        patch.setattr(Path, "rename", fail_rename)
        with pytest.raises(OSError, match="initialization interrupted"):
            migrate(request_data, mapping)
    ownership = (
        backup_path(request_data)
        / "migration-records/chip/qubit/migration-initialization.json"
    )
    assert ownership.is_file()
    raw = json.loads(ownership.read_text())
    assert not result_target.exists() or window == "database-rename"
    assert not database_target.exists()
    migrate(request_data, mapping)
    entry = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    assert entry.entry_id == raw["manifest"]["identity"]["entry_id"]
    assert snapshot(entry.setup).general.created_at == raw["created_at"]
    assert not (request_data.results_root / raw["temporary_name"]).exists()
    assert not (request_data.database_root / raw["temporary_name"]).exists()


def test_initial_window_yaml_formatting_is_skipped_without_rewriting(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    interrupted_initialization: Path,
) -> None:
    assert interrupted_initialization.is_file(), "initialization must be owned"
    entry_path = request_data.results_root / request_data.name
    paths = [entry_path / "setup.yaml", entry_path / "module_cfg.yaml"]
    for path in paths:
        raw = YAML(typ="safe").load(path.read_text())
        with path.open("w") as stream:
            YAML(typ="safe").dump(raw, stream)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    # Data-only leaves the initial setup/library unchanged after bootstrap.
    migrate(replace(request_data, part="data"), mapping)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    } == before


@pytest.mark.parametrize(
    "change", ["declaration", "mapping", "identity", "source", "value", "unknown"]
)
def test_initialization_conflicts_leave_all_owned_content_untouched(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    interrupted_initialization: Path,
    change: str,
) -> None:
    assert interrupted_initialization.is_file(), "initialization must be owned"
    request = request_data
    if change == "declaration":
        request = replace(request, assume_flux_unit="mA")
    elif change == "mapping":
        mapping = replace(mapping, mapping_version="1.1")
    elif change == "identity":
        request = replace(request, name="another-entry")
    elif change == "source":
        (request.result_root / "chip/qubit/new.bin").write_bytes(b"new source")
    else:
        path = request.results_root / request.name / "setup.yaml"
        raw = YAML(typ="safe").load(path.read_text())
        if change == "value":
            raw["general"]["description"] = "edited"
        else:
            raw["format_version"] = "2.1"
            raw["future"] = {"unknown": "must not disappear"}
        with path.open("w") as stream:
            YAML(typ="safe").dump(raw, stream)
    roots = [
        request_data.results_root,
        request_data.database_root,
        backup_path(request_data),
    ]
    before = {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(MigrationInputError) as caught:
        migrate(request, mapping)
    assert caught.value.differences, "immutable ownership conflicts must be located"
    after = {
        path: path.read_bytes()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert (request_data.result_root / "chip/qubit").exists()
