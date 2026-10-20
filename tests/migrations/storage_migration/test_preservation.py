"""Public converter source-root protection and durable removed-source identity."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.datafile import load_run_data
from zcu_tools.resources.entry import ResultEntry

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
from tests._filesystem import symlink_or_skip
from tests.migrations.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def test_resume_report_inside_source_is_refused_before_publication(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    migrate_storage(
        request_data,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    manifest_file = manifest_path(request_data)
    manifest = json.loads(manifest_file.read_text())
    forbidden = request_data.result_root / "chip/qubit/resume-report.json"
    manifest["report_path"] = str(forbidden)
    manifest_file.write_text(json.dumps(manifest))
    before = manifest_file.read_bytes()
    with pytest.raises(MigrationInputError, match="source"):
        migrate_storage(
            replace(request_data, resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert not forbidden.exists()
    assert manifest_file.read_bytes() == before


@pytest.mark.parametrize("source_root", ["result", "Database"])
@pytest.mark.parametrize("link_segment", ["chip", "qubit"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_selected_source_symlink_is_refused_without_writes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    source_root: str,
    link_segment: str,
    dry_run: bool,
) -> None:
    roots = (request_data.result_root, request_data.database_root)
    for root in roots:
        (root / "chip/qubit/retained.bin").write_bytes(b"legacy bytes")
    root = roots[0] if source_root == "result" else roots[1]
    link = root / "chip" if link_segment == "chip" else root / "chip/qubit"
    target = root / "actual-source"
    link.rename(target)
    symlink_or_skip(link, target, target_is_directory=True)
    linked_file = root / "chip/qubit/retained.bin"
    target_file = target / (
        "qubit/retained.bin" if link_segment == "chip" else "retained.bin"
    )
    before = {
        path: path.read_bytes()
        for source in roots
        for path in source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(MigrationInputError, match="symlink"):
        migrate_storage(
            replace(request_data, dry_run=dry_run),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert link.is_symlink()
    assert link.readlink() == target
    assert linked_file.read_bytes() == target_file.read_bytes() == b"legacy bytes"
    assert {path: path.read_bytes() for path in before} == before
    assert not (request_data.results_root / request_data.name).exists()
    assert not (request_data.database_root / request_data.name).exists()
    assert not (root.parent.parent / f"{request_data.name}-backup").exists()


@pytest.mark.parametrize(
    "placement",
    [
        "inside_source",
        "enclosing_source",
        "source_alias",
        "report",
        "report_alias",
        "nested_database",
        "database_alias",
    ],
)
def test_writes_overlapping_sources_are_refused(
    request_data: MigrationRequest, mapping: MigrationMapping, placement: str
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    root = request_data.result_root / "chip/qubit"
    alias = request_data.result_root.parent / "result-alias"
    if placement in ("source_alias", "report_alias"):
        symlink_or_skip(alias, root, target_is_directory=True)
    request = request_data
    if placement == "inside_source":
        request = replace(request, results_root=root)
    elif placement == "enclosing_source":
        request = replace(request, results_root=request.result_root, name="chip")
    elif placement == "source_alias":
        request = replace(request, results_root=alias)
    elif placement in ("report", "report_alias"):
        request = replace(
            request,
            report_path=(root if placement == "report" else alias) / "migration.json",
        )
    else:
        database = root / "Database"
        (database / "chip" / "qubit").mkdir(parents=True)
        (database / "chip" / "qubit" / "retained.txt").write_bytes(b"legacy bytes")
        if placement == "database_alias":
            database_alias = root.parent / "database-alias"
            symlink_or_skip(database_alias, database, target_is_directory=True)
            database = database_alias
        request = replace(request, database_root=database)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    destinations = (
        request.results_root / request.name,
        request.database_root / request.name,
    )
    existed = tuple(path.exists() for path in destinations)
    destination_bytes = tuple(
        {
            path.relative_to(target): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }
        for target in destinations
    )
    with pytest.raises(MigrationInputError, match="source"):
        migrate_storage(
            request,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == before
    assert tuple(path.exists() for path in destinations) == existed
    assert (
        tuple(
            {
                path.relative_to(target): path.read_bytes()
                for path in target.rglob("*")
                if path.is_file()
            }
            for target in destinations
        )
        == destination_bytes
    )


@pytest.mark.parametrize("resume", [False, True])
def test_same_result_root_creates_independent_entry_and_keeps_other_entries(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    resume: bool,
) -> None:
    request = replace(request_data, results_root=request_data.result_root)
    write_meta(request, "context", {"old_frequency": 8.0})
    unrelated = request.result_root / "existing-entry" / "keep.bin"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"unrelated entry")
    if resume:
        migrate_storage(
            replace(request, part="data"),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    report = migrate_storage(
        replace(request, resume=resume),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert report.destination.result_path == request.result_root / request.name
    assert (report.destination.result_path / "points/context/point.yaml").is_file()
    assert unrelated.read_bytes() == b"unrelated entry"
    assert not (request.result_root / "chip/qubit").exists()
    assert not (request.database_root / "chip/qubit").exists()


@pytest.mark.parametrize("reuse", ["file", "symlink"])
def test_removed_source_reuse_blocks_resume_without_deleting_new_data(
    request_data: MigrationRequest, mapping: MigrationMapping, reuse: str
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    request, source = write_run(request_data)
    first = migrate_storage(
        replace(request, part="data"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    manifest_before = json.loads(manifest_path(request).read_text())
    external = request.result_root.parent / "new-acquisition.hdf5"
    external.write_bytes(b"new unrelated acquisition")
    if reuse == "symlink":
        symlink_or_skip(source, external)
    else:
        source.write_bytes(external.read_bytes())
    for part in ("parameters", "data"):
        with pytest.raises(MigrationInputError):
            migrate_storage(
                replace(request, part=part, resume=True),
                mapping=mapping,
                validate_cfg=validate_cfg,
                validate_native=noop_validation,
            )
    assert first.converted_files[0].destination.is_file()
    assert first.moved_labber_files[0].destination.is_file()
    assert (
        json.loads(manifest_path(request).read_text())["runs"]
        == manifest_before["runs"]
    )
    assert source.read_bytes() == external.read_bytes() == b"new unrelated acquisition"
    assert source.is_symlink() == (reuse == "symlink")


def _convert(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    return migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )


def _entry(request: MigrationRequest) -> ResultEntry:
    return ResultEntry.open(
        request.name,
        result_root=request.results_root,
        database_root=request.database_root,
    )


def _write(path: Path, content: bytes = b"unconverted\x00\xff\r\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _snapshot(root: Path) -> dict[Path, bytes | None]:
    return {
        path.relative_to(root): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_unsupported_sources_publish_at_entry_and_point_locations_with_exact_hashes(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    old_result = request_data.result_root / "chip/qubit"
    old_database = request_data.database_root / "chip/qubit"
    write_meta(request_data, "exps/context", {})
    paths = {
        _write(old_result / "notes/readme.bin"): "notes/readme.bin",
        _write(
            old_result / "autofluxdep_runs/run/image/fit.png"
        ): "autofluxdep_runs/run/image/fit.png",
        _write(
            old_result / "exps/context/notes/readme.bin"
        ): "points/context/notes/readme.bin",
        _write(
            old_result / "exps/context/image/raw.svg"
        ): "points/context/image/raw.svg",
        _write(old_database / "notes/readme.bin"): "notes/readme.bin",
    }
    before = _snapshot(request_data.repo_root or old_result)
    dry = _convert(replace(request_data, dry_run=True), mapping)
    assert _snapshot(request_data.repo_root or old_result) == before
    assert not (request_data.result_root.parent.parent / "repo-backup").exists()
    assert dry.converted_files == dry.preserved_files == ()
    report = _convert(request_data, mapping)
    entry = _entry(request_data)
    for source, relative in paths.items():
        target = (
            entry.database_path
            if source.is_relative_to(old_database)
            else entry.result_path
        ) / relative
        assert target.is_file(), f"Preserve {source} at {target}"
        assert target.read_bytes() == b"unconverted\x00\xff\r\n"
        item = next(item for item in report.preserved_files if item.source == source)
        assert item.destination == target
        assert (
            item.source_hash
            == item.destination_hash
            == hashlib.sha256(target.read_bytes()).hexdigest()
        )
        assert item.status == "preserved"
    assert report.converted_files == ()
    assert not old_result.exists()
    assert not old_database.exists()
    assert (
        _convert(replace(request_data, resume=True), mapping).preserved_files
        == report.preserved_files
    )


@pytest.mark.parametrize("native_success", [True, False])
def test_all_labber_originals_share_one_layout_independent_of_native_success(
    request_data: MigrationRequest, mapping: MigrationMapping, native_success: bool
) -> None:
    request, source = write_run(
        request_data, cfg_values=None if native_success else {"missing_frequency": 1}
    )
    original = source.read_bytes()
    report = _convert(request, mapping)
    target = _entry(request).database_path / "Labber/2025/10/scan.hdf5"
    assert target.is_file(), "Native failure must not change the original Labber layout"
    assert target.read_bytes() == original
    copies = (*report.preserved_files, *report.moved_labber_files)
    item = next(item for item in copies if item.source == source)
    assert item.destination == target
    assert (
        item.source_hash
        == item.destination_hash
        == hashlib.sha256(original).hexdigest()
    )
    if native_success:
        stored = load_run_data(report.converted_files[0].destination)
        assert stored.metadata.labber_path == "Labber/2025/10/scan.hdf5"
    else:
        assert report.converted_files == ()
        assert any(item.location == "run evidence" for item in report.pending)
    _convert(replace(request, resume=True), mapping)
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    "root,relative",
    [
        ("result", "setup.yaml"),
        ("result", "setup.yaml/child.bin"),
        ("result", "points"),
        ("result", "points/context/blob.bin"),
        ("result", "records/unknown.bin"),
        ("result", "runs/unknown.bin"),
        ("result", "samples/unknown.bin"),
        ("result", "predictors/unknown.bin"),
        ("result", "exps/context/point.yaml"),
        ("result", "exps/context/figures/unknown.bin"),
        ("Database", "runs/unknown.bin"),
        ("Database", "Labber/unknown.bin"),
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_reserved_destinations_are_refused_with_source_and_target_before_writes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    root: str,
    relative: str,
    dry_run: bool,
) -> None:
    old_root = (
        request_data.result_root if root == "result" else request_data.database_root
    )
    source = _write(old_root / "chip/qubit" / relative)
    base = request_data.results_root if root == "result" else request_data.database_root
    destination = base / request_data.name / relative
    if relative.startswith("exps/"):
        destination = (
            base / request_data.name / "points" / relative.removeprefix("exps/")
        )
    before = _snapshot(request_data.repo_root or old_root)
    with pytest.raises((MigrationInputError, FileExistsError)) as caught:
        _convert(replace(request_data, dry_run=dry_run), mapping)
    assert str(source) in str(caught.value)
    assert str(destination) in str(caught.value)
    assert _snapshot(request_data.repo_root or old_root) == before
    assert not (request_data.result_root.parent.parent / "repo-backup").exists()


@pytest.mark.parametrize("kind", ["same", "ancestor", "descendant"])
def test_source_routes_cannot_claim_the_same_or_nested_destination(
    request_data: MigrationRequest, mapping: MigrationMapping, kind: str
) -> None:
    root = request_data.result_root / "chip/qubit/exps/context"
    first = _write(root / "image/fit.png")
    if kind == "same":
        second = _write(root / "figures/fit.png")
    elif kind == "ancestor":
        second = _write(root / "figures")
    else:
        second = _write(root / "figures/fit.png/child.bin")
    before = _snapshot(request_data.repo_root or root)
    with pytest.raises((MigrationInputError, FileExistsError)) as caught:
        _convert(request_data, mapping)
    assert str(first) in str(caught.value)
    assert str(second) in str(caught.value)
    assert str(request_data.name) in str(caught.value)
    assert _snapshot(request_data.repo_root or root) == before


def test_missing_point_defers_unknown_files_and_blocks_cleanup_until_data_resume(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = _write(request_data.result_root / "chip/qubit/exps/context/notes/blob.bin")
    report = _convert(replace(request_data, part="data"), mapping)
    entry = _entry(request_data)
    target = entry.result_path / "points/context/notes/blob.bin"
    assert not target.parent.parent.exists()
    assert any(
        item.source == source and "point" in item.reason.lower()
        for item in report.pending
    )
    _convert(replace(request_data, part="parameters", resume=True), mapping)
    assert _entry(request_data).list_points() == ["context"]
    assert source.is_file(), (
        "Deferred data publication must block cleanup after parameters"
    )
    assert not target.exists()
    completed = _convert(replace(request_data, part="data", resume=True), mapping)
    assert target.read_bytes() == b"unconverted\x00\xff\r\n"
    assert not any(item.source == source for item in completed.pending)
    assert not (request_data.result_root / "chip/qubit").exists()


@pytest.mark.parametrize("occupation", ["file", "directory", "ancestor"])
def test_resume_never_adopts_unowned_existing_destinations_even_with_equal_bytes(
    request_data: MigrationRequest, mapping: MigrationMapping, occupation: str
) -> None:
    source = _write(request_data.database_root / "chip/qubit/notes/blob.bin")
    _convert(replace(request_data, part="parameters"), mapping)
    target = _entry(request_data).database_path / "notes/blob.bin"
    if occupation == "file":
        _write(target, source.read_bytes())
    elif occupation == "directory":
        target.mkdir(parents=True)
        _write(target / "child.bin")
    else:
        _write(target.parent)
    before = _snapshot(request_data.repo_root or source.parent)
    with pytest.raises((MigrationInputError, FileExistsError)) as caught:
        _convert(replace(request_data, part="data", resume=True), mapping)
    assert str(source) in str(caught.value)
    assert str(target) in str(caught.value)
    assert _snapshot(request_data.repo_root or source.parent) == before


def test_preservation_hash_mismatch_is_refused_without_overwriting(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    _write(request_data.database_root / "chip/qubit/notes/blob.bin")
    _convert(replace(request_data, part="data"), mapping)
    target = _entry(request_data).database_path / "notes/blob.bin"
    assert target.is_file()
    target.write_bytes(b"caller edit")
    before = _snapshot(request_data.repo_root or target.parent)
    with pytest.raises(MigrationInputError, match="hash"):
        _convert(replace(request_data, part="data", resume=True), mapping)
    assert _snapshot(request_data.repo_root or target.parent) == before


def test_interrupted_unknown_file_publication_prevents_cleanup_and_resumes(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write(request_data.database_root / "chip/qubit/notes/blob.bin")
    target = request_data.database_root / request_data.name / "notes/blob.bin"
    link = os.link

    def interrupt(src: Path, dst: Path) -> None:
        if dst == target:
            raise OSError("copy publication interrupted")
        link(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupt)
        with pytest.raises(OSError, match="copy publication interrupted"):
            _convert(request_data, mapping)
    assert source.is_file()
    _convert(replace(request_data, resume=True), mapping)
    assert target.read_bytes() == b"unconverted\x00\xff\r\n"
    assert not source.exists()


@pytest.mark.parametrize(
    "occupation", ["incomplete_point", "point_file", "target_file"]
)
def test_resume_point_occupancy_reports_source_and_target_before_any_writes(
    request_data: MigrationRequest, mapping: MigrationMapping, occupation: str
) -> None:
    source = _write(request_data.result_root / "chip/qubit/exps/context/notes/blob.bin")
    request = replace(request_data, part="data")
    _convert(request, mapping)
    target = _entry(request).result_path / "points/context/notes/blob.bin"
    point = target.parent.parent
    if occupation == "incomplete_point":
        point.mkdir()
    elif occupation == "point_file":
        _write(point)
    else:
        _write(target)
    before = _snapshot(request_data.result_root.parent)
    with pytest.raises((ValueError, OSError)) as caught:
        _convert(replace(request, resume=True), mapping)
    assert str(source) in str(caught.value)
    assert str(target) in str(caught.value)
    assert _snapshot(request_data.result_root.parent) == before
