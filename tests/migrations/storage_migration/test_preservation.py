"""Public converter source-root protection and durable removed-source identity."""

import json
from dataclasses import replace

import pytest

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    migrate_storage,
)
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
    alias.symlink_to(root, target_is_directory=True)
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
            database_alias.symlink_to(database, target_is_directory=True)
            database = database_alias
        request = replace(request, database_root=database)
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
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
    assert not (request.results_root / request.name).exists()
    assert not (request.database_root / request.name).exists()


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
        source.symlink_to(external)
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
