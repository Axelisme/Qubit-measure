"""Public converter source-root protection and durable removed-source identity."""

import json
from dataclasses import replace

import pytest
from zcu_tools.resources.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    migrate_storage,
)

from tests.resources.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
    write_run,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def test_resume_report_inside_old_root_is_refused_before_publication(
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
    forbidden = request_data.result_root / "resume-report.json"
    manifest["report_path"] = str(forbidden)
    manifest_file.write_text(json.dumps(manifest))
    before = manifest_file.read_bytes()
    with pytest.raises(MigrationInputError, match="old result root"):
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
        "same_root",
        "nested_results",
        "results_alias",
        "report",
        "report_alias",
        "nested_database",
        "database_alias",
    ],
)
def test_writes_inside_entire_old_result_root_are_refused(
    request_data: MigrationRequest, mapping: MigrationMapping, placement: str
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    root = request_data.result_root
    alias = root.parent / "result-alias"
    alias.symlink_to(root, target_is_directory=True)
    request = request_data
    if placement == "same_root":
        request = replace(request, results_root=root)
    elif placement == "nested_results":
        request = replace(request, results_root=root / "new-results")
    elif placement == "results_alias":
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
    with pytest.raises(MigrationInputError, match="old result root"):
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


@pytest.mark.parametrize("reuse", ["file", "symlink"])
def test_removed_source_identity_survives_reused_path(
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
    second = migrate_storage(
        replace(request, part="parameters", resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    third = migrate_storage(
        replace(request, part="data", resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert first.entry_id == second.entry_id == third.entry_id
    assert third.converted_files == first.converted_files
    assert third.moved_labber_files == first.moved_labber_files
    assert (
        json.loads(manifest_path(request).read_text())["runs"]
        == manifest_before["runs"]
    )
    assert source.read_bytes() == external.read_bytes() == b"new unrelated acquisition"
    assert source.is_symlink() == (reuse == "symlink")
    third.converted_files[0].destination.write_bytes(b"tampered native")
    with pytest.raises(MigrationInputError, match="destination hash changed"):
        migrate_storage(
            replace(request, part="data", resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert source.read_bytes() == b"new unrelated acquisition"
