"""Destination, report and manifest ownership across migration parts and resumes."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry

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
)

pytestmark = pytest.mark.usefixtures("registry_guard")


@pytest.mark.parametrize("flag, replacement", [(True, 1), (False, 0)])
def test_first_report_boolean_number_collision_is_not_owned(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    flag: bool,
    replacement: int,
) -> None:
    migrate_storage(
        request_data,
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request_data)
    manifest = json.loads(state_path.read_text())
    manifest["report_published"] = False
    manifest["report"]["future_nested"] = {"flags": [flag]}
    state_path.write_text(json.dumps(manifest))
    existing = json.loads(json.dumps(manifest["report"]))
    existing["future_nested"]["flags"][0] = replacement
    report_path = Path(manifest["report_path"])
    report_path.write_text(json.dumps(existing))
    before = report_path.read_bytes()
    with pytest.raises(MigrationInputError, match="unowned report collision"):
        migrate_storage(
            replace(request_data, resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert report_path.read_bytes() == before
    assert (
        json.loads(state_path.read_text())["report"]["future_nested"]["flags"][0]
        is flag
    )


@pytest.mark.parametrize("which", ["result", "database"])
def test_existing_destination_is_never_merged(
    request_data: MigrationRequest, mapping: MigrationMapping, which: str
) -> None:
    target = (
        request_data.results_root if which == "result" else request_data.database_root
    ) / request_data.name
    target.mkdir(parents=True)
    marker = target / "caller.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        migrate_storage(
            request_data,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert marker.read_text() == "unchanged"
    other = (
        request_data.database_root if which == "result" else request_data.results_root
    ) / request_data.name
    assert not other.exists()


def test_resume_refuses_arbitrary_entry(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    ResultEntry.create(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    with pytest.raises(MigrationInputError, match="manifest"):
        migrate_storage(
            replace(request_data, resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )


@pytest.mark.parametrize("first", ["parameters", "data"])
def test_parts_accumulate_with_same_identity(
    request_data: MigrationRequest, mapping: MigrationMapping, first: str
) -> None:
    write_meta(request_data, "context", {"old_frequency": 8.0})
    params = request_data.result_root / "chip" / "qubit" / "params.json"
    params.write_text('{"legacy": true}')
    initial = migrate_storage(
        replace(request_data, part=first),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    second = "data" if first == "parameters" else "parameters"
    result = migrate_storage(
        replace(request_data, part=second, resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert result.entry_id == initial.entry_id
    assert result.part == "all"
    assert len(result.key_mappings) == 1


@pytest.mark.parametrize("change", ["mapping", "source", "report"])
def test_resume_rejects_changed_identity(
    request_data: MigrationRequest, mapping: MigrationMapping, change: str
) -> None:
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    request = replace(request_data, resume=True)
    if change == "mapping":
        mapping = replace(mapping, mapping_version="1.1")
    elif change == "source":
        (request.result_root / "other" / "qubit").mkdir(parents=True)
        (request.database_root / "other" / "qubit").mkdir(parents=True)
        request = replace(request, source_chip="other")
    else:
        request = replace(
            request, report_path=request.result_root.parent / "another-report.json"
        )
    with pytest.raises(MigrationInputError):
        migrate_storage(
            request,
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )


def test_report_rebuild_retains_future_fields(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request_data)
    manifest = json.loads(state_path.read_text())
    manifest["format_version"] = "1.8"
    manifest["future"] = {"nested": [True, 9]}
    manifest["identity"]["future_identity"] = 7
    manifest["report"]["format_version"] = "1.9"
    manifest["report"]["future_report"] = {"kept": True}
    state_path.write_text(json.dumps(manifest))
    report_path = Path(manifest["report_path"])
    report_path.unlink()
    migrate_storage(
        replace(request_data, part="data", resume=True),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    after = json.loads(state_path.read_text())
    report = json.loads(report_path.read_text())
    assert after["future"] == manifest["future"]
    assert after["identity"]["future_identity"] == 7
    assert report["future_report"] == {"kept": True}
    assert report["part"] == "all"


def test_owned_copy_hash_conflict_is_not_repaired(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    source = request_data.result_root / "chip/qubit/image/fit.png"
    source.parent.mkdir()
    source.write_bytes(b"original image")
    report = migrate_storage(
        replace(request_data, part="data"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    destination = report.preserved_files[0].destination
    destination.write_bytes(b"caller edit")
    with pytest.raises(MigrationInputError, match="hash"):
        migrate_storage(
            replace(request_data, resume=True, part="data"),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert destination.read_bytes() == b"caller edit"
    assert source.read_bytes() == b"original image"


def test_existing_independent_report_is_never_overwritten(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    path = request_data.result_root.parent / "caller-report.json"
    path.write_bytes(b"caller file")
    with pytest.raises(FileExistsError):
        migrate_storage(
            replace(request_data, report_path=path),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
    assert path.read_bytes() == b"caller file"
    assert not (request_data.results_root / request_data.name).exists()


@pytest.mark.parametrize("same_content", [True, False])
def test_first_report_publication_window_requires_same_report(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    same_content: bool,
) -> None:
    migrate_storage(
        replace(request_data, part="parameters"),
        mapping=mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    state_path = manifest_path(request_data)
    manifest = json.loads(state_path.read_text())
    manifest["report_published"] = False
    state_path.write_text(json.dumps(manifest))
    report_path = Path(manifest["report_path"])
    if same_content:
        result = migrate_storage(
            replace(request_data, part="data", resume=True),
            mapping=mapping,
            validate_cfg=validate_cfg,
            validate_native=noop_validation,
        )
        assert result.part == "all"
    else:
        report_path.write_text('{"unrelated": true}')
        with pytest.raises(MigrationInputError, match="unowned report"):
            migrate_storage(
                replace(request_data, part="data", resume=True),
                mapping=mapping,
                validate_cfg=validate_cfg,
                validate_native=noop_validation,
            )
        assert report_path.read_text() == '{"unrelated": true}'
