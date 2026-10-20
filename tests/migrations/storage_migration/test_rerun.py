"""Owned reruns compare fixed outputs; remove with the offline migration tool."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.entry import ResultEntry, source_path

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import noop_validation, write_meta

pytestmark = pytest.mark.usefixtures("registry_guard")


def convert(request: MigrationRequest, mapping: MigrationMapping) -> MigrationReport:
    return migrate_storage(
        request,
        mapping=mapping,
        validate_cfg=lambda _tag, _cfg: None,
        validate_native=noop_validation,
    )


def files_under(*roots: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_completed_rerun_verifies_without_writing(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    resume: bool,
    dry_run: bool,
) -> None:
    write_meta(request_data, "", {"old_frequency": 3.0, "raw": {"x": 4}})
    write_meta(request_data, "exps/bias", {"old_frequency": 4.0})
    raw = request_data.result_root / "chip/qubit/opaque.bin"
    raw.write_bytes(b"unaltered")
    first = convert(request_data, mapping)
    before = files_under(first.destination.result_path, first.destination.database_path)
    try:
        report = convert(replace(request_data, resume=resume, dry_run=dry_run), mapping)
    except MigrationInputError as error:
        pytest.fail(f"An owned identical rerun must succeed: {error}")
    assert report.entry_id == first.entry_id
    assert {
        first.destination.result_path / name
        for name in ("setup.yaml", "points/bias/point.yaml", "opaque.bin")
    } <= {row.destination for row in report.skipped}
    assert (
        files_under(first.destination.result_path, first.destination.database_path)
        == before
    )


@pytest.mark.parametrize("point", [False, True])
def test_legal_completed_edit_is_located_and_does_not_change_reports(
    request_data: MigrationRequest, mapping: MigrationMapping, point: bool
) -> None:
    write_meta(request_data, "exps/bias", {"old_frequency": 4.0})
    first = convert(request_data, mapping)
    entry = ResultEntry.open(
        request_data.name,
        result_root=request_data.results_root,
        database_root=request_data.database_root,
    )
    view = entry.use_point("bias") if point else entry.setup
    with view.edit() as draft:
        draft.set("C1.frequency", 18.0)
    target = entry.result_path / ("points/bias/point.yaml" if point else "setup.yaml")
    assert first.backup_path is not None
    before = files_under(entry.result_path, entry.database_path, first.backup_path)
    with pytest.raises(MigrationInputError) as caught:
        convert(replace(request_data, resume=True), mapping)
    assert any(
        row.destination == target and row.expected != row.actual
        for row in caught.value.differences
    ), "Typed differences must locate legal edits"
    assert (
        files_under(entry.result_path, entry.database_path, first.backup_path) == before
    )


def test_owner_yaml_formatting_is_not_a_difference_but_unknown_raw_is(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    first = convert(request_data, mapping)
    target = first.destination.result_path / "setup.yaml"
    raw = YAML(typ="safe").load(target.read_text())
    # JSON is YAML, but has a different layout and key order.
    target.write_text(json.dumps(raw, sort_keys=True))
    before = target.read_bytes(), target.stat().st_mtime_ns
    report = convert(replace(request_data, resume=True), mapping)
    assert any(row.destination == target for row in report.skipped)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    raw["format_version"] = "2.1"
    raw["future"] = {"opaque": 7}
    target.write_text(json.dumps(raw))
    edited = target.read_bytes()
    with pytest.raises(MigrationInputError) as caught:
        convert(replace(request_data, resume=True), mapping)
    assert any(row.destination == target for row in caught.value.differences)
    assert target.read_bytes() == edited


@pytest.mark.parametrize("changed", [False, True])
def test_replayed_source_is_compared_and_never_cleaned_twice(
    request_data: MigrationRequest, mapping: MigrationMapping, changed: bool
) -> None:
    write_meta(request_data, "", {"old_frequency": 3.0})
    report = convert(request_data, mapping)
    assert report.backup_path is not None
    for label, root in (
        ("result", request_data.result_root),
        ("Database", request_data.database_root),
    ):
        shutil.copytree(report.backup_path / label / "chip/qubit", root / "chip/qubit")
    source = request_data.result_root / "chip/qubit/meta_info.json"
    if changed:
        source.write_text('{"old_frequency": 9.0}')
    before = files_under(
        request_data.result_root, request_data.database_root, report.backup_path
    )
    if changed:
        with pytest.raises(MigrationInputError) as caught:
            convert(replace(request_data, resume=True), mapping)
        assert any(row.source == source for row in caught.value.differences)
    else:
        rerun = convert(replace(request_data, resume=True), mapping)
        assert any(
            row.destination == report.destination.result_path / "setup.yaml"
            for row in rerun.skipped
        )
    assert (
        files_under(
            request_data.result_root, request_data.database_root, report.backup_path
        )
        == before
    )


@pytest.mark.parametrize("after_rename", [False, True])
def test_point_prepared_publication_resumes_fixed_time_and_formal_binding(
    request_data: MigrationRequest, mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch, after_rename: bool
) -> None:
    write_meta(request_data, "exps/bias", {"old_frequency": 4.0})
    target = request_data.results_root / request_data.name / "points/bias"
    rename = os.rename
    def interrupt(source: str | Path, destination: str | Path, **kwargs: object) -> None:
        if Path(destination) == target:
            if after_rename:
                rename(source, destination, **kwargs)
            raise OSError("point rename interrupted")
        rename(source, destination, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, "rename", interrupt)
        with pytest.raises(OSError, match="point rename interrupted"):
            convert(request_data, mapping)
    manifest = target.parent.parent / "records/migration-state.json"
    state = json.loads(manifest.read_text())
    publication = state["points"][0]
    temporary = target.parent / publication["temporary_label"]
    prepared = (target if after_rename else temporary) / "point.yaml"
    created_at = YAML(typ="safe").load(prepared.read_text())["general"]["created_at"]
    report = convert(request_data, mapping)
    entry = ResultEntry.open(request_data.name, result_root=request_data.results_root,
                             database_root=request_data.database_root)
    point = entry.use_point("bias")
    assert source_path(point) == target / "point.yaml"
    assert point.general.created_at == created_at
    assert point.C1.frequency == 4.0
    assert not temporary.exists()
    assert any(row.new_file == target / "point.yaml" for row in report.key_mappings)


def test_old_owned_manifest_without_document_baseline_is_not_adopted(
    request_data: MigrationRequest, mapping: MigrationMapping
) -> None:
    report = convert(request_data, mapping)
    manifest = report.destination.result_path / "records/migration-state.json"
    raw = json.loads(manifest.read_text())
    raw.pop("documents", None)
    manifest.write_text(json.dumps(raw))
    assert report.backup_path is not None
    before = files_under(
        report.destination.result_path,
        report.destination.database_path,
        report.backup_path,
    )
    with pytest.raises(MigrationInputError) as caught:
        convert(replace(request_data, resume=True), mapping)
    assert any(
        row.destination == manifest and row.comparison == "document-baseline"
        for row in caught.value.differences
    )
    assert (
        files_under(
            report.destination.result_path,
            report.destination.database_path,
            report.backup_path,
        )
        == before
    )


def test_interrupted_publication_automatically_fills_missing_item(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = request_data.result_root / "chip/qubit/opaque.bin"
    raw.write_bytes(b"fixed bytes")
    target = request_data.results_root / request_data.name / "opaque.bin"
    original = os.link

    def interrupt(
        source: str | Path, destination: str | Path, **kwargs: object
    ) -> None:
        if Path(destination) == target:
            raise OSError("injected publication interruption")
        original(source, destination, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupt)
        with pytest.raises(OSError, match="injected"):
            convert(request_data, mapping)
    assert not target.exists()
    try:
        report = convert(request_data, mapping)
    except MigrationInputError as error:
        pytest.fail(f"Owned interruption must resume without a flag: {error}")
    assert target.read_bytes() == b"fixed bytes"
    assert report.failures == ()
