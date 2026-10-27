"""Typed dispersive and T1 analysis publication, located evidence and resume."""

import importlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.entry import document_value, snapshot

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    migrate_storage,
)
from migrations.storage_migration.t1_curve_file import load_t1_curve_file
from tests.migrations.storage_migration.fakes import (
    assert_difference,
    noop_validation,
    validate_cfg,
    write_meta,
)
from tests.migrations.storage_migration.params_support import (
    backup_path,
    load_migrated_predictor,
    migrate,
    open_entry,
    write_params,
)

pytestmark = pytest.mark.usefixtures("analysis_kind")


@pytest.mark.parametrize("kind", ["predictors", "dispersive", "t1_curve"])
def test_completed_analysis_rerun_uses_typed_values_and_locates_legal_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
    kind: str,
) -> None:
    values = dict(analysis_sections)
    values["fluxdep_fit"] = {
        "params": {"EJ": 8.0, "EC": 1.0, "EL": 0.5},
        "flux_half": 0.2,
        "flux_period": -1.0,
        "flux_bias": 0.0,
    }
    write_params(request_data, values)
    report = migrate(request_data, analysis_mapping)
    target = report.destination.result_path / kind / "Qlegacy.json"
    raw = json.loads(target.read_text())
    target.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")))
    before = target.read_bytes(), target.stat().st_mtime_ns
    rerun = migrate(replace(request_data, resume=True), analysis_mapping)
    assert any(row.destination == target for row in rerun.skipped)
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    raw["source"]["run_id"] = "legal-user-edit"
    target.write_text(json.dumps(raw))
    edited = target.read_bytes()
    with pytest.raises(MigrationInputError) as caught:
        migrate(replace(request_data, resume=True), analysis_mapping)
    assert any(
        row.destination == target and row.expected != row.actual
        for row in caught.value.differences
    )
    assert target.read_bytes() == edited


def test_analysis_files_publish_typed_unknowns_and_keep_original_sections(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
) -> None:
    write_params(request_data, analysis_sections)
    report = migrate(request_data, analysis_mapping)
    entry = open_entry(request_data)
    for kind in ("dispersive", "t1_curve"):
        file = entry.result_path / kind / "Qlegacy.json"
        assert file.is_file(), f"migration must publish {kind} fit"
        package = (
            "zcu_tools.resources"
            if kind == "dispersive"
            else "migrations.storage_migration"
        )
        owner = importlib.import_module(f"{package}.{kind}_file")
        loaded = getattr(owner, f"load_{kind}_file")(file)
        assert loaded.source.run_id is None
        assert any(
            row.new_file == file and row.action == "value"
            for row in report.key_mappings
        )
        if kind == "dispersive":
            assert entry.dispersive_path == file.parent
            assert loaded.params.g is None
            assert loaded.params.bare_rf == 5.1
            assert loaded.params.plot_transitions.r_f == 5.2
            assert loaded.params.plot_transitions.model_extra == {"0.1": [1, 2]}
            assert loaded.source.created_at is None
            assert any(
                item.location == "dispersive.timestamp" for item in report.pending
            )
        else:
            assert (entry.result_path / "t1_curve") == file.parent
            assert loaded.params.params.Q_cap is None
            assert loaded.params.params.Temp == 0.04
            assert loaded.params.stderr.Q_cap is None
            assert loaded.params.bounds == {"Q_cap": (1.0, None)}
            assert loaded.source.created_at == "2020-07-07T16:30:00+08:00"
    doc = snapshot(entry.setup)
    assert document_value(doc, "Qlegacy.ext.legacy_params") == {
        "dispersive": {"timestamp": "unproven", "future": [1, None]}
    }


@pytest.mark.parametrize(
    ("section", "kind", "content"),
    [
        ("dispersive", "dispersive", {"g": None, "timestamp": "unproven"}),
        ("t1_curve_fit", "t1_curve", {"params": {"Q_cap": None}, "timestamp": None}),
        (
            "fluxdep_fit",
            "dispersive",
            {"plot_transitions": {"r_f": 5.2}, "timestamp": "2020-07-07T16:30:00"},
        ),
    ],
)
def test_analysis_timestamp_locators_survive_source_cleanup_and_resume(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    legacy_fit: JsonObject,
    section: str,
    kind: str,
    content: JsonObject,
) -> None:
    legacy_fit["timestamp"] = "unproven"
    values: JsonObject = {"flux_unit": "A", "fluxdep_fit": legacy_fit}
    if section == "fluxdep_fit":
        legacy_fit.update(content)
    else:
        values[section] = content
    source = write_params(request_data, values)
    original = source.read_bytes()
    destination = request_data.results_root / request_data.name

    dry = migrate_storage(
        replace(request_data, dry_run=True),
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert source.read_bytes() == original
    assert not destination.exists()
    assert not backup_path(request_data).exists()

    report = migrate_storage(
        request_data,
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    backup_source = backup_path(request_data) / "result/chip/qubit/params.json"
    assert backup_source.read_bytes() == original
    assert TypeAdapter(JsonObject).validate_json(backup_source.read_text()) == values
    assert not source.parent.exists()
    assert not (request_data.database_root / "chip/qubit").exists()
    file = destination / kind / "Qlegacy.json"
    loaded = (
        load_dispersive_file(file) if kind == "dispersive" else load_t1_curve_file(file)
    )
    assert loaded.source.created_at is None
    assert load_migrated_predictor(request_data).source.created_at is None
    locations = {f"{section}.timestamp", "fluxdep_fit.timestamp"}
    rows = [
        row
        for row in report.key_mappings
        if row.old_key in locations and row.action == "pending"
    ]
    assert any(
        row.new_file == file and row.old_key == f"{section}.timestamp" for row in rows
    )
    for row in rows:
        assert row.old_file == source
        assert f"{backup_source}#{row.old_key}" in row.reason
    pending = [item for item in report.pending if item.location in locations]
    assert pending
    for item in pending:
        assert item.source == backup_source
        assert f"{backup_source}#{item.location}" in item.reason
    dry_pending = [item for item in dry.pending if item.location in locations]
    assert [(item.location, item.reason) for item in dry_pending] == [
        (item.location, item.reason) for item in pending
    ]
    assert all(item.source == source for item in dry_pending)

    published = file.read_bytes()
    resumed = migrate_storage(
        replace(request_data, resume=True),
        mapping=analysis_mapping,
        validate_cfg=validate_cfg,
        validate_native=noop_validation,
    )
    assert resumed.backup_path == report.backup_path == backup_path(request_data)
    assert resumed.pending == report.pending
    assert file.read_bytes() == published
    assert backup_source.read_bytes() == original


def test_transition_only_fit_publishes_dispersive_without_inventing_fit_values(
    request_data: MigrationRequest, analysis_mapping: MigrationMapping
) -> None:
    write_params(request_data, {"fluxdep_fit": {"plot_transitions": {"r_f": 5.2}}})
    migrate(request_data, analysis_mapping)
    file = open_entry(request_data).result_path / "dispersive/Qlegacy.json"
    assert file.is_file(), "transition seed alone must publish dispersive data"
    owner = importlib.import_module("zcu_tools.resources.dispersive_file")
    loaded = owner.load_dispersive_file(file)
    assert loaded.params.g is None
    assert loaded.params.bare_rf is None
    assert loaded.params.plot_transitions.r_f == 5.2


@pytest.mark.parametrize(
    ("section", "value", "kind"),
    [
        ("dispersive", {"g": True}, "dispersive"),
        ("t1_curve_fit", {"params": {"Temp": None}, "free": ["Q_cap"]}, "t1_curve"),
    ],
)
def test_invalid_analysis_content_stays_located_pending(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    section: str,
    value: JsonObject,
    kind: str,
) -> None:
    write_params(request_data, {section: value})
    report = migrate(request_data, analysis_mapping)
    file = open_entry(request_data).result_path / kind / "Qlegacy.json"
    assert any(
        row.old_key == section
        and row.new_file == file
        and row.action == "pending"
        and "validation" in row.reason.lower()
        for row in report.key_mappings
    )
    assert (
        document_value(
            snapshot(open_entry(request_data).setup),
            f"Qlegacy.ext.legacy_params.{section}",
        )
        == value
    )


@pytest.mark.parametrize("kind", ["dispersive", "t1_curve"])
@pytest.mark.parametrize("conflict", [False, True])
def test_analysis_file_resume_validates_unfinished_and_preserves_completed_edits(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    conflict: bool,
) -> None:
    write_params(request_data, analysis_sections)
    write_meta(request_data, "bias", {})
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupt(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == destination / "setup.yaml" and dst.name == "point.yaml":
            raise OSError("analysis checkpoint interrupted")
        return copyfile(src, dst, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", interrupt)
        with pytest.raises(OSError, match="analysis checkpoint interrupted"):
            migrate(request_data, analysis_mapping)
    file = destination / kind / "Qlegacy.json"
    assert file.is_file(), "fit must be published before the interrupted checkpoint"
    raw = json.loads(file.read_text(encoding="utf-8"))
    if conflict:
        raw["version"] = True
    file.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
    before = file.read_bytes()
    if conflict:
        with pytest.raises(MigrationInputError, match=kind.replace("_", "[ _-]?")):
            migrate(replace(request_data, resume=True), analysis_mapping)
    else:
        migrate(replace(request_data, resume=True), analysis_mapping)
        assert file.read_bytes() == before
        raw["source"]["run_id"] = "user-edited"
        file.write_text(json.dumps(raw), encoding="utf-8")
        edited = file.read_bytes()
        assert_difference(
            lambda: migrate(replace(request_data, resume=True), analysis_mapping),
            request_data,
            [file],
        )
        assert file.read_bytes() == edited


def test_unfinished_t1_resume_rejects_new_active_null_channel(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_params(request_data, analysis_sections)
    write_meta(request_data, "bias", {})
    destination = request_data.results_root / request_data.name
    copyfile = shutil.copyfile

    def interrupt(src: Path, dst: Path, *, follow_symlinks: bool = True) -> Path:
        if src == destination / "setup.yaml" and dst.name == "point.yaml":
            raise OSError("T1 checkpoint interrupted")
        return copyfile(src, dst, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", interrupt)
        with pytest.raises(OSError, match="T1 checkpoint interrupted"):
            migrate(request_data, analysis_mapping)
    file = open_entry(request_data).result_path / "t1_curve/Qlegacy.json"
    raw = json.loads(file.read_text(encoding="utf-8"))
    raw["params"]["params"]["Q_ind"] = None
    file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(MigrationInputError, match="differs"):
        migrate(replace(request_data, resume=True), analysis_mapping)


def test_analysis_dry_run_reports_the_same_files_without_writes(
    request_data: MigrationRequest,
    analysis_mapping: MigrationMapping,
    analysis_sections: JsonObject,
) -> None:
    write_params(request_data, analysis_sections)
    dry = migrate(replace(request_data, dry_run=True), analysis_mapping)
    destination = request_data.results_root / request_data.name
    for kind in ("dispersive", "t1_curve"):
        assert any(
            row.new_file == destination / kind / "Qlegacy.json"
            and row.action == "value"
            for row in dry.key_mappings
        )
    assert not destination.exists()
    assert not backup_path(request_data).exists()
