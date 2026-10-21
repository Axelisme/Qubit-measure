"""Library publication failures, dry-run guarantees and durable resume ownership."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
)
from tests.migrations.storage_migration.fakes import (
    assert_difference,
    manifest_path,
    write_meta,
)
from tests.migrations.storage_migration.module_cfg_support import convert as _convert
from tests.migrations.storage_migration.module_cfg_support import entry as _entry
from tests.migrations.storage_migration.module_cfg_support import library as _library
from tests.migrations.storage_migration.module_cfg_support import manifest as _manifest
from tests.migrations.storage_migration.module_cfg_support import payload as _payload

pytestmark = pytest.mark.usefixtures("registry_guard")


def test_dry_run_prepares_library_without_destination_or_manifest_writes(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters", dry_run=True)
    meta = write_meta(request, "context", {})
    source = _library(meta, _payload())
    original = source.read_bytes()
    report = _convert(request, library_mapping)
    assert report.pending == ()
    rows = [item for item in report.key_mappings if item.old_file == source]
    assert len(rows) == 5
    assert all(item.new_file is None for item in rows)
    assert all(item.new_path is not None and item.action == "module" for item in rows)
    assert not request.results_root.exists()
    assert not (request.database_root / request.name).exists()
    assert not manifest_path(request).exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "contents",
    [
        "modules: {}\n",
        "waveforms: {}\n",
        "modules: []\nwaveforms: {}\n",
        "modules: {}\nwaveforms: []\n",
        "",
        "null\n",
        "[]\n",
        "[entry]\n",
        "7\n",
        "legacy\n",
    ],
    ids=[
        "missing-waveforms",
        "missing-modules",
        "modules-sequence",
        "waveforms-sequence",
        "empty-yaml",
        "null-yaml",
        "empty-sequence",
        "sequence",
        "numeric-scalar",
        "text-scalar",
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_malformed_legacy_envelope_fails_without_claiming_conversion(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    contents: str,
    dry_run: bool,
) -> None:
    request = replace(request_data, part="parameters", dry_run=dry_run)
    meta = write_meta(request, "context", {})
    source = meta.parent / "module_cfg.yaml"
    source.write_text(contents, encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(MigrationInputError, match="mapping envelope"):
        _convert(request, library_mapping)
    assert source.read_bytes() == original
    if dry_run:
        assert not request.results_root.exists()
    else:
        state = _manifest(request)
        assert not state.parameters_complete
        assert state.module_files == ()
        assert state.report.converted_files == ()
        assert any(
            item.source == source and item.operation == "module_cfg"
            for item in state.report.failures
        )


def test_library_io_failure_is_reported_and_never_changes_sources(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(request_data, part="parameters")
    meta = write_meta(request, "context", {})
    source = _library(meta, _payload())
    original = (meta.read_bytes(), source.read_bytes())
    real_replace = os.replace

    def reject_library(source_path: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if target.name == "module_cfg.yaml" and target.exists():
            raise OSError("synthetic library replace failure")
        real_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", reject_library)
    with pytest.raises(OSError, match="synthetic library replace failure"):
        _convert(request, library_mapping)
    state = _manifest(request)
    failure = next(item for item in state.report.failures if item.source == source)
    assert failure.operation == "module_cfg"
    assert "synthetic library replace failure" in failure.error
    assert not state.parameters_complete
    assert state.module_files == ()
    assert _entry(request).module_library("context").snapshot().module_keys() == ()
    assert (meta.read_bytes(), source.read_bytes()) == original


def test_completed_library_is_hash_checked_without_rewriting_on_resume(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(write_meta(request, "context", {}), _payload())
    original = source.read_bytes()
    report = _convert(request, library_mapping)
    path = _entry(request).module_library("context").source_path
    completed = path.read_bytes()
    real_replace = os.replace

    def forbid_library_write(source_path: str | Path, destination: str | Path) -> None:
        if Path(destination) == path:
            raise AssertionError("completed library was rewritten")
        real_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", forbid_library_write)
    resumed = _convert(replace(request, resume=True), library_mapping)
    assert resumed.converted_files == report.converted_files
    assert path.read_bytes() == completed
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "changed", ["source", "destination", "destination-invalid-yaml"]
)
def test_resume_rejects_changed_library_hash_without_overwriting(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    changed: str,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(write_meta(request, "context", {}), _payload())
    _convert(request, library_mapping)
    destination = _entry(request).module_library("context").source_path
    backup = (
        request.result_root.parent.parent
        / "repo-backup/result/chip/qubit/context/module_cfg.yaml"
    )
    backup_before = backup.read_bytes()
    path = source if changed == "source" else destination
    if changed == "destination-invalid-yaml":
        path.write_bytes(b"not-a-library\n")
    elif changed == "source":
        path.write_bytes(path.read_bytes() + b"\n ")
    else:
        raw = YAML(typ="safe").load(path.read_text())
        raw["format_version"] = "2.1"
        raw["future"] = {"authored": "different"}
        path.write_text(json.dumps(raw))
    before = (source.read_bytes(), destination.read_bytes())
    assert_difference(
        lambda: _convert(replace(request, resume=True), library_mapping),
        request,
        [backup if changed == "source" else destination],
    )
    assert (source.read_bytes(), destination.read_bytes()) == before
    assert backup.read_bytes() == backup_before


def test_library_formatting_rerun_is_semantically_equal_and_not_rewritten(
    request_data: MigrationRequest, library_mapping: MigrationMapping
) -> None:
    request = replace(request_data, part="parameters")
    _library(write_meta(request, "context", {}), _payload())
    _convert(request, library_mapping)
    destination = _entry(request).module_library("context").source_path
    raw = YAML(typ="safe").load(destination.read_text())
    destination.write_text(json.dumps(raw, sort_keys=True))
    before = destination.read_bytes(), destination.stat().st_mtime_ns
    resumed = _convert(replace(request, resume=True), library_mapping)
    assert any(
        row.destination == destination and row.comparison == "library"
        for row in resumed.skipped
    )
    assert (destination.read_bytes(), destination.stat().st_mtime_ns) == before


def test_partial_parameter_resume_does_not_rewrite_recorded_library(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = replace(request_data, part="parameters")
    first = _library(write_meta(request, "a", {}), _payload())
    second = _library(write_meta(request, "b", {}), _payload())
    original = (first.read_bytes(), second.read_bytes())
    real_replace = os.replace

    def fail_second(source_path: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if (
            target.name == "module_cfg.yaml"
            and target.parent.name == "b"
            and target.exists()
        ):
            raise OSError("second library failed before commit")
        real_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="second library failed"):
        _convert(request, library_mapping)
    state = _manifest(request)
    assert not state.parameters_complete
    assert len(state.module_files) == 1
    first_destination = _entry(request).module_library("a").source_path
    completed = first_destination.read_bytes()

    def forbid_first(source_path: str | Path, destination: str | Path) -> None:
        if Path(destination) == first_destination:
            raise AssertionError("recorded first library was rewritten")
        real_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", forbid_first)
    report = _convert(replace(request, resume=True), library_mapping)
    assert first_destination.read_bytes() == completed
    assert _entry(request).module_library("b").snapshot().module_keys() == (
        "C1.pulses.pi",
        "C1.readout.main",
    )
    assert len(_manifest(request).module_files) == 2
    assert _manifest(request).parameters_complete
    assert report.failures == ()
    assert (first.read_bytes(), second.read_bytes()) == original
