"""Point-local library conversion through the shipped public migration seam."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.cfg_model import FlatTopWaveformCfg, PulseCfg, PulseReadoutCfg, PulseSpec
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.entry.views import FieldView
from zcu_tools.resources.storage_migration import (
    MigrationInputError,
    MigrationManifest,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    ModuleRule,
    migrate_storage,
)

from tests.resources.storage_migration.fakes import (
    manifest_path,
    noop_validation,
    validate_cfg,
    write_meta,
)

pytestmark = pytest.mark.usefixtures("registry_guard")


def _library(meta: Path, values: YamlMap) -> Path:
    source = meta.parent / "module_cfg.yaml"
    source.write_text(json.dumps(values), encoding="utf-8")
    return source


def _pulse(waveform: str = "flat") -> YamlMap:
    return {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 123.5,
        "gain": 0.25,
        "waveform": waveform,
    }


def _payload() -> YamlMap:
    return {
        "modules": {
            "same": _pulse(),
            "readout": {
                "type": "readout/pulse",
                "pulse_cfg": "same",
                "ro_cfg": {
                    "type": "readout/direct",
                    "ro_ch": 0,
                    "ro_length": 2.5,
                    "ro_freq": 123.5,
                },
            },
        },
        "waveforms": {
            "same": {"style": "const", "length": 0.125},
            "ramp": {"style": "gauss", "length": 0.05, "sigma": 0.01},
            "flat": {"style": "flat_top", "length": 0.25, "raise_waveform": "ramp"},
        },
    }


@pytest.fixture
def library_mapping(mapping: MigrationMapping) -> MigrationMapping:
    return replace(
        mapping,
        module_rules=(
            ModuleRule(
                old_name="same",
                target_path="C1.pulses.pi",
                reference_path="C1.module.chosen",
                reason="Explicit pulse",
            ),
            ModuleRule(
                old_name="readout",
                target_path="C1.readout.main",
                reference_path=None,
                reason="Explicit readout",
            ),
            *(
                ModuleRule(
                    old_name=old,
                    target_path=f"C1.waveforms.{new}",
                    reference_path=None,
                    reason="Explicit waveform",
                    kind="waveform",
                )
                for old, new in (("same", "base"), ("ramp", "ramp"), ("flat", "flat"))
            ),
        ),
    )


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


def _manifest(request: MigrationRequest) -> MigrationManifest:
    return TypeAdapter(MigrationManifest).validate_json(
        manifest_path(request).read_bytes()
    )


def test_parameters_publish_canonical_specs_references_units_and_manual_sources(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters")
    meta = write_meta(request, "context", {"old_frequency": 12.5})
    source = _library(meta, _payload())
    original = (meta.read_bytes(), source.read_bytes())
    report = _convert(request, library_mapping)
    entry = _entry(request)
    library = entry.module_library("context")
    snapshot = library.snapshot()
    assert library.source_path == entry.result_path / "points/context/module_cfg.yaml"
    assert snapshot.module_keys() == ("C1.pulses.pi", "C1.readout.main")
    assert snapshot.waveform_keys() == (
        "C1.waveforms.base",
        "C1.waveforms.flat",
        "C1.waveforms.ramp",
    )
    spec = snapshot.module_spec("C1.pulses.pi")
    assert isinstance(spec, PulseSpec)
    assert spec.waveform == "C1.waveforms.flat"
    pulse = snapshot.get_module("C1.pulses.pi")
    assert isinstance(pulse, PulseCfg)
    assert pulse.freq == 123.5
    assert pulse.gain == 0.25
    assert isinstance(pulse.waveform, FlatTopWaveformCfg)
    assert pulse.waveform.length == 0.25
    assert pulse.waveform.raise_waveform.length == 0.05
    readout = snapshot.get_module("C1.readout.main")
    assert isinstance(readout, PulseReadoutCfg)
    assert readout.pulse_cfg.freq == 123.5
    assert readout.ro_cfg.ro_length == 2.5
    module = entry.use_point("context").C1.module
    assert isinstance(module, FieldView)
    assert module.chosen == "C1.pulses.pi"
    provenance = snapshot.meta("C1.pulses.pi.freq")
    assert provenance is not None and provenance.source == "manual"
    assert provenance.run_id is None and provenance.cloned_from is None
    assert entry.ledger.events() == ()
    decisions = {
        (item.library_kind, item.old_key): item.new_path
        for item in report.key_mappings
        if item.old_file == source
    }
    assert decisions[("module", "same")] == "C1.pulses.pi"
    assert decisions[("waveform", "same")] == "C1.waveforms.base"
    publication = next(item for item in report.converted_files if item.source == source)
    assert publication.destination == library.source_path
    assert publication.source_hash == hashlib.sha256(original[1]).hexdigest()
    assert (
        publication.destination_hash
        == hashlib.sha256(library.source_path.read_bytes()).hexdigest()
    )
    state = _manifest(request)
    assert state.parameters_complete
    assert len(state.module_files) == 1
    assert state.module_files[0].source == publication.source
    assert state.module_files[0].destination_hash == publication.destination_hash
    assert report.pending == ()
    assert report.failures == ()
    assert (meta.read_bytes(), source.read_bytes()) == original


def test_parameters_keep_arb_asset_key_without_reading_or_moving_assets(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(
        write_meta(request, "context", {}),
        {
            "modules": {"same": _pulse("same")},
            "waveforms": {"same": {"style": "arb", "data": "same"}},
        },
    )
    original = source.read_bytes()
    report = _convert(request, library_mapping)
    snapshot = _entry(request).module_library("context").snapshot()
    assert snapshot.entry_value("C1.waveforms.base", kind="waveform")["data"] == "same"
    with pytest.raises(ValueError, match="assets"):
        snapshot.get_waveform("C1.waveforms.base")
    assert any(item.source == source for item in report.converted_files)
    assert source.read_bytes() == original


def test_root_library_has_no_point_destination_or_setup_slot(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters")
    meta = write_meta(request, "", {"old_frequency": 12.5})
    source = _library(meta, _payload())
    original = source.read_bytes()
    report = _convert(request, library_mapping)
    entry = _entry(request)
    assert not entry.list_points()
    assert entry.setup.C1.frequency == 12.5
    module = entry.setup.C1.module
    assert isinstance(module, FieldView)
    with pytest.raises(AttributeError):
        _ = module.chosen
    rows = [item for item in report.key_mappings if item.old_file == source]
    assert len(rows) == 5
    assert all(item.new_file is None for item in rows)
    assert {item.location for item in report.pending if item.source == source} == {
        "modules.same",
        "modules.readout",
        "waveforms.same",
        "waveforms.ramp",
        "waveforms.flat",
    }
    assert source.read_bytes() == original


def test_unconvertible_entries_stay_located_pending_beside_successful_file(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters")
    meta = write_meta(request, "context", {})
    source = _library(
        meta,
        {
            "modules": {
                "same": _pulse("same"),
                "missing_child": _pulse("not_mapped"),
                "expression": _pulse("=ml.wave"),
                "unsupported": {"type": "unknown"},
                "no_rule": {"type": "reset/none"},
            },
            "waveforms": {
                "same": {"style": "const", "length": 0.125},
                "expression": {"style": "const", "length": "=md.duration"},
            },
            "future_root": {"keep": True},
        },
    )
    extra_rules = tuple(
        ModuleRule(
            old_name=name,
            target_path=f"C1.pulses.{name}",
            reference_path=None,
            reason="Explicit candidate",
        )
        for name in ("missing_child", "expression", "unsupported")
    )
    mapping = replace(
        library_mapping,
        module_rules=(
            *library_mapping.module_rules,
            *extra_rules,
            ModuleRule(
                old_name="expression",
                target_path="C1.waveforms.expression",
                reference_path=None,
                reason="Explicit waveform",
                kind="waveform",
            ),
        ),
    )
    original = source.read_bytes()
    report = _convert(request, mapping)
    snapshot = _entry(request).module_library("context").snapshot()
    assert snapshot.module_keys() == ("C1.pulses.pi",)
    assert snapshot.waveform_keys() == ("C1.waveforms.base",)
    pending = [item for item in report.pending if item.source == source]
    assert {item.location for item in pending} == {
        "modules.missing_child",
        "modules.expression",
        "modules.unsupported",
        "modules.no_rule",
        "waveforms.expression",
        "future_root",
    }
    assert all(item.reason for item in pending)
    assert any(item.source == source for item in report.converted_files)
    assert report.failures == ()
    assert _manifest(request).parameters_complete
    assert source.read_bytes() == original


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


@pytest.mark.parametrize("changed", ["source", "destination"])
def test_resume_rejects_changed_library_hash_without_overwriting(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    changed: str,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(write_meta(request, "context", {}), _payload())
    _convert(request, library_mapping)
    destination = _entry(request).module_library("context").source_path
    path = source if changed == "source" else destination
    path.write_bytes(path.read_bytes() + b"\n ")
    before = (source.read_bytes(), destination.read_bytes())
    with pytest.raises(MigrationInputError, match="hash"):
        _convert(replace(request, resume=True), library_mapping)
    assert (source.read_bytes(), destination.read_bytes()) == before


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


@pytest.mark.parametrize(
    "rules",
    [
        (
            ModuleRule(
                old_name="same",
                target_path="C1.pulses.pi",
                reference_path=None,
                reason="one",
            ),
            ModuleRule(
                old_name="same",
                target_path="C1.pulses.other",
                reference_path=None,
                reason="duplicate",
            ),
        ),
        (
            ModuleRule(
                old_name="wave",
                target_path="C1.pulses.pi",
                reference_path=None,
                reason="wrong family",
                kind="waveform",
            ),
        ),
        (
            ModuleRule(
                old_name="wave",
                target_path="C1.waveforms.base",
                reference_path="C1.module.chosen",
                reason="waveform slot",
                kind="waveform",
            ),
        ),
        (
            ModuleRule(
                old_name="pulse",
                target_path="C1.unknown.x",
                reference_path=None,
                reason="bad root",
            ),
        ),
    ],
)
def test_invalid_library_mapping_fails_before_entry_creation(
    request_data: MigrationRequest,
    mapping: MigrationMapping,
    rules: tuple[ModuleRule, ...],
) -> None:
    request = replace(request_data, part="parameters")
    with pytest.raises(MigrationInputError):
        _convert(request, replace(mapping, module_rules=rules))
    assert not request.results_root.exists()
