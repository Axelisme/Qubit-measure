"""Setup and point slots reference only successfully published library specs."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from zcu_tools.resources.entry import document_value, snapshot

from migrations.storage_migration import MigrationMapping, MigrationRequest, ModuleRule
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
from tests.migrations.storage_migration.module_cfg_support import pulse as _pulse

pytestmark = pytest.mark.usefixtures("registry_guard")


def test_root_library_populates_setup_slots_without_a_point(
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
    assert any(item.source == source for item in report.converted_files)
    assert document_value(snapshot(entry.setup), "C1.module.chosen") == "C1.pulses.pi"
    library = entry.module_library()
    assert library.snapshot().module_keys() == ("C1.pulses.pi", "C1.readout.main")
    assert set(library.snapshot().waveform_keys()) == {
        "C1.waveforms.base",
        "C1.waveforms.ramp",
        "C1.waveforms.flat",
    }
    assert not any(item.source == source for item in report.pending)
    assert source.read_bytes() == original


@pytest.fixture
def slot_mapping(library_mapping: MigrationMapping) -> MigrationMapping:
    return replace(
        library_mapping,
        module_rules=(
            ModuleRule(
                old_name="unsupported",
                target_path="C1.pulses.unsupported",
                reference_path="C1.module.chosen",
                reason="Preferred pulse when convertible",
            ),
            ModuleRule(
                old_name="missing_child",
                target_path="C1.pulses.missing_child",
                reference_path="C1.module.deferred",
                reason="Pulse with explicit child mapping required",
            ),
            *library_mapping.module_rules,
        ),
    )


@pytest.mark.parametrize("label", [None, "exps/context"])
@pytest.mark.parametrize("has_success", [False, True])
def test_slots_only_reference_published_specs_in_partial_or_pending_library(
    request_data: MigrationRequest,
    slot_mapping: MigrationMapping,
    label: str | None,
    has_success: bool,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(
        write_meta(request, label or "", {}),
        {
            "modules": {
                "unsupported": {"type": "unknown"},
                "same": _pulse("same") if has_success else _pulse("=md.waveform"),
                "missing_child": _pulse("not_mapped"),
            },
            "waveforms": {"same": {"style": "const", "length": 0.125}},
        },
    )
    original = source.read_bytes()
    planned = _convert(replace(request, dry_run=True), slot_mapping)
    assert not request.results_root.exists()
    assert not (request.database_root / request.name).exists()
    assert not manifest_path(request).exists()
    assert source.read_bytes() == original

    report = _convert(request, slot_mapping)
    entry = _entry(request)
    point = None if label is None else "context"
    view = entry.setup if point is None else entry.use_point(point)
    slots = document_value(snapshot(view), "C1.module")
    assert slots == ({"chosen": "C1.pulses.pi"} if has_success else {})
    library = entry.module_library(point).snapshot()
    assert library.module_keys() == (("C1.pulses.pi",) if has_success else ())
    if has_success:
        assert library.module_spec("C1.pulses.pi").type == "pulse"
        assert library.value("C1.pulses.pi.gain") == 0.25
    expected_pending = {"modules.missing_child"}
    assert document_value(
        snapshot(view), "general.ext.legacy_module_cfg.modules.unsupported"
    ) == {"type": "unknown"}
    assert any(
        item.old_file == source
        and item.action == "preserved"
        and item.source_keys == ("modules", "unsupported")
        and item.destination_keys
        == ("general", "ext", "legacy_module_cfg", "modules", "unsupported")
        and item.new_file
        == entry.result_path
        / ("setup.yaml" if point is None else "points/context/point.yaml")
        for item in report.key_mappings
    )
    if not has_success:
        expected_pending.add("modules.same")
    assert {
        item.location for item in report.pending if item.source == source
    } == expected_pending
    assert report.pending == planned.pending
    assert all(item.reason for item in report.pending)
    assert report.failures == ()
    assert source.read_bytes() == original


@pytest.mark.parametrize("override_success", [False, True])
def test_published_slots_preserve_point_inheritance_overrides_and_completed_edits(
    request_data: MigrationRequest,
    slot_mapping: MigrationMapping,
    override_success: bool,
) -> None:
    request = replace(request_data, part="parameters")
    _library(write_meta(request, "", {}), _payload())
    write_meta(request, "exps/inherited", {})
    pulse = _pulse()
    pulse["gain"] = 0.75
    source = _library(
        write_meta(request, "exps/override", {}),
        {
            "modules": {
                "unsupported": {"type": "unknown"},
                "same": pulse if override_success else {"type": "unknown"},
            },
            "waveforms": _payload()["waveforms"],
        },
    )
    report = _convert(request, slot_mapping)
    entry = _entry(request)
    for label in (None, "inherited", "override"):
        view = entry.setup if label is None else entry.use_point(label)
        has_spec = label is None or (label == "override" and override_success)
        assert document_value(snapshot(view), "C1.module") == (
            {"chosen": "C1.pulses.pi"} if has_spec else {}
        )
        library = entry.module_library(label).snapshot()
        if has_spec:
            assert library.module_spec("C1.pulses.pi").type == "pulse"
            assert library.value("C1.pulses.pi.gain") == (
                0.75 if label == "override" else 0.25
            )
        else:
            assert library.module_keys() == ()
        with view.edit() as draft:
            draft.set("general.description", "User edit after parameters completed")
            if label is None:
                draft.set("C1.module.chosen", "C1.readout.main")
            elif has_spec:
                draft.set("C1.module", {"user_choice": "C1.pulses.pi"})
    assert entry.module_library().snapshot().value("C1.readout.main.ro_length") == 2.5
    assert document_value(
        snapshot(entry.use_point("override")),
        "general.ext.legacy_module_cfg.modules.unsupported",
    ) == {"type": "unknown"}
    assert any(
        item.old_file == source
        and item.action == "preserved"
        and item.source_keys == ("modules", "unsupported")
        and item.destination_keys
        == ("general", "ext", "legacy_module_cfg", "modules", "unsupported")
        and item.new_file == entry.result_path / "points/override/point.yaml"
        for item in report.key_mappings
    )
    assert_difference(
        lambda: _convert(replace(request, resume=True), slot_mapping),
        request,
        [entry.result_path / "setup.yaml"],
    )
    entry = _entry(request)
    for label in (None, "inherited", "override"):
        view = entry.setup if label is None else entry.use_point(label)
        expected = (
            {"chosen": "C1.readout.main"}
            if label is None
            else {"user_choice": "C1.pulses.pi"}
            if label == "override" and override_success
            else {}
        )
        assert document_value(snapshot(view), "C1.module") == expected
        assert view.general.description == "User edit after parameters completed"


@pytest.mark.parametrize("label", [None, "exps/context"])
def test_slots_are_not_published_when_library_commit_fails(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    monkeypatch: pytest.MonkeyPatch,
    label: str | None,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(write_meta(request, label or "", {}), _payload())
    real_replace = os.replace

    def reject_library(source_path: str | Path, destination: str | Path) -> None:
        if Path(destination).name == "module_cfg.yaml" and Path(destination).exists():
            raise OSError("library publication failed")
        real_replace(source_path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", reject_library)
        with pytest.raises(OSError, match="library publication failed"):
            _convert(request, library_mapping)
    entry = _entry(request)
    point = None if label is None else "context"
    view = entry.setup if point is None else entry.use_point(point)
    assert document_value(snapshot(view), "C1.module") == {}
    assert entry.module_library(point).snapshot().module_keys() == ()
    failure = next(
        item for item in _manifest(request).report.failures if item.source == source
    )
    assert failure.operation == "module_cfg"
    assert "library publication failed" in failure.error
    resumed = _convert(replace(request, resume=True), library_mapping)
    assert resumed.failures == ()
    entry = _entry(request)
    view = entry.setup if point is None else entry.use_point(point)
    assert document_value(snapshot(view), "C1.module.chosen") == "C1.pulses.pi"
    assert (
        entry.module_library(point).snapshot().module_spec("C1.pulses.pi").type
        == "pulse"
    )
