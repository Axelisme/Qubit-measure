"""Point-local library conversion through the shipped public migration seam."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from ruamel.yaml import YAML
from zcu_tools.cfg_model import FlatTopWaveformCfg, PulseCfg, PulseReadoutCfg, PulseSpec
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import ResultEntry, document_value, snapshot
from zcu_tools.resources.entry.views import FieldView

from migrations.storage_migration import (
    MigrationInputError,
    MigrationManifest,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    ModuleRule,
    migrate_storage,
)
from tests.migrations.storage_migration.fakes import (
    assert_difference,
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
                "gen_ch": 0,
                "nqz": 1,
                "freq": 123.5,
                "gain": 0.25,
                "waveform": "flat",
                "ro_ch": 0,
                "ro_length": 2.5,
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


def _historical_pulse() -> YamlMap:
    return {
        "type": "pulse",
        "ch": 2,
        "nqz": 2,
        "freq": 456.75,
        "gain": 0.125,
        "mixer_freq": 100.5,
        "desc": "Drive description",
        "waveform": {
            "style": "flat_top",
            "length": 0.25,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        },
    }


@pytest.fixture
def historical_mapping(mapping: MigrationMapping) -> MigrationMapping:
    return replace(
        mapping,
        module_rules=tuple(
            ModuleRule(
                old_name=name,
                target_path=f"C1.{group}.{name}",
                reference_path=f"C1.module.{name}" if group != "waveforms" else None,
                reason="Historical reader shape",
                kind="waveform" if group == "waveforms" else "module",
            )
            for group, names in (
                ("pulses", ("pi_amp", "pi2_amp", "pi_len", "pi2_len")),
                ("readout", ("readout_dpm", "readout_rf")),
                ("reset", ("reset_bath", "reset_bath_e")),
                ("waveforms", ("mist_waveform", "qub_cos", "qub_flat", "ro_waveform")),
            )
            for name in names
        ),
    )


def _historical_library() -> YamlMap:
    modules: YamlMap = {
        name: _historical_pulse() for name in ("pi_amp", "pi2_amp", "pi_len", "pi2_len")
    }
    for name in ("readout_dpm", "readout_rf"):
        modules[name] = {
            "type": "base",
            "desc": "Readout description",
            "pulse_cfg": _historical_pulse(),
            "ro_cfg": {"ro_ch": 1, "ro_length": 0.9, "trig_offset": 0.625},
        }
    for name, phase in (("reset_bath", -103.0), ("reset_bath_e", 77.0)):
        modules[name] = {
            "type": "bath",
            "desc": "Bath description",
            "cavity_tone_cfg": _historical_pulse(),
            "qubit_tone_cfg": _historical_pulse(),
            "pi2_cfg": {**_historical_pulse(), "phase": phase},
        }
    return {
        "modules": modules,
        "waveforms": {
            "mist_waveform": {"style": "const", "length": 1.0},
            "qub_cos": {"style": "cosine", "length": 2.0},
            "qub_flat": {
                "style": "flat_top",
                "length": 2.0,
                "raise_waveform": {"style": "cosine", "length": 0.02},
            },
            "ro_waveform": {
                "style": "flat_top",
                "length": 5.0,
                "raise_waveform": {"style": "cosine", "length": 0.1},
            },
        },
    }


@pytest.mark.parametrize("label", ["", "exps/context"])
def test_historical_library_reconstructs_eight_modules_and_four_waveforms(
    request_data: MigrationRequest,
    historical_mapping: MigrationMapping,
    label: str,
) -> None:
    request = replace(request_data, part="parameters")
    source = _library(write_meta(request, label, {}), _historical_library())
    planned = _convert(replace(request, dry_run=True), historical_mapping)
    assert not request.results_root.exists()
    assert not (request.database_root / request.name).exists()
    report = _convert(request, historical_mapping)
    entry = _entry(request)
    point = "context" if label else None
    library = entry.module_library(point).snapshot()
    assert len(library.module_keys()) == 8
    assert len(library.waveform_keys()) == 4
    pulse = library.module_spec("C1.pulses.pi_amp")
    assert pulse == PulseSpec.model_validate(_historical_pulse())
    assert library.value("C1.pulses.pi_amp.phase") == 0.0
    assert library.value("C1.pulses.pi_amp.pre_delay") == 0.0
    assert library.value("C1.pulses.pi_amp.post_delay") == 0.0
    for name in ("readout_dpm", "readout_rf"):
        readout = library.entry_value(f"C1.readout.{name}", kind="module")
        assert readout == {
            "type": "readout/pulse",
            "desc": "Readout description",
            "gen_ch": 2,
            "nqz": 2,
            "freq": 456.75,
            "gain": 0.125,
            "mixer_freq": 100.5,
            "phase": 0.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
            "waveform": {
                "style": "flat_top",
                "length": 0.25,
                "raise_waveform": {"style": "cosine", "length": 0.02},
            },
            "ro_ch": 1,
            "ro_length": 0.9,
            "trig_offset": 0.625,
        }
        assert library.get_module(f"C1.readout.{name}").type == "readout/pulse"
    for name, phase in (("reset_bath", -103.0), ("reset_bath_e", 77.0)):
        bath = library.module_spec(f"C1.reset.{name}")
        assert bath.type == "reset/bath"
        assert library.value(f"C1.reset.{name}.desc") == "Bath description"
        for child in ("cavity_tone_cfg", "qubit_tone_cfg", "pi2_cfg"):
            assert library.value(f"C1.reset.{name}.{child}.desc") == "Drive description"
            assert library.value(f"C1.reset.{name}.{child}.mixer_freq") == 100.5
        assert library.value(f"C1.reset.{name}.pi2_cfg.phase") == phase
        assert library.get_module(f"C1.reset.{name}").type == "reset/bath"
    assert library.get_waveform("C1.waveforms.mist_waveform").length == 1.0
    assert library.get_waveform("C1.waveforms.qub_cos").length == 2.0
    assert library.get_waveform("C1.waveforms.qub_flat").length == 2.0
    assert library.get_waveform("C1.waveforms.ro_waveform").length == 5.0
    view = entry.setup if point is None else entry.use_point(point)
    assert document_value(snapshot(view), "C1.module") == {
        rule.old_name: rule.target_path
        for rule in historical_mapping.module_rules
        if rule.kind == "module"
    }
    assert not any(item.source == source for item in report.pending)
    assert planned.pending == report.pending
    assert report.failures == ()


@pytest.mark.parametrize("label", ["", "exps/context"])
def test_module_unknown_content_is_visible_without_duplicate_converted_leaves(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
    label: str,
) -> None:
    request = replace(request_data, part="parameters")
    pulse: YamlMap = {
        **_pulse("same"),
        "desc": "Formal description",
        "mixer_freq": 80.0,
        "name": "legacy drive",
        "extra.with.dot": {"expr": "=unknown.value"},
        "waveform": {
            "style": "flat_top",
            "length": 0.3,
            "note": [],
            "raise_waveform": {"style": "cosine", "length": 0.02, "old.key": None},
        },
    }
    source = _library(
        write_meta(request, label, {}),
        {
            "modules": {"same": pulse, "unknown.entry": {"type": "future", "value": 8}},
            "waveforms": {"same": {"style": "const", "length": 0.1, "note": "keep"}},
            "unknown.section": {"original": [1, None, "=not.evaluated"]},
        },
    )
    planned = _convert(replace(request, dry_run=True), library_mapping)
    assert not request.results_root.exists()
    report = _convert(request, library_mapping)
    entry = _entry(request)
    point = "context" if label else None
    library = entry.module_library(point).snapshot()
    assert library.module_keys() == ("C1.pulses.pi",)
    assert library.value("C1.pulses.pi.desc") == "Formal description"
    assert library.value("C1.pulses.pi.mixer_freq") == 80.0
    assert library.get_module("C1.pulses.pi").type == "pulse"
    view = entry.setup if point is None else entry.use_point(point)
    assert document_value(snapshot(view), "general.ext.legacy_module_cfg") == {
        "modules": {
            "same": {
                "name": "legacy drive",
                "extra.with.dot": {"expr": "=unknown.value"},
                "waveform": {"note": [], "raise_waveform": {"old.key": None}},
            },
            "unknown.entry": {"type": "future", "value": 8},
        },
        "waveforms": {"same": {"note": "keep"}},
        "unknown.section": {"original": [1, None, "=not.evaluated"]},
    }
    rows = [
        item
        for item in report.key_mappings
        if item.old_file == source and item.action == "preserved"
    ]
    assert rows
    assert all(item.new_path is None for item in rows)
    assert all(
        item.new_file
        == entry.result_path / ("points/context/point.yaml" if point else "setup.yaml")
        for item in rows
    )
    assert any(
        item.source_keys == ("modules", "same", "extra.with.dot")
        and item.destination_keys
        == ("general", "ext", "legacy_module_cfg", "modules", "same", "extra.with.dot")
        for item in rows
    )
    assert report.pending == planned.pending == ()
    assert report.failures == ()
    with view.edit() as draft:
        draft.set("general.ext.legacy_module_cfg", {"user": "edited"})
    assert_difference(lambda: _convert(replace(request, resume=True), library_mapping), request,
                      [entry.result_path / ("points/context/point.yaml" if point else "setup.yaml")])
    edited = _entry(request)
    edited_view = edited.setup if point is None else edited.use_point(point)
    assert document_value(snapshot(edited_view), "general.ext.legacy_module_cfg") == {
        "user": "edited"
    }


def test_supported_module_missing_required_evidence_is_pending_and_preserved(
    request_data: MigrationRequest,
    library_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters")
    pulse = _pulse("same")
    del pulse["gain"]
    source = _library(
        write_meta(request, "", {}),
        {
            "modules": {"same": pulse},
            "waveforms": {"same": {"style": "const", "length": 0.1}},
        },
    )
    report = _convert(request, library_mapping)
    entry = _entry(request)
    assert document_value(snapshot(entry.setup), "general.ext") == {
        "legacy_module_cfg": {"modules": {"same": pulse}}
    }
    assert entry.module_library().snapshot().module_keys() == ()
    assert document_value(snapshot(entry.setup), "C1.module") == {}
    assert any(
        item.source == source
        and item.location == "modules.same"
        and "gain" in item.reason
        for item in report.pending
    )
    assert any(
        item.old_file == source
        and item.action == "preserved"
        and item.source_keys == ("modules", "same")
        for item in report.key_mappings
    )


def test_missing_discriminator_requires_complete_historical_pulse_shape(
    request_data: MigrationRequest,
    historical_mapping: MigrationMapping,
) -> None:
    request = replace(request_data, part="parameters")
    pulse = _historical_pulse()
    del pulse["type"]
    source = _library(
        write_meta(request, "", {}),
        {
            "modules": {"pi_amp": pulse, "pi2_amp": {"waveform": "unknown"}},
            "waveforms": {},
        },
    )
    report = _convert(request, historical_mapping)
    entry = _entry(request)
    assert entry.module_library().snapshot().module_keys() == ("C1.pulses.pi_amp",)
    assert (
        entry.module_library().snapshot().module_spec("C1.pulses.pi_amp").type
        == "pulse"
    )
    assert document_value(snapshot(entry.setup), "general.ext.legacy_module_cfg") == {
        "modules": {"pi2_amp": {"waveform": "unknown"}}
    }
    assert not any(item.source == source for item in report.pending)


@pytest.mark.parametrize("ro_frequency", [456.75, 123.0])
def test_base_readout_frequency_agreement_controls_publication(
    request_data: MigrationRequest,
    historical_mapping: MigrationMapping,
    ro_frequency: float,
) -> None:
    request = replace(request_data, part="parameters")
    readout: YamlMap = {
        "type": "base",
        "pulse_cfg": _historical_pulse(),
        "ro_cfg": {
            "ro_ch": 1,
            "ro_length": 0.9,
            "trig_offset": 0.625,
            "ro_freq": ro_frequency,
        },
    }
    source = _library(
        write_meta(request, "", {}),
        {
            "modules": {"readout_dpm": readout},
            "waveforms": {},
        },
    )
    report = _convert(request, historical_mapping)
    entry = _entry(request)
    expected_keys = ("C1.readout.readout_dpm",) if ro_frequency == 456.75 else ()
    assert entry.module_library().snapshot().module_keys() == expected_keys
    raw = document_value(snapshot(entry.setup), "general.ext")
    if ro_frequency == 456.75:
        assert raw == {
            "legacy_module_cfg": {
                "modules": {
                    "readout_dpm": {
                        "pulse_cfg": {"desc": "Drive description", "type": "pulse"}
                    }
                }
            }
        }
        assert not any(item.source == source for item in report.pending)
    else:
        assert raw == {"legacy_module_cfg": {"modules": {"readout_dpm": readout}}}
        assert any(
            item.source == source and "456.75" in item.reason and "123.0" in item.reason
            for item in report.pending
        )
        assert document_value(snapshot(entry.setup), "C1.module") == {}


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
    assert readout.freq == 123.5
    assert readout.ro_length == 2.5
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
    assert_difference(lambda: _convert(replace(request, resume=True), slot_mapping), request,
                      [entry.result_path / "setup.yaml"])
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
    library = _entry(request).module_library("context").snapshot()
    assert library.module_keys() == ("C1.pulses.pi",)
    assert library.waveform_keys() == ("C1.waveforms.base",)
    pending = [item for item in report.pending if item.source == source]
    assert {item.location for item in pending} == {
        "modules.missing_child",
        "modules.expression",
        "waveforms.expression",
    }
    raw = document_value(
        snapshot(_entry(request).use_point("context")), "general.ext.legacy_module_cfg"
    )
    assert raw == {
        "modules": {
            "missing_child": _pulse("not_mapped"),
            "expression": _pulse("=ml.wave"),
            "unsupported": {"type": "unknown"},
            "no_rule": {"type": "reset/none"},
        },
        "waveforms": {"expression": {"style": "const", "length": "=md.duration"}},
        "future_root": {"keep": True},
    }
    for keys in (("modules", "unsupported"), ("modules", "no_rule"), ("future_root",)):
        assert any(
            item.old_file == source
            and item.action == "preserved"
            and item.source_keys == keys
            and item.destination_keys == ("general", "ext", "legacy_module_cfg", *keys)
            and item.new_file
            == _entry(request).result_path / "points/context/point.yaml"
            for item in report.key_mappings
        )
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
    assert_difference(lambda: _convert(replace(request, resume=True), library_mapping), request,
                      [backup if changed == "source" else destination])
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
    assert any(row.destination == destination and row.comparison == "library" for row in resumed.skipped)
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
