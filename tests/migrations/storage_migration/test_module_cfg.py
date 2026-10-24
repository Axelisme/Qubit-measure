"""Canonical and historical library specs, source preservation and mapping validation."""

import hashlib
from dataclasses import replace

import pytest
from zcu_tools.cfg_model import FlatTopWaveformCfg, PulseCfg, PulseReadoutCfg, PulseSpec
from zcu_tools.resources.document_store import YamlMap
from zcu_tools.resources.entry import document_value, snapshot
from zcu_tools.resources.entry.views import FieldView

from migrations.storage_migration import (
    MigrationInputError,
    MigrationMapping,
    MigrationRequest,
    ModuleRule,
)
from tests.migrations.storage_migration.fakes import assert_difference, write_meta
from tests.migrations.storage_migration.module_cfg_support import convert as _convert
from tests.migrations.storage_migration.module_cfg_support import entry as _entry
from tests.migrations.storage_migration.module_cfg_support import library as _library
from tests.migrations.storage_migration.module_cfg_support import manifest as _manifest
from tests.migrations.storage_migration.module_cfg_support import payload as _payload
from tests.migrations.storage_migration.module_cfg_support import pulse as _pulse

pytestmark = pytest.mark.usefixtures("registry_guard")


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
    assert_difference(
        lambda: _convert(replace(request, resume=True), library_mapping),
        request,
        [entry.result_path / ("points/context/point.yaml" if point else "setup.yaml")],
    )
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
                "no_rule": {"type": "pulse", "waveform": "unmapped"},
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
            "no_rule": {"type": "pulse", "waveform": "unmapped"},
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
