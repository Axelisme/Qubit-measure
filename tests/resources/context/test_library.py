"""Specs, materialization and single-file publication contracts."""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    PulseCfg,
    PulseReadoutCfg,
    PulseSpec,
    waveform_spec_from_raw,
)
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibrary,
    ModuleLibraryDocument,
    library_key_kind,
)
from zcu_tools.resources.document_store import (
    ConflictError,
    DocumentChange,
    DocumentValue,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.program.v2.support.waveform_assets import InMemoryWaveformAsset


def pulse(*, gain: float = 0.25, waveform: str = "Q1.waveforms.base") -> PulseSpec:
    return PulseSpec(ch=0, nqz=1, freq=100.0, gain=gain, waveform=waveform)


def document() -> ModuleLibraryDocument:
    return ModuleLibraryDocument(
        format="zcu.module-library",
        format_version="2.0",
        components={
            "Q1": ComponentLibrary(
                pulses={"pi": pulse(), "pi2": pulse(gain=0.125)},
                waveforms={"base": ConstWaveformCfg(length=0.05)},
            )
        },
    )


def test_snapshot_public_context_retains_capture_bound_asset_path(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    repository = ArbWaveformDatabase(first)
    repository.save("shape", idata=np.array([0.0, 0.5]), time=np.array([0.0, 1.0]))
    captured = LibrarySnapshot(document(), assets=repository.bound_info())
    other = ArbWaveformDatabase(tmp_path / "second")
    other.save("shape", idata=[0.9, 0.9], time=[0.0, 9.0])
    context = captured.validation_context()
    assert context.assets is not None
    assert context.assets.path_for("shape") == (first / "shape.npz").resolve()


def assets() -> InMemoryWaveformAsset:
    return InMemoryWaveformAsset(
        "present",
        WaveformSamples(
            time=np.array([0.0, 0.05]),
            idata=np.array([0.0, 1.0]),
            qdata=None,
        ),
    )


@pytest.fixture
def library_path(tmp_path: Path) -> Path:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        "format: zcu.module-library\nformat_version: '2.0'\n", encoding="utf-8"
    )
    return path


def open_library(library_path: Path) -> ModuleLibrary:
    path = library_path
    return ModuleLibrary(path, assets=assets())


def write_document(library_path: Path, value: ModuleLibraryDocument) -> None:
    library_path.write_text(value.model_dump_json(), encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("Q1.pulses.pi", "module"),
        ("R1.readout.default", "module"),
        ("Q1.reset.none", "module"),
        ("Q1.waveforms.base", "waveform"),
    ],
)
def test_library_key_kind_uses_exact_root_partition(path: str, kind: str) -> None:
    assert library_key_kind(path) == kind


@pytest.mark.parametrize(
    "path",
    [
        "",
        "Q1.pulses",
        "Q1.pulses.pi.gain",
        ".pulses.pi",
        "Q1.unknown.x",
        "Q1.waveforms.",
    ],
)
def test_library_key_kind_rejects_invalid_or_leaf_paths(path: str) -> None:
    with pytest.raises(ValueError, match="[Ll]ibrary"):
        library_key_kind(path)


def test_library_without_assets_supports_specs_but_rejects_arb_materialization(
    library_path: Path,
) -> None:
    path = library_path
    library = ModuleLibrary(path)
    with library.edit() as draft:
        draft.set_waveform(
            "Q1.waveforms.arb",
            waveform_spec_from_raw({"style": "arb", "data": "present"}),
        )
    snapshot = library.snapshot()
    assert (
        snapshot.entry_value("Q1.waveforms.arb", kind="waveform")["data"] == "present"
    )
    with pytest.raises(ValueError, match="assets"):
        snapshot.get_waveform("Q1.waveforms.arb")


def test_header_only_open_and_empty_edit_do_not_rewrite(
    library_path: Path,
) -> None:
    original = library_path.read_bytes()
    library = open_library(library_path)
    assert library.source_path == library_path.resolve()
    assert library.snapshot().module_keys() == ()
    assert library.snapshot().waveform_keys() == ()
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)
    with library.edit() as draft:
        assert draft.snapshot().document().components == {}
    assert library_path.read_bytes() == original
    assert changes == []


def test_missing_document_is_not_created(library_path: Path) -> None:
    library_path.unlink()
    with pytest.raises(FileNotFoundError):
        open_library(library_path)
    assert not library_path.exists()


@pytest.mark.parametrize(
    "content",
    [
        "waveforms: {}\nmodules: {}\n",
        "format: wrong\nformat_version: '2.0'\n",
        "format: zcu.module-library\nformat_version: '1.0'\n",
        "format: zcu.module-library\nformat_version: '2.0'\nunknown: true\n",
    ],
)
def test_open_rejects_invalid_document_without_rewriting(
    library_path: Path, content: str
) -> None:
    library_path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="format|header|version|Extra"):
        open_library(library_path)
    assert library_path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("Q1.pulses.pi.gain", DocumentValue(True, 0.25)),
        ("Q1.pulses.pi.mixer_freq", DocumentValue(True, None)),
        ("Q1.pulses.missing.gain", DocumentValue(False, None)),
        ("Q1.pulses.pi.absent", DocumentValue(False, None)),
        ("Q1.pulses.pi.waveform.length", DocumentValue(True, 0.05)),
    ],
)
def test_snapshot_lookup_preserves_requested_missing_and_null(
    path: str, expected: DocumentValue
) -> None:
    assert LibrarySnapshot(document()).lookup(path) == expected


def test_snapshot_lookup_propagates_missing_child_instead_of_absent() -> None:
    value = document()
    value.components["Q1"].waveforms.clear()
    with pytest.raises(ValueError, match="chain Q1.pulses.pi -> Q1.waveforms.base"):
        LibrarySnapshot(value).lookup("Q1.pulses.pi.waveform.length")


def test_snapshot_preserves_specs_and_materializes_independent_cfgs() -> None:
    original = document()
    snapshot = LibrarySnapshot(original, assets=assets())
    original.components["Q1"].pulses["pi"].gain = 0.9
    assert snapshot.module_keys() == ("Q1.pulses.pi", "Q1.pulses.pi2")
    assert snapshot.waveform_keys() == ("Q1.waveforms.base",)
    spec = snapshot.module_spec("Q1.pulses.pi")
    assert isinstance(spec, PulseSpec)
    assert spec.waveform == "Q1.waveforms.base"
    spec.gain = 0.8
    cfg = snapshot.get_module("Q1.pulses.pi")
    assert isinstance(cfg, PulseCfg)
    assert isinstance(cfg.waveform, ConstWaveformCfg)
    assert cfg.gain == 0.25
    cfg.waveform.length = 0.8
    assert snapshot.value("Q1.pulses.pi.waveform.length") == 0.05
    copied = snapshot.document()
    copied.components["Q1"].pulses.clear()
    assert snapshot.module_keys() == ("Q1.pulses.pi", "Q1.pulses.pi2")
    assert (
        snapshot.entry_value("Q1.pulses.pi", kind="module")["waveform"]
        == "Q1.waveforms.base"
    )


def test_validation_context_is_fresh_and_uses_fixed_snapshot() -> None:
    provider = assets()
    snapshot = LibrarySnapshot(document(), assets=provider)
    first = snapshot.validation_context()
    second = snapshot.validation_context()
    assert first is not second
    assert first.modules is snapshot
    assert first.assets is provider
    assert second.modules is snapshot


def test_snapshot_is_memory_only_and_does_not_implicitly_refresh(
    library_path: Path,
) -> None:
    write_document(library_path, document())
    library = open_library(library_path)
    first = library.snapshot()
    changed = document()
    changed.components["Q1"].pulses["pi"].gain = 0.5
    write_document(library_path, changed)
    library_path.unlink()
    assert first.value("Q1.pulses.pi.gain") == 0.25
    assert library.snapshot().value("Q1.pulses.pi.gain") == 0.25


@pytest.mark.parametrize(
    "path", ["Q1", "Q1.pulses", "Q1..pi", "Q1.pulses.pi.gain", ".pulses.pi"]
)
def test_entry_projection_requires_exact_concrete_root(path: str) -> None:
    snapshot = LibrarySnapshot(document())
    with pytest.raises(ValueError, match="root|path"):
        snapshot.entry_value(path, kind="module")


def test_projection_kind_and_missing_root_are_distinct() -> None:
    snapshot = LibrarySnapshot(document())
    with pytest.raises(ValueError, match="kind|partition"):
        snapshot.entry_value("Q1.waveforms.base", kind="module")
    with pytest.raises(KeyError):
        snapshot.entry_value("Q1.pulses.absent", kind="module")
    with pytest.raises(AttributeError):
        snapshot.value("Q1.pulses.pi.absent")
    with pytest.raises(AttributeError):
        snapshot.value("Q1.pulses.absent.gain")


def test_entry_projection_does_not_expand_missing_asset_or_child() -> None:
    value = document()
    value.components["Q1"].pulses["pi"].waveform = "Q1.waveforms.absent"
    value.components["Q1"].waveforms["arb"] = waveform_spec_from_raw(
        {"style": "arb", "data": "absent"}
    )
    snapshot = LibrarySnapshot(value)
    assert (
        snapshot.entry_value("Q1.pulses.pi", kind="module")["waveform"]
        == "Q1.waveforms.absent"
    )
    assert snapshot.entry_value("Q1.waveforms.arb", kind="waveform")["data"] == "absent"
    with pytest.raises(ValueError, match="absent"):
        snapshot.get_module("Q1.pulses.pi")
    with pytest.raises(ValueError, match="assets"):
        snapshot.get_waveform("Q1.waveforms.arb")


def test_draft_snapshot_supplies_previous_writes_to_later_materialization(
    library_path: Path,
) -> None:
    library = open_library(library_path)
    with library.edit() as draft:
        draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.05))
        draft.set_module("Q1.pulses.pi", pulse())
        assert library.snapshot().module_keys() == ()
        assert draft.snapshot().value("Q1.pulses.pi.waveform.length") == 0.05
    assert open_library(library_path).snapshot().value("Q1.pulses.pi.gain") == 0.25


def test_body_failure_does_not_publish_or_write(library_path: Path) -> None:
    original = library_path.read_bytes()
    library = open_library(library_path)
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)

    def rejected_edit() -> None:
        with library.edit() as draft:
            draft.set_module("Q1.pulses.pi", pulse())
            raise RuntimeError("reject body")

    with pytest.raises(RuntimeError, match="reject body"):
        rejected_edit()
    assert library.snapshot().module_keys() == ()
    assert library_path.read_bytes() == original
    assert changes == []


def test_invalid_family_rejects_operation_without_poisoning_draft(
    library_path: Path,
) -> None:
    library = open_library(library_path)
    with library.edit() as draft:
        with pytest.raises(ValueError, match="family|partition"):
            draft.set_module("Q1.reset.pi", pulse())
        assert draft.snapshot().module_keys() == ()
        draft.set_module("Q1.pulses.pi", pulse())
    assert library.snapshot().module_keys() == ("Q1.pulses.pi",)


def test_edit_expires_and_nested_edit_is_rejected(library_path: Path) -> None:
    library = open_library(library_path)
    with library.edit() as draft, pytest.raises(RuntimeError), library.edit():
        pass
    with pytest.raises(RuntimeError):
        draft.snapshot()
    with pytest.raises(RuntimeError):
        draft.set_module("Q1.pulses.pi", pulse())


@pytest.mark.parametrize(
    "target", ["Q1.pulses.pi", "Q1.pulses.pi2", "Q1.reset.pi", "Q1.waveforms.pi"]
)
def test_invalid_rename_does_not_mutate_original(target: str) -> None:
    snapshot = LibrarySnapshot(document())
    before = snapshot.document()
    with pytest.raises(ValueError, match="destination|family|partition|kind"):
        snapshot.renamed("Q1.pulses.pi", target, kind="module")
    assert snapshot.document() == before


def test_active_rename_preserves_old_snapshot(
    library_path: Path,
) -> None:
    write_document(library_path, document())
    library = open_library(library_path)
    before = library.snapshot().entry_value("Q1.pulses.pi", kind="module")
    with library.edit() as draft:
        draft.rename("Q1.pulses.pi", "Q1.pulses.copy", kind="module")
        assert draft.snapshot().entry_value("Q1.pulses.copy", kind="module") == before
        assert library.snapshot().entry_value("Q1.pulses.pi", kind="module") == before
    reopened = open_library(library_path).snapshot()
    assert reopened.entry_value("Q1.pulses.copy", kind="module") == before
    assert "Q1.pulses.pi" not in reopened.module_keys()


def test_delete_candidate_and_live_delete_do_not_cascade(
    library_path: Path,
) -> None:
    write_document(library_path, document())
    library = open_library(library_path)
    before = library.snapshot()
    deleted = before.deleted("Q1.waveforms.base", kind="waveform")
    assert deleted.waveform_keys() == ()
    assert before.waveform_keys() == ("Q1.waveforms.base",)
    assert (
        deleted.entry_value("Q1.pulses.pi", kind="module")["waveform"]
        == "Q1.waveforms.base"
    )
    with library.edit() as draft:
        draft.delete("Q1.waveforms.base", kind="waveform")
    assert open_library(library_path).snapshot().waveform_keys() == ()
    with pytest.raises(ValueError, match="Q1.waveforms.base"):
        library.snapshot().get_module("Q1.pulses.pi")


def test_conflict_rejects_complete_losing_proposal(
    library_path: Path,
) -> None:
    write_document(library_path, document())
    losing, winning = open_library(library_path), open_library(library_path)

    def competing_edits() -> None:
        with losing.edit() as draft:
            draft.set_module("Q1.pulses.pi", pulse(gain=0.5))
            draft.set_module("Q1.pulses.extra", pulse(gain=0.7))
            with winning.edit() as winner:
                winner.set_module("Q1.pulses.pi", pulse(gain=0.6))

    with pytest.raises(ConflictError):
        competing_edits()
    current = open_library(library_path).snapshot()
    assert current.value("Q1.pulses.pi.gain") == 0.6
    assert "Q1.pulses.extra" not in current.module_keys()
    assert losing.snapshot().value("Q1.pulses.pi.gain") == 0.25


def test_refresh_publishes_after_install_and_unsubscribe_is_idempotent(
    library_path: Path,
) -> None:
    write_document(library_path, document())
    library, other = open_library(library_path), open_library(library_path)
    observed: list[tuple[str, float]] = []

    def observe(change: DocumentChange) -> None:
        gain = library.snapshot().value("Q1.pulses.pi.gain")
        assert isinstance(gain, float)
        observed.append((change.reason, gain))

    unsubscribe: Callable[[], None] = library.subscribe(observe)
    with other.edit() as draft:
        draft.set_module("Q1.pulses.pi", pulse(gain=0.5))
    assert library.snapshot().value("Q1.pulses.pi.gain") == 0.25
    assert library.refresh()
    assert observed == [("refresh", 0.5)]
    assert not library.refresh()
    unsubscribe()
    unsubscribe()
    with library.edit() as draft:
        draft.set_module("Q1.pulses.pi", pulse(gain=0.6))
    assert observed == [("refresh", 0.5)]


def test_invalid_refresh_retains_published_snapshot(library_path: Path) -> None:
    write_document(library_path, document())
    library = open_library(library_path)
    library_path.write_text("format: wrong\nformat_version: '2.0'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        library.refresh()
    assert library.snapshot().value("Q1.pulses.pi.gain") == 0.25


def test_inline_waveform_and_asset_capability_are_kept_out_of_entry_projection() -> (
    None
):
    value = document()
    value.components["Q1"].pulses["pi"].waveform = ConstWaveformCfg(length=0.12)
    value.components["Q1"].waveforms["arb"] = waveform_spec_from_raw(
        {"style": "arb", "data": "present"}
    )
    snapshot = LibrarySnapshot(value, assets=assets())
    assert snapshot.value("Q1.pulses.pi.waveform.length") == 0.12
    assert snapshot.entry_value("Q1.waveforms.arb", kind="waveform") == {
        "style": "arb",
        "data": "present",
    }
    waveform = snapshot.get_waveform("Q1.waveforms.arb")
    assert waveform.length == 0.05
    assert snapshot.waveform_spec("Q1.waveforms.arb").model_dump() == {
        "style": "arb",
        "data": "present",
    }


def test_pure_waveform_rename_does_not_evaluate_reference_cycle_or_asset() -> None:
    value = document()
    value.components["Q1"].waveforms["cycle"] = waveform_spec_from_raw(
        {"style": "flat_top", "length": 0.1, "raise_waveform": "Q1.waveforms.cycle"}
    )
    value.components["Q1"].waveforms["arb"] = waveform_spec_from_raw(
        {"style": "arb", "data": "absent"}
    )
    snapshot = LibrarySnapshot(value)
    renamed = snapshot.renamed(
        "Q1.waveforms.cycle", "Q1.waveforms.copy", kind="waveform"
    )
    assert (
        renamed.entry_value("Q1.waveforms.copy", kind="waveform")["raise_waveform"]
        == "Q1.waveforms.cycle"
    )
    moved = renamed.renamed("Q1.waveforms.arb", "R1.waveforms.copy", kind="waveform")
    assert moved.entry_value("R1.waveforms.copy", kind="waveform")["data"] == "absent"


def test_atomic_replace_failure_does_not_publish_candidate(
    library_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_document(library_path, document())
    library = open_library(library_path)
    original = library_path.read_bytes()
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)

    def reject_replace(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("replace rejected")

    monkeypatch.setattr(os, "replace", reject_replace)
    with pytest.raises(OSError, match="replace rejected"), library.edit() as draft:
        draft.set_module("Q1.pulses.pi", pulse(gain=0.5))
    assert library.snapshot().value("Q1.pulses.pi.gain") == 0.25
    assert library_path.read_bytes() == original
    assert changes == []


def test_obsolete_nested_readout_is_reported_while_flat_readout_loads(
    library_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = {
        "format": "zcu.module-library",
        "format_version": "2.0",
        "components": {
            "Q1": {
                "readout": {
                    "probe": {
                        "type": "readout/pulse",
                        "gen_ch": 1,
                        "nqz": 1,
                        "freq": 100.0,
                        "gain": 0.5,
                        "waveform": "Q1.waveforms.base",
                        "ro_ch": 0,
                        "ro_length": 1.0,
                    },
                    "broken_probe": {
                        "type": "readout/pulse",
                        "pulse_cfg": {
                            "type": "pulse",
                            "ch": 0,
                            "nqz": 1,
                            "freq": 100.0,
                            "gain": 0.5,
                            "waveform": {"style": "const", "length": 0.5},
                        },
                        "ro_cfg": {
                            "type": "readout/direct",
                            "ro_ch": 0,
                            "ro_freq": 100.0,
                            "ro_length": 1.0,
                        },
                    },
                },
                "waveforms": {"base": {"style": "const", "length": 0.5}},
            }
        },
    }
    path = library_path
    path.write_text(json.dumps(raw), encoding="utf-8")
    original = path.read_bytes()

    with caplog.at_level(logging.WARNING):
        library = open_library(library_path)

    snapshot = library.snapshot()
    assert snapshot.module_keys() == ("Q1.readout.probe",)
    cfg = snapshot.get_module("Q1.readout.probe")
    assert isinstance(cfg, PulseReadoutCfg)
    assert (cfg.gen_ch, cfg.ro_ch, cfg.freq) == (1, 0, 100.0)
    assert cfg.waveform.length == 0.5
    with pytest.raises(KeyError, match="broken_probe"):
        snapshot.module_spec("Q1.readout.broken_probe")
    assert path.read_bytes() == original
    assert library.snapshot().invalid_entries()["Q1.readout.broken_probe"]
    assert not library.refresh()


def test_mux_and_pulse_readouts_roundtrip_and_materialize_fixed_snapshot(
    library_path: Path,
) -> None:
    from zcu_tools import cfg_model

    raw: cfg_model.CfgInputMap = {
        "type": "readout/mux",
        "gen_ch": 9,
        "nqz": 2,
        "mixer_freq": 6000.0,
        "length": 2.0,
        "ro_length": 1.5,
        "tones": [
            {
                "freq": 6020.0,
                "gain": 0.05,
                "phase": 90.0,
                "ro_ch": 4,
                "label": "upper",
            },
            {"freq": 5980.0, "gain": -0.04, "phase": 0.0, "ro_ch": 1, "label": ""},
        ],
    }
    library = open_library(library_path)
    with library.edit() as draft:
        draft.set_module("Q1.readout.mux", cfg_model.module_spec_from_raw(raw))
        draft.set_module(
            "Q1.readout.pulse",
            cfg_model.module_spec_from_raw(
                {
                    "type": "readout/pulse",
                    "gen_ch": 0,
                    "nqz": 1,
                    "freq": 100.0,
                    "gain": 0.1,
                    "ro_ch": 0,
                    "ro_length": 1.0,
                    "waveform": {"style": "const", "length": 0.5},
                }
            ),
        )
    fixed = open_library(library_path).snapshot()
    spec = fixed.module_spec("Q1.readout.mux")
    assert isinstance(spec, cfg_model.MuxReadoutSpec)
    expected_tones = TypeAdapter(list[dict[str, object]]).validate_python(raw["tones"])
    assert fixed.entry_value("Q1.readout.mux", kind="module")["tones"] == [
        {**tone, "classifier": None} for tone in expected_tones
    ]
    cfg = cfg_model.ModuleCfgFactory.from_raw(
        "Q1.readout.mux", context=fixed.validation_context()
    )
    assert isinstance(cfg, cfg_model.MuxReadoutCfg)
    assert [(tone.freq, tone.ro_ch) for tone in cfg.tones] == [(6020.0, 4), (5980.0, 1)]
    assert isinstance(fixed.get_module("Q1.readout.pulse"), PulseReadoutCfg)
    with library.edit() as draft:
        draft.set_module("Q1.readout.mux", spec.with_updates(length=3.0))
    library_path.unlink()
    cfg.tones.reverse()
    assert fixed.get_module("Q1.readout.mux").to_dict()["tones"] == raw["tones"]
    assert fixed.get_module("Q1.readout.mux").to_dict()["length"] == 2.0
