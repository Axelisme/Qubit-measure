"""Specs, materialization, provenance and single-file publication contracts."""

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    WaveformSamples,
    waveform_spec_from_raw,
)
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
from zcu_tools.resources.entry.ledger import RecordsLedger
from zcu_tools.resources.entry.ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    LedgerEvent,
)
from zcu_tools.resources.entry.provenance import Provenance, ProvenanceData

from tests.program.v2.support.waveform_assets import InMemoryWaveformAsset

ENTRY_ID = "78fa0941-f9ce-49d4-9665-4eb9302b8162"
EVENT_ID = "b13fea4c-1678-4b5c-b7a2-edb975c20611"
AT = "2026-10-04T00:00:00Z"


def pulse(*, gain: float = 0.25, waveform: str = "Q1.waveforms.base") -> PulseSpec:
    return PulseSpec(ch=0, nqz=1, freq=100.0, gain=gain, waveform=waveform)


def document() -> ModuleLibraryDocument:
    return ModuleLibraryDocument(
        format="zcu.module-library",
        format_version="1.0",
        components={
            "Q1": ComponentLibrary(
                pulses={"pi": pulse(), "pi2": pulse(gain=0.125)},
                waveforms={"base": ConstWaveformCfg(length=0.05)},
            )
        },
    )


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
def files(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        "format: zcu.module-library\nformat_version: '1.0'\n", encoding="utf-8"
    )
    records = tmp_path / "records"
    records.mkdir()
    return path, records / "ledger.jsonl"


def open_library(files: tuple[Path, Path]) -> ModuleLibrary:
    path, ledger = files
    return ModuleLibrary(path, assets=assets(), ledger=ledger, entry_id=ENTRY_ID)


def write_document(files: tuple[Path, Path], value: ModuleLibraryDocument) -> None:
    files[0].write_text(value.model_dump_json(), encoding="utf-8")


def local_source(files: tuple[Path, Path]) -> Provenance:
    RecordsLedger(files[1].parent, entry_id=ENTRY_ID).append(
        LedgerEvent(
            id=EVENT_ID,
            kind="accepted",
            at=AT,
            entry_id=ENTRY_ID,
            origin="notebook",
            payload=AcceptedPayload(
                writes=(AcceptedWrite(path="Q1.pulses.pi.gain", value=0.25),),
                analysis_summary={},
            ),
        ),
        record={"summary": {"gain": 0.25}},
    )
    return Provenance(EVENT_ID, "rabi", None, AT, 0.01)


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
    files: tuple[Path, Path],
) -> None:
    path, ledger = files
    library = ModuleLibrary(path, ledger=ledger, entry_id=ENTRY_ID)
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
    files: tuple[Path, Path],
) -> None:
    original = files[0].read_bytes()
    library = open_library(files)
    assert library.source_path == files[0].resolve()
    assert library.snapshot().module_keys() == ()
    assert library.snapshot().waveform_keys() == ()
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)
    with library.edit() as draft:
        assert draft.snapshot().document().components == {}
    assert files[0].read_bytes() == original
    assert changes == []


def test_missing_document_is_not_created(files: tuple[Path, Path]) -> None:
    files[0].unlink()
    with pytest.raises(FileNotFoundError):
        open_library(files)
    assert not files[0].exists()


@pytest.mark.parametrize(
    "content",
    [
        "waveforms: {}\nmodules: {}\n",
        "format: wrong\nformat_version: '1.0'\n",
        "format: zcu.module-library\nformat_version: '2.0'\n",
        "format: zcu.module-library\nformat_version: '1.0'\nunknown: true\n",
    ],
)
def test_open_rejects_invalid_document_without_rewriting(
    files: tuple[Path, Path], content: str
) -> None:
    files[0].write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="format|header|version|Extra"):
        open_library(files)
    assert files[0].read_text(encoding="utf-8") == content


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
    files: tuple[Path, Path],
) -> None:
    write_document(files, document())
    library = open_library(files)
    first = library.snapshot()
    changed = document()
    changed.components["Q1"].pulses["pi"].gain = 0.5
    write_document(files, changed)
    files[0].unlink()
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
    assert snapshot.meta("Q1.pulses.pi.gain") is None


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
    files: tuple[Path, Path],
) -> None:
    library = open_library(files)
    with library.edit() as draft:
        draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.05))
        draft.set_module("Q1.pulses.pi", pulse())
        assert library.snapshot().module_keys() == ()
        assert draft.snapshot().value("Q1.pulses.pi.waveform.length") == 0.05
    assert open_library(files).snapshot().value("Q1.pulses.pi.gain") == 0.25
    source = library.snapshot().meta("Q1.pulses.pi.gain")
    assert source is not None
    assert source.source == "manual"


def test_body_failure_does_not_publish_or_write(files: tuple[Path, Path]) -> None:
    original = files[0].read_bytes()
    library = open_library(files)
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)

    def rejected_edit() -> None:
        with library.edit() as draft:
            draft.set_module("Q1.pulses.pi", pulse())
            raise RuntimeError("reject body")

    with pytest.raises(RuntimeError, match="reject body"):
        rejected_edit()
    assert library.snapshot().module_keys() == ()
    assert files[0].read_bytes() == original
    assert changes == []


def test_invalid_family_rejects_operation_without_poisoning_draft(
    files: tuple[Path, Path],
) -> None:
    library = open_library(files)
    with library.edit() as draft:
        with pytest.raises(ValueError, match="family|partition"):
            draft.set_module("Q1.reset.pi", pulse())
        assert draft.snapshot().module_keys() == ()
        draft.set_module("Q1.pulses.pi", pulse())
    assert library.snapshot().module_keys() == ("Q1.pulses.pi",)


def test_edit_expires_and_nested_edit_is_rejected(files: tuple[Path, Path]) -> None:
    library = open_library(files)
    with library.edit() as draft, pytest.raises(RuntimeError), library.edit():
        pass
    with pytest.raises(RuntimeError):
        draft.snapshot()
    with pytest.raises(RuntimeError):
        draft.set_module("Q1.pulses.pi", pulse())


def test_manual_replacement_clears_previous_sources_and_clone_marker() -> None:
    value = document()
    value.provenance["Q1.pulses.pi.gain"] = ProvenanceData(
        source=EVENT_ID,
        kind="rabi",
        run_id=None,
        at=AT,
        stderr=0.01,
        cloned_from={"entry_id": ENTRY_ID, "point": "first"},
    )
    snapshot = LibrarySnapshot(value)
    replacement = snapshot.replaced_module("Q1.pulses.pi", pulse(gain=0.5))
    source = replacement.meta("Q1.pulses.pi.gain")
    assert source is not None
    assert source.source == "manual"
    assert source.cloned_from is None
    original_source = snapshot.meta("Q1.pulses.pi.gain")
    assert original_source is not None
    assert original_source.source == EVENT_ID
    assert replacement.value("Q1.pulses.pi.gain") == 0.5
    envelope = replacement.replaced_waveform(
        "Q1.waveforms.base", ConstWaveformCfg(length=0.1)
    )
    assert envelope.value("Q1.pulses.pi.waveform.length") == 0.1
    assert replacement.value("Q1.pulses.pi.waveform.length") == 0.05


def test_source_acceptance_round_trips_local_evidence_without_changing_value(
    files: tuple[Path, Path],
) -> None:
    source = local_source(files)
    write_document(files, document())
    library = open_library(files)
    before = library.snapshot().entry_value("Q1.pulses.pi", kind="module")
    with library.edit() as draft:
        draft.accept_source("Q1.pulses.pi", provenance=source)
        assert draft.snapshot().meta("Q1.pulses.pi.gain") == source
    snapshot = open_library(files).snapshot()
    assert snapshot.entry_value("Q1.pulses.pi", kind="module") == before
    assert snapshot.meta("Q1.pulses.pi.gain") == source
    evidence = RecordsLedger(files[1].parent, entry_id=ENTRY_ID).get(EVENT_ID)
    assert evidence.record == {"summary": {"gain": 0.25}}


def test_invalid_source_leaves_draft_and_disk_unchanged(
    files: tuple[Path, Path],
) -> None:
    write_document(files, document())
    library = open_library(files)
    original = files[0].read_bytes()
    with library.edit() as draft:
        with pytest.raises(ValueError, match="does not exist") as failure:
            draft.accept_source(
                "Q1.pulses.pi", provenance=Provenance(EVENT_ID, "rabi", None, AT, 0.01)
            )
        assert isinstance(failure.value.__cause__, KeyError)
        assert draft.snapshot().meta("Q1.pulses.pi.gain") is None
    assert files[0].read_bytes() == original


def test_open_validates_nonmanual_source_and_attachment(
    files: tuple[Path, Path],
) -> None:
    local_source(files)
    value = document()
    value.provenance["Q1.pulses.pi.gain"] = ProvenanceData(
        source=EVENT_ID,
        kind="rabi",
        run_id=None,
        at=AT,
        stderr=0.01,
    )
    write_document(files, value)
    open_library(files)
    (files[1].parent / f"{EVENT_ID}.json").unlink()
    with pytest.raises(FileNotFoundError):
        open_library(files)


def test_rename_preserves_complete_metadata_and_segment_boundaries() -> None:
    value = document()
    value.components["Q1"].pulses["pi"].waveform = "Q1.waveforms.absent"
    value.provenance = {
        "Q1.pulses.pi": ProvenanceData(
            source="manual", kind=None, run_id=None, at=AT, stderr=None
        ),
        "Q1.pulses.pi.gain": ProvenanceData(
            source=EVENT_ID,
            kind="rabi",
            run_id=None,
            at=AT,
            stderr=0.01,
            cloned_from={"entry_id": ENTRY_ID, "point": "first"},
        ),
        "Q1.pulses.pi2.gain": ProvenanceData(
            source="manual", kind=None, run_id=None, at=AT, stderr=None
        ),
    }
    snapshot = LibrarySnapshot(value)
    before = snapshot.entry_state("Q1.pulses.pi", kind="module")
    assert set(before.provenance) == {(), ("gain",)}
    renamed = snapshot.renamed("Q1.pulses.pi", "R1.pulses.copy", kind="module")
    assert renamed.entry_state("R1.pulses.copy", kind="module") == before
    assert renamed.meta("Q1.pulses.pi.gain") is None
    assert renamed.meta("Q1.pulses.pi2.gain") == snapshot.meta("Q1.pulses.pi2.gain")
    assert snapshot.entry_state("Q1.pulses.pi", kind="module") == before
    before.value["gain"] = 0.9
    clone = before.provenance[("gain",)].cloned_from
    assert clone is not None
    clone["point"] = "changed"
    assert renamed.entry_state("R1.pulses.copy", kind="module").value["gain"] == 0.25
    moved_source = renamed.meta("R1.pulses.copy.gain")
    assert moved_source is not None
    assert moved_source.cloned_from is not None
    assert moved_source.cloned_from["point"] == "first"


@pytest.mark.parametrize(
    "target", ["Q1.pulses.pi", "Q1.pulses.pi2", "Q1.reset.pi", "Q1.waveforms.pi"]
)
def test_invalid_rename_does_not_mutate_original(target: str) -> None:
    snapshot = LibrarySnapshot(document())
    before = snapshot.document()
    with pytest.raises(ValueError, match="destination|family|partition|kind"):
        snapshot.renamed("Q1.pulses.pi", target, kind="module")
    assert snapshot.document() == before


def test_active_rename_commits_spec_and_sources_together_without_ledger_append(
    files: tuple[Path, Path],
) -> None:
    source = local_source(files)
    write_document(files, document())
    library = open_library(files)
    with library.edit() as draft:
        draft.accept_source("Q1.pulses.pi", provenance=source)
    before = library.snapshot().entry_state("Q1.pulses.pi", kind="module")
    ledger_bytes = files[1].read_bytes()
    with library.edit() as draft:
        draft.rename("Q1.pulses.pi", "Q1.pulses.copy", kind="module")
        assert draft.snapshot().entry_state("Q1.pulses.copy", kind="module") == before
        assert library.snapshot().entry_state("Q1.pulses.pi", kind="module") == before
    reopened = open_library(files).snapshot()
    assert reopened.entry_state("Q1.pulses.copy", kind="module") == before
    assert "Q1.pulses.pi" not in reopened.module_keys()
    assert files[1].read_bytes() == ledger_bytes


def test_delete_candidate_and_live_delete_do_not_cascade(
    files: tuple[Path, Path],
) -> None:
    write_document(files, document())
    library = open_library(files)
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
    assert open_library(files).snapshot().waveform_keys() == ()
    with pytest.raises(ValueError, match="Q1.waveforms.base"):
        library.snapshot().get_module("Q1.pulses.pi")


def test_conflict_rejects_complete_losing_proposal_and_sources(
    files: tuple[Path, Path],
) -> None:
    write_document(files, document())
    losing, winning = open_library(files), open_library(files)

    def competing_edits() -> None:
        with losing.edit() as draft:
            draft.set_module("Q1.pulses.pi", pulse(gain=0.5))
            draft.set_module("Q1.pulses.extra", pulse(gain=0.7))
            with winning.edit() as winner:
                winner.set_module("Q1.pulses.pi", pulse(gain=0.6))

    with pytest.raises(ConflictError):
        competing_edits()
    current = open_library(files).snapshot()
    assert current.value("Q1.pulses.pi.gain") == 0.6
    assert current.meta("Q1.pulses.pi.gain") == winning.snapshot().meta(
        "Q1.pulses.pi.gain"
    )
    assert "Q1.pulses.extra" not in current.module_keys()
    assert losing.snapshot().value("Q1.pulses.pi.gain") == 0.25


def test_refresh_publishes_after_install_and_unsubscribe_is_idempotent(
    files: tuple[Path, Path],
) -> None:
    write_document(files, document())
    library, other = open_library(files), open_library(files)
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


def test_invalid_refresh_retains_published_snapshot(files: tuple[Path, Path]) -> None:
    write_document(files, document())
    library = open_library(files)
    files[0].write_text("format: wrong\nformat_version: '1.0'\n", encoding="utf-8")
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


def test_atomic_replace_failure_does_not_publish_candidate_or_sources(
    files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_document(files, document())
    library = open_library(files)
    original = files[0].read_bytes()
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)

    def reject_replace(_source: str | Path, _destination: str | Path) -> None:
        raise OSError("replace rejected")

    monkeypatch.setattr(os, "replace", reject_replace)
    with pytest.raises(OSError, match="replace rejected"), library.edit() as draft:
        draft.set_module("Q1.pulses.pi", pulse(gain=0.5))
    assert library.snapshot().value("Q1.pulses.pi.gain") == 0.25
    assert library.snapshot().meta("Q1.pulses.pi.gain") is None
    assert files[0].read_bytes() == original
    assert changes == []


def test_source_refresh_failure_retains_snapshot_and_recovery_publishes(
    files: tuple[Path, Path],
) -> None:
    write_document(files, document())
    library = open_library(files)
    changes: list[DocumentChange] = []
    library.subscribe(changes.append)
    changed = document()
    changed.provenance["Q1.pulses.pi.gain"] = ProvenanceData(
        source=EVENT_ID,
        kind="rabi",
        run_id=None,
        at=AT,
        stderr=0.01,
    )
    write_document(files, changed)
    with pytest.raises(ValueError, match="does not exist"):
        library.refresh()
    assert library.snapshot().meta("Q1.pulses.pi.gain") is None
    assert changes == []
    source = local_source(files)
    assert library.refresh()
    assert library.snapshot().meta("Q1.pulses.pi.gain") == source
    assert len(changes) == 1
