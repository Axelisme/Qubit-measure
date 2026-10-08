"""Cross-entry clone imports remain resolvable from the destination alone."""

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from ruamel.yaml import YAML
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.resources.context.library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    ImportPayload,
    LedgerEvent,
    PointView,
    Provenance,
    ResultEntry,
    SavedPayload,
)
from zcu_tools.resources.entry.provenance import ProvenanceData
from zcu_tools.resources.entry.schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    PointDocument,
)


@pytest.fixture
def foreign_recorded_point(
    entry_roots: tuple[Path, Path],
) -> tuple[ResultEntry, PointView, LedgerEvent]:
    results, database = entry_roots
    foreign = ResultEntry.create("other", result_root=results, database_root=database)
    foreign.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0)
    point = foreign.new_point("source")
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at="2026-10-06T02:00:00Z",
        entry_id=foreign.entry_id,
        origin="gui",
        payload=AcceptedPayload(
            writes=(
                AcceptedWrite(path="Q1.rate", value=5300.0),
                AcceptedWrite(path="Q1.duration", value=12.0),
            ),
            analysis_summary={"method": "fit"},
        ),
    )
    foreign.ledger.append(event, record={"fit": {"samples": [1, True, None, "µ"]}})
    provenance = Provenance(event.id, "fit", None, event.at, 0.18)
    with point.edit() as draft:
        draft.set("Q1.rate", 5300.0, provenance=provenance)
        draft.set("Q1.duration", 12.0, provenance=provenance)
        draft.description = "source environment"
    (results / "other/points/source/module_cfg.yaml").write_text(
        "format: zcu.module-library\nformat_version: '1.0'\ncomponents: {Q1: {}}\n",
        encoding="utf-8",
    )
    return foreign, point, event


def write_source_library(
    entry: ResultEntry, event: LedgerEvent
) -> ModuleLibraryDocument:
    """Publish valid Specs with shared, library-only and manual source leaves."""
    value = ModuleLibraryDocument(
        format="zcu.module-library",
        format_version="1.0",
        components={
            "Q1": ComponentLibrary(
                pulses={
                    "pi": PulseSpec(
                        ch=0,
                        nqz=1,
                        freq=100.0,
                        gain=0.25,
                        waveform="Q1.waveforms.missing",
                    )
                },
                waveforms={"base": ConstWaveformCfg(length=0.05)},
            )
        },
        provenance={
            "Q1.pulses.pi.gain": ProvenanceData.from_provenance(
                Provenance(event.id, "rabi", None, event.at, 0.01)
            ),
            "Q1.waveforms.base.length": ProvenanceData.from_provenance(
                Provenance("manual", None, None, event.at, None)
            ),
        },
    )
    (entry.result_path / "points/source/module_cfg.yaml").write_text(
        value.model_dump_json(), encoding="utf-8"
    )
    return value


@pytest.mark.parametrize("source_mode", ["label", "view"])
@pytest.mark.parametrize("same_entry", [True, False])
def test_clone_rewrites_library_and_point_sources_with_one_shared_import(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    source_mode: str,
    same_entry: bool,
) -> None:
    foreign, point, event = foreign_recorded_point
    source_document = write_source_library(foreign, event)
    source_file = foreign.result_path / "points/source/module_cfg.yaml"
    before = source_file.read_bytes()
    destination = foreign if same_entry else entry
    argument = (
        point if source_mode == "view" else ("source" if same_entry else "other/source")
    )
    cloned = destination.new_point("library-copy", clone_from=argument)
    stored = DocumentStore(
        destination.result_path / "points/library-copy/module_cfg.yaml",
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
    ).snapshot()
    library = LibrarySnapshot(stored)
    assert stored.components == source_document.components
    spec = library.module_spec("Q1.pulses.pi")
    assert isinstance(spec, PulseSpec)
    assert spec.waveform == "Q1.waveforms.missing"
    copied = library.meta("Q1.pulses.pi.gain")
    assert copied is not None
    point_meta = cloned.meta("Q1.rate")
    assert point_meta is not None
    expected_id = event.id if same_entry else destination.ledger.events()[0].id
    assert copied == replace(
        source_document.provenance["Q1.pulses.pi.gain"].to_provenance(),
        source=expected_id,
        cloned_from={"entry_id": foreign.entry_id, "point": "source"},
    )
    assert copied.source == point_meta.source
    assert len(destination.ledger.events()) == 1
    manual = library.meta("Q1.waveforms.base.length")
    assert manual is not None
    assert manual.source == "manual"
    assert manual.cloned_from == {"entry_id": foreign.entry_id, "point": "source"}
    assert source_file.read_bytes() == before


def test_library_only_import_keeps_evidence_after_source_entry_moves(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, _point, event = foreign_recorded_point
    original = foreign.ledger.get(event.id)
    write_source_library(foreign, event)
    # Point has only manual sources; this event is now library-only.
    with foreign.use_point("source").edit() as draft:
        draft.set("Q1.rate", 5300.0)
        draft.set("Q1.duration", 12.0)
    entry.new_point("library-only", clone_from="other/source", origin="mcp")
    imported = entry.ledger.events()[0]
    assert len(entry.ledger.events()) == 1
    assert imported.origin == "mcp"
    assert isinstance(imported.payload, ImportPayload)
    assert imported.payload.source_event_json == original.event_json
    assert imported.payload.source_record == original.event.record
    results, database = entry_roots
    (results / "other").rename(results / "moved-other")
    (database / "other").rename(database / "moved-other")
    stored = DocumentStore(
        entry.result_path / "points/library-only/module_cfg.yaml",
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
    ).snapshot()
    copied = stored.provenance["Q1.pulses.pi.gain"]
    assert copied.source == imported.id
    assert copied.cloned_from == {"entry_id": foreign.entry_id, "point": "source"}
    assert entry.ledger.get(copied.source).record == original.record


def test_clone_imports_distinct_library_and_point_events(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, _point, event = foreign_recorded_point
    second = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at=event.at,
        entry_id=foreign.entry_id,
        origin="notebook",
        payload=AcceptedPayload(
            writes=(AcceptedWrite(path="Q1.waveforms.base.length", value=0.05),),
            analysis_summary={},
        ),
    )
    foreign.ledger.append(second, record={"length": 0.05})
    document = write_source_library(foreign, event)
    document.provenance["Q1.waveforms.base.length"] = ProvenanceData.from_provenance(
        Provenance(second.id, "fit", None, second.at, 0.002)
    )
    (foreign.result_path / "points/source/module_cfg.yaml").write_text(
        document.model_dump_json(), encoding="utf-8"
    )
    point = entry.new_point("distinct", clone_from="other/source")
    imports: dict[str, str] = {}
    for imported in entry.ledger.events():
        assert isinstance(imported.payload, ImportPayload)
        imports[imported.payload.source_event_id] = imported.id
    assert set(imports) == {event.id, second.id}
    copied = DocumentStore(
        entry.result_path / "points/distinct/module_cfg.yaml",
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
    ).snapshot()
    point_meta = point.meta("Q1.rate")
    assert point_meta is not None
    assert (
        point_meta.source
        == copied.provenance["Q1.pulses.pi.gain"].source
        == imports[event.id]
    )
    assert copied.provenance["Q1.waveforms.base.length"].source == imports[second.id]
    assert entry.ledger.get(imports[second.id]).record == {"length": 0.05}


def test_manual_library_clone_produces_no_import(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, _point, event = foreign_recorded_point
    with foreign.use_point("source").edit() as draft:
        draft.set("Q1.rate", 5300.0)
        draft.set("Q1.duration", 12.0)
    document = write_source_library(foreign, event)
    document.provenance["Q1.pulses.pi.gain"].source = "manual"
    (foreign.result_path / "points/source/module_cfg.yaml").write_text(
        document.model_dump_json(), encoding="utf-8"
    )
    entry.new_point("manual-library", clone_from="other/source")
    copied = DocumentStore(
        entry.result_path / "points/manual-library/module_cfg.yaml",
        ModuleLibraryDocument,
        format=MODULE_LIBRARY_FORMAT,
        supported_version=MODULE_LIBRARY_VERSION,
    ).snapshot()
    assert entry.ledger.events() == ()
    assert all(record.source == "manual" for record in copied.provenance.values())
    assert all(
        record.cloned_from == {"entry_id": foreign.entry_id, "point": "source"}
        for record in copied.provenance.values()
    )


def test_failed_library_commit_removes_new_point_but_keeps_completed_import(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign, _point, event = foreign_recorded_point
    write_source_library(foreign, event)
    source = foreign.result_path / "points/source"
    before = {
        name: (source / name).read_bytes() for name in ("point.yaml", "module_cfg.yaml")
    }
    target = entry.result_path / "points/io-failure/module_cfg.yaml"
    replace_file = os.replace

    def fail_library_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == target:
            raise OSError("library commit failed")
        replace_file(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_library_replace)
        with pytest.raises(OSError, match="library commit failed"):
            entry.new_point("io-failure", clone_from="other/source")
    assert not target.parent.exists()
    assert entry.list_points() == []
    assert {name: (source / name).read_bytes() for name in before} == before
    imports = entry.ledger.events()
    assert len(imports) == 1
    assert isinstance(imports[0].payload, ImportPayload)
    assert imports[0].payload.source_event_id == event.id
    assert entry.ledger.get(imports[0].id).record == foreign.ledger.get(event.id).record


@pytest.mark.parametrize(
    "failure", ["missing_event", "attachment", "source_schema", "source_path", "header"]
)
def test_library_clone_failure_cleans_destination_and_keeps_source(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    failure: str,
) -> None:
    foreign, _point, event = foreign_recorded_point
    # Isolate source evidence to the library, not the existing point seam.
    with foreign.use_point("source").edit() as draft:
        draft.set("Q1.rate", 5300.0)
        draft.set("Q1.duration", 12.0)
    document = write_source_library(foreign, event)
    source = foreign.result_path / "points/source/module_cfg.yaml"
    if failure == "missing_event":
        (foreign.result_path / "records/ledger.jsonl").write_text("", encoding="utf-8")
    elif failure == "attachment":
        (foreign.result_path / f"records/{event.id}.json").unlink()
    elif failure == "source_schema":
        source.write_text(
            source.read_text().replace('"stderr":0.01', '"stderr":"invalid"'),
            encoding="utf-8",
        )
    elif failure == "source_path":
        document.provenance["Q1.pulses.pi.absent"] = document.provenance.pop(
            "Q1.pulses.pi.gain"
        )
        source.write_text(document.model_dump_json(), encoding="utf-8")
    else:
        source.write_text(
            source.read_text().replace(
                '"format_version":"1.0"', '"format_version":"2.0"'
            ),
            encoding="utf-8",
        )
    before = source.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        entry.new_point("failed-library", clone_from="other/source")
    assert not (entry.result_path / "points/failed-library").exists()
    assert entry.list_points() == []
    assert source.read_bytes() == before


@pytest.mark.parametrize("same_entry", [False, True])
@pytest.mark.parametrize("manual_source", [False, True])
def test_clone_preserves_forward_minor_point_provenance(
    entry: ResultEntry,
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    same_entry: bool,
    manual_source: bool,
) -> None:
    foreign, recorded, event = foreign_recorded_point
    source_point = foreign.new_point("manual") if manual_source else recorded
    label = "manual" if manual_source else "source"
    source_file = foreign.result_path / f"points/{label}/point.yaml"
    source_store = DocumentStore(
        source_file,
        PointDocument,
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    )
    with source_store.edit() as document:
        document.format_version = "1.1"
        document.provenance["Q1.rate"]["future_quality"] = {
            "checked": True,
            "method": "future-validator",
        }
    before = source_file.read_bytes()
    original_metadata = source_store.snapshot().provenance["Q1.rate"]
    destination = foreign if same_entry else entry
    events_before = len(destination.ledger.events())

    cloned = destination.new_point("future-copy", clone_from=source_point)

    stored = DocumentStore(
        destination.result_path / "points/future-copy/point.yaml",
        PointDocument,
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    ).snapshot()
    metadata = stored.provenance["Q1.rate"]
    public_metadata = cloned.meta("Q1.rate")
    assert public_metadata is not None
    if manual_source:
        expected_id = "manual"
    elif same_entry:
        expected_id = event.id
    else:
        expected_id = destination.ledger.events()[-1].id
    assert public_metadata.source == expected_id
    assert metadata == {
        **original_metadata,
        "source": expected_id,
        "cloned_from": {"entry_id": foreign.entry_id, "point": label},
    }
    assert cloned.Q1.rate == source_point.Q1.rate
    assert len(destination.ledger.events()) == events_before + int(
        not same_entry and not manual_source
    )
    if not same_entry and not manual_source:
        imported = destination.ledger.get(expected_id)
        assert isinstance(imported.event.payload, ImportPayload)
        assert imported.event.payload.source_event_id == event.id
    assert source_file.read_bytes() == before


@pytest.mark.parametrize("source_mode", ["label", "view"])
@pytest.mark.parametrize("origin", ["notebook", "mcp"])
def test_cross_entry_clone_imports_once_and_uses_published_values(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    source_mode: str,
    origin: Literal["notebook", "mcp"],
) -> None:
    results, _database = entry_roots
    foreign, stale_point, original_event = foreign_recorded_point
    provenance = stale_point.meta("Q1.rate")
    assert provenance is not None
    fresh = foreign.use_point("source")
    with fresh.edit() as draft:
        draft.set("Q1.rate", 5400.0, provenance=provenance)
    assert stale_point.Q1.rate == 5300.0
    original = foreign.ledger.get(original_event.id)
    source_path = results / "other/points/source"
    before = {
        name: (source_path / name).read_bytes()
        for name in ("point.yaml", "module_cfg.yaml")
    }
    # Clone does not depend on the destination setup.
    (results / "entry/setup.yaml").write_text("invalid: [", encoding="utf-8")
    source_arg = "other/source" if source_mode == "label" else stale_point
    clone = entry.new_point("target", clone_from=source_arg, origin=origin)
    assert clone.Q1.rate == 5400.0
    assert clone.Q1.duration == 12.0
    assert clone.description == "source environment"
    assert (results / "entry/points/target/module_cfg.yaml").read_bytes() == before[
        "module_cfg.yaml"
    ]
    imports = entry.ledger.events()
    assert len(imports) == 1
    imported = imports[0]
    assert imported.id != original_event.id
    assert imported.kind == "import"
    assert imported.entry_id == entry.entry_id
    assert imported.origin == origin
    assert isinstance(imported.payload, ImportPayload)
    assert imported.payload.source_entry_id == foreign.entry_id
    assert imported.payload.source_event_id == original_event.id
    assert imported.payload.source_event_json == original.event_json
    assert imported.payload.source_record == original.event.record
    assert imported.record == f"records/{imported.id}.json"
    assert entry.ledger.get(imported.id).record == original.record
    for path in ("Q1.rate", "Q1.duration"):
        source_meta = fresh.meta(path)
        assert source_meta is not None
        assert clone.meta(path) == replace(
            source_meta,
            source=imported.id,
            cloned_from={"entry_id": foreign.entry_id, "point": "source"},
        )
    manual = clone.meta("general.description")
    assert manual is not None
    assert manual.source == "manual"
    assert manual.cloned_from == {"entry_id": foreign.entry_id, "point": "source"}
    assert {name: (source_path / name).read_bytes() for name in before} == before
    assert entry.use_point("target").meta("Q1.rate") == clone.meta("Q1.rate")


def test_cross_entry_manual_clone_needs_no_import_and_same_entry_keeps_source(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, recorded, event = foreign_recorded_point
    foreign.new_point("manual")
    manual = entry.new_point("manual-copy", clone_from="other/manual")
    assert manual.Q1.rate == 5000.0
    assert entry.ledger.events() == ()
    source_meta = manual.meta("Q1.rate")
    assert source_meta is not None
    assert source_meta.source == "manual"
    assert source_meta.cloned_from == {"entry_id": foreign.entry_id, "point": "manual"}
    same_entry = foreign.new_point("local", clone_from=recorded)
    assert len(foreign.ledger.events()) == 1
    local_meta = same_entry.meta("Q1.rate")
    assert local_meta is not None
    assert local_meta.source == event.id
    assert local_meta.cloned_from == {"entry_id": foreign.entry_id, "point": "source"}


def test_nested_import_remains_locally_resolvable_after_source_entries_move(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    results, database = entry_roots
    foreign, _point, event = foreign_recorded_point
    # Future-minor unknown data and exact whitespace are historical evidence.
    ledger_path = results / "other/records/ledger.jsonl"
    raw_event = json.loads(ledger_path.read_text())
    raw_event["format_version"] = "1.2"
    raw_event["future_header"] = {"confidence": None}
    raw_event["payload"]["future_payload"] = [1, True]
    raw = "  " + json.dumps(raw_event, ensure_ascii=False) + "  "
    ledger_path.write_text(raw + "\n", encoding="utf-8")
    entry.new_point("first", clone_from="other/source")
    first_import = entry.ledger.events()[0]
    first_evidence = entry.ledger.get(first_import.id)
    third = ResultEntry.create("third", result_root=results, database_root=database)
    cloned = third.new_point("final", clone_from="entry/first")
    final_import = third.ledger.events()[0]
    assert isinstance(final_import.payload, ImportPayload)
    assert final_import.payload.source_event_json == first_evidence.event_json
    assert final_import.payload.source_event_id == first_import.id
    assert final_import.payload.source_entry_id == entry.entry_id
    assert final_import.payload.source_record == first_import.record
    assert isinstance(first_import.payload, ImportPayload)
    assert first_import.payload.source_event_json == raw
    assert first_import.payload.source_event_id == event.id
    assert first_import.payload.source_entry_id == foreign.entry_id
    assert third.ledger.get(final_import.id).record == first_evidence.record
    assert ledger_path.read_text() == raw + "\n"
    for name in ("other", "entry"):
        (results / name).rename(results / f"moved-{name}")
        (database / name).rename(database / f"moved-{name}")
    reopened = ResultEntry.open("third", result_root=results, database_root=database)
    point = reopened.use_point("final")
    assert point.Q1.rate == 5300.0
    assert point.Q1.duration == 12.0
    for path in ("Q1.rate", "Q1.duration", "general.description"):
        metadata = point.meta(path)
        assert metadata is not None
        assert metadata.cloned_from == {"entry_id": entry.entry_id, "point": "first"}
        if metadata.source != "manual":
            found = reopened.ledger.get(metadata.source)
            assert found.event.id == final_import.id
            assert found.record == first_evidence.record
    assert cloned.meta("Q1.rate") == point.meta("Q1.rate")


@pytest.mark.parametrize("failure", ["absent_event", "record", "point", "header"])
def test_foreign_source_failure_does_not_publish_destination_or_modify_source(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    failure: str,
) -> None:
    results, _database = entry_roots
    foreign, _point, event = foreign_recorded_point
    source = results / "other/points/source/point.yaml"
    ledger_path = results / "other/records/ledger.jsonl"
    if failure == "absent_event":
        ledger_path.write_text("", encoding="utf-8")
    elif failure == "record":
        (results / f"other/records/{event.id}.json").unlink()
    elif failure == "point":
        document = YAML(typ="rt").load(source.read_text())
        document["components"]["Q1"]["rate"] = "invalid"
        with source.open("w", encoding="utf-8") as stream:
            YAML(typ="rt").dump(document, stream)
    else:
        raw = json.loads(ledger_path.read_text())
        raw["entry_id"] = str(uuid4())
        ledger_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    before = source.read_bytes()
    ledger_before = ledger_path.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        entry.new_point("failed", clone_from="other/source")
    assert not (results / "entry/points/failed").exists()
    assert entry.list_points() == []
    assert source.read_bytes() == before
    assert ledger_path.read_bytes() == ledger_before
    assert foreign.entry_id != entry.entry_id


@pytest.mark.parametrize("different_root", ["results", "database"])
def test_foreign_view_with_other_roots_is_rejected_without_publishing(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    tmp_path: Path,
    different_root: str,
) -> None:
    results, database = entry_roots
    foreign = ResultEntry.create(
        "elsewhere",
        result_root=tmp_path / "other-results"
        if different_root == "results"
        else results,
        database_root=tmp_path / "other-db"
        if different_root == "database"
        else database,
    )
    point = foreign.new_point("source")
    with pytest.raises(ValueError, match="root"):
        entry.new_point("failed", clone_from=point)
    assert not (results / "entry/points/failed").exists()
    assert entry.ledger.events() == ()


def test_existing_cross_entry_destination_is_not_merged(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    results, _database = entry_roots
    entry.new_point("target")
    source = results / "entry/points/target/point.yaml"
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        entry.new_point("target", clone_from="other/source")
    assert source.read_bytes() == before
    assert entry.ledger.events() == ()


def test_future_minor_import_preserves_payload_branch_collision_evidence(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
) -> None:
    foreign, view, _event = foreign_recorded_point
    results, _database = entry_roots
    event = LedgerEvent(
        id=str(uuid4()),
        kind="saved",
        at="2026-10-06T02:00:00Z",
        entry_id=foreign.entry_id,
        origin="notebook",
        payload=SavedPayload(run_id="r", outputs=()),
    )
    foreign.ledger.append(event, record={"sample": 1})
    with view.edit() as draft:
        draft.set(
            "Q1.rate",
            5300.0,
            provenance=Provenance(event.id, "fit", None, event.at, None),
        )
    path = results / "other/records/ledger.jsonl"
    lines = path.read_text().splitlines()
    data = json.loads(lines[-1])
    data["format_version"] = "1.1"
    data["payload"].update(
        run_id="future-run",
        tab="Spectroscopy",
        experiment="future-spec",
        cfg_summary={},
        roles={},
    )
    raw = json.dumps(data)
    path.write_text("\n".join([*lines[:-1], raw]) + "\n", encoding="utf-8")
    before = path.read_bytes()
    source_record = foreign.ledger.get(event.id).record
    cloned = entry.new_point("future-copy", clone_from=view)
    metadata = cloned.meta("Q1.rate")
    assert metadata is not None
    imported = entry.ledger.get(metadata.source)
    assert isinstance(imported.event.payload, ImportPayload)
    assert imported.event.payload.source_event_json == raw
    assert imported.record == source_record
    assert cloned.Q1.rate == 5300.0
    assert path.read_bytes() == before


@pytest.mark.parametrize("same_entry", [True, False])
def test_constructor_bound_view_clones_without_later_identity_registration(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    same_entry: bool,
) -> None:
    foreign, _view, event = foreign_recorded_point
    results, database = entry_roots
    source = results / "other/points/source/point.yaml"
    store = DocumentStore(
        source,
        PointDocument,
        format=PARAMETER_FORMAT,
        supported_version=PARAMETER_VERSION,
    )
    manual = PointView(
        store,
        source,
        ledger=results / "other/records/ledger.jsonl",
        entry_id=foreign.entry_id,
        result_root=results,
        database_root=database,
    )
    destination = foreign if same_entry else entry
    cloned = destination.new_point("manual-view-copy", clone_from=manual)
    assert cloned.Q1.rate == 5300.0
    assert len(foreign.ledger.events()) == 1
    metadata = cloned.meta("Q1.rate")
    assert metadata is not None
    assert metadata.cloned_from == {"entry_id": foreign.entry_id, "point": "source"}
    if same_entry:
        assert metadata.source == event.id
    else:
        imported = destination.ledger.get(metadata.source)
        assert isinstance(imported.event.payload, ImportPayload)
        assert imported.event.payload.source_entry_id == foreign.entry_id
        assert imported.event.payload.source_event_id == event.id
        assert (
            imported.event.payload.source_event_json
            == foreign.ledger.get(event.id).event_json
        )
