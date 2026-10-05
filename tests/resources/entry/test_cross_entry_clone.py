"""Cross-entry clone imports remain resolvable from the destination alone."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from ruamel.yaml import YAML
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    ImportPayload,
    LedgerEvent,
    PointView,
    Provenance,
    ResultEntry,
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
        "format: zcu.module-library\nformat_version: '1.0'\nQ1: {pulse: arbitrary}\n",
        encoding="utf-8",
    )
    return foreign, point, event


@pytest.mark.parametrize("source_mode", ["label", "view"])
@pytest.mark.parametrize("origin,call_id", [("notebook", None), ("mcp", "clone-call")])
def test_cross_entry_clone_imports_once_and_uses_published_values(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    source_mode: str,
    origin: Literal["notebook", "mcp"],
    call_id: str | None,
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
    clone = entry.new_point(
        "target", clone_from=source_arg, origin=origin, call_id=call_id
    )
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
    assert imported.call_id == call_id
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


@pytest.mark.parametrize(
    "origin,call_id", [("mcp", None), ("mcp", ""), ("gui", "extra")]
)
def test_cross_entry_clone_rejects_invalid_origin_call_id_before_publication(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    foreign_recorded_point: tuple[ResultEntry, PointView, LedgerEvent],
    origin: Literal["mcp", "gui"],
    call_id: str | None,
) -> None:
    results, _database = entry_roots
    with pytest.raises(ValueError, match="call_id"):
        entry.new_point(
            "failed", clone_from="other/source", origin=origin, call_id=call_id
        )
    assert not (results / "entry/points/failed").exists()
    assert entry.ledger.events() == ()
