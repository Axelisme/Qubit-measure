"""Public ledger persistence, lookup, schema, and lock contracts."""

from __future__ import annotations

import json
from io import BufferedWriter, TextIOWrapper
from pathlib import Path
from typing import IO, Literal
from uuid import uuid4

import pytest
from filelock import FileLock, Timeout
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    AcquiredPayload,
    AnalyzedPayload,
    ImportPayload,
    JsonObject,
    LedgerEvent,
    RecordsLedger,
    ResultEntry,
    SavedOutput,
    SavedPayload,
    SourceReference,
)

_AT = "2026-10-06T02:00:00Z"


@pytest.fixture
def entry_records(tmp_path: Path) -> tuple[ResultEntry, Path]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    return entry, tmp_path / "results/entry/records"


def _accepted(entry_id: str, *, event_id: str | None = None) -> LedgerEvent:
    return LedgerEvent(
        id=event_id or str(uuid4()),
        kind="accepted",
        at=_AT,
        entry_id=entry_id,
        origin="notebook",
        payload=AcceptedPayload(
            writes=(AcceptedWrite(path="Q1.rate", value=5300.0),),
            analysis_summary={"frequency": 5300.0},
        ),
    )


def test_empty_entry_and_independent_handles_observe_first_append(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    other = RecordsLedger(records, entry_id=entry.entry_id)
    assert entry.ledger.events() == ()
    assert other.events() == ()
    assert not (records / "ledger.jsonl").exists()
    unknown = str(uuid4())
    with pytest.raises(KeyError, match=unknown):
        other.get(unknown)
    event = _accepted(entry.entry_id)
    entry.ledger.append(event)
    assert other.events() == (event,)
    assert other.get(event.id).event == event
    assert other.get(event.id).record is None
    payload = other.get(event.id).event.payload
    assert isinstance(payload, AcceptedPayload)
    assert payload.run_id is None
    assert payload.source is None


def test_all_event_kinds_round_trip_in_append_order(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, _records = entry_records
    acquired = LedgerEvent(
        id=str(uuid4()),
        kind="acquired",
        at=_AT,
        entry_id=entry.entry_id,
        origin="gui",
        payload=AcquiredPayload(
            run_id="run-1",
            tab="Spectroscopy",
            experiment="test/spectroscopy",
            cfg_summary={"shots": 10},
            roles={"qubit": "Q1"},
            point="a",
        ),
    )
    saved = LedgerEvent(
        id=str(uuid4()),
        kind="saved",
        at=_AT,
        entry_id=entry.entry_id,
        origin="notebook",
        payload=SavedPayload(
            run_id="run-1",
            outputs=(
                SavedOutput(
                    artifact="result",
                    member="data",
                    format="data_h5",
                    path="/db/run-1/data.h5",
                    status="saved",
                ),
            ),
        ),
    )
    analyzed = LedgerEvent(
        id=str(uuid4()),
        kind="analyzed",
        at=_AT,
        entry_id=entry.entry_id,
        origin="mcp",
        call_id="call-1",
        payload=AnalyzedPayload(
            analysis_kind="fit",
            run_ids=("run-1",),
            summary={"error": None},
        ),
    )
    accepted = _accepted(entry.entry_id)
    imported = LedgerEvent(
        id=str(uuid4()),
        kind="import",
        at=_AT,
        entry_id=entry.entry_id,
        origin="notebook",
        payload=ImportPayload(
            source_entry_id=str(uuid4()),
            source_event_id=str(uuid4()),
            source_event_json=' {"historical": true} ',
        ),
    )
    events = (acquired, saved, analyzed, accepted, imported)
    for event in events:
        entry.ledger.append(event)
    assert entry.ledger.events() == events
    assert tuple(entry.ledger.get(event.id).event for event in events) == events


def test_append_preserves_history_and_publishes_local_record_without_mutating_event(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    original = _accepted(entry.entry_id)
    content: JsonObject = {"fit": {"value": 5.3, "samples": [1, True, None, "µ"]}}
    entry.ledger.append(original, record=content)
    first_bytes = (records / "ledger.jsonl").read_bytes()
    assert first_bytes.endswith(b"\n")
    assert original.record is None
    found = entry.ledger.get(original.id)
    assert found.event.record == f"records/{original.id}.json"
    assert found.record == content
    assert json.loads(found.event_json)["id"] == original.id
    assert json.loads((records / f"{original.id}.json").read_text()) == content
    second = _accepted(entry.entry_id)
    entry.ledger.append(second)
    assert (records / "ledger.jsonl").read_bytes().startswith(first_bytes)
    assert tuple(event.id for event in entry.ledger.events()) == (
        original.id,
        second.id,
    )


def test_duplicate_and_foreign_entry_rejections_preserve_published_bytes(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event, record={"value": 1})
    before = (records / "ledger.jsonl").read_bytes()
    record_before = (records / f"{event.id}.json").read_bytes()
    with pytest.raises(ValueError, match="ledger"):
        entry.ledger.append(event)
    foreign = _accepted(str(uuid4()))
    with pytest.raises(ValueError, match="ledger"):
        entry.ledger.append(foreign, record={"value": 2})
    assert (records / "ledger.jsonl").read_bytes() == before
    assert (records / f"{event.id}.json").read_bytes() == record_before
    assert not (records / f"{foreign.id}.json").exists()


def test_caller_record_reference_is_rejected_before_writing(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    referenced = event.model_copy(update={"record": f"records/{event.id}.json"})
    with pytest.raises(ValueError, match="record"):
        entry.ledger.append(referenced, record={"value": 1})
    assert entry.ledger.events() == ()
    assert not (records / f"{event.id}.json").exists()


def test_existing_orphan_record_is_not_overwritten(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    orphan = records / f"{event.id}.json"
    orphan.write_text('{"orphan": true}\n', encoding="utf-8")
    before = orphan.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        entry.ledger.append(event, record={"replacement": True})
    assert orphan.read_bytes() == before
    assert entry.ledger.events() == ()


def test_two_handles_append_without_overwrite(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    first = entry.ledger
    second = RecordsLedger(records / ".." / "records", entry_id=entry.entry_id)
    events = tuple(_accepted(entry.entry_id) for _ in range(20))
    for index, event in enumerate(events):
        (first if index % 2 else second).append(event)
    assert first.events() == events
    assert second.events() == events
    assert all(second.get(event.id).event == event for event in events)


def test_lock_timeout_keeps_native_type_and_diagnostic_notes(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    ledger = RecordsLedger(records, entry_id=entry.entry_id, lock_timeout=0.0)
    lock_path = records / "ledger.jsonl.lock"
    with FileLock(str(lock_path)), pytest.raises(Timeout) as caught:
        ledger.events()
    assert Path(caught.value.lock_file) == lock_path
    notes = " ".join(caught.value.__notes__)
    assert str(records / "ledger.jsonl") in notes
    assert "0.0" in notes
    assert ledger.events() == ()


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf")])
def test_constructor_rejects_invalid_lock_timeout(
    entry_records: tuple[ResultEntry, Path],
    timeout: float,
) -> None:
    entry, records = entry_records
    with pytest.raises(ValueError, match="lock_timeout"):
        RecordsLedger(records, entry_id=entry.entry_id, lock_timeout=timeout)


def test_constructor_requires_records_directory_and_uuid(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        RecordsLedger(missing, entry_id=str(uuid4()))
    file = tmp_path / "file"
    file.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        RecordsLedger(file, entry_id=str(uuid4()))
    with pytest.raises(ValueError, match="entry_id"):
        RecordsLedger(tmp_path, entry_id="not-a-uuid")
    assert not missing.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"format": "not.ledger"},
        {"format_version": "2.0"},
        {"id": "evt-1"},
        {"origin": "remote"},
        {"kind": "unknown"},
        {"at": "2026-10-06T02:00:00"},
        {"at": "2026-10-06T02:00:00+01:00"},
        {"origin": "mcp", "call_id": None},
        {"origin": "gui", "call_id": "extra"},
        {"unexpected": True},
        {"kind": "saved"},
    ],
)
def test_event_model_rejects_invalid_header_origin_and_payload_pair(
    entry_records: tuple[ResultEntry, Path],
    change: dict[str, object],
) -> None:
    entry, _records = entry_records
    data = _accepted(entry.entry_id).model_dump(mode="json")
    data.update(change)
    with pytest.raises(
        ValueError, match="format|id|origin|kind|at|call_id|unexpected|payload"
    ):
        LedgerEvent.model_validate(data)


@pytest.mark.parametrize("value", [b"bytes", (1, 2), float("nan"), float("inf")])
def test_payload_rejects_non_json_values(value: object) -> None:
    with pytest.raises(ValueError, match="value"):
        AcceptedPayload.model_validate(
            {
                "writes": [{"path": "Q1.rate", "value": value}],
                "analysis_summary": {},
            }
        )


@pytest.mark.parametrize(
    "status,error", [("saved", "failed"), ("failed", None), ("failed", "")]
)
def test_saved_output_status_requires_consistent_error(
    status: str, error: str | None
) -> None:
    with pytest.raises(ValueError, match="error|status"):
        SavedOutput.model_validate(
            {
                "artifact": "result",
                "member": "data",
                "format": "data_h5",
                "path": "/db/data.h5",
                "status": status,
                "error": error,
            }
        )


def test_accepted_payload_requires_writes_and_allows_explicit_source() -> None:
    with pytest.raises(ValueError, match="writes"):
        AcceptedPayload(writes=(), analysis_summary={})
    source = SourceReference(kind="event", id=str(uuid4()))
    payload = AcceptedPayload(
        writes=(AcceptedWrite(path="Q1.rate", value=5300.0),),
        analysis_summary={},
        run_id="run-1",
        source=source,
    )
    assert payload.source == source
    assert payload.run_id == "run-1"


@pytest.mark.parametrize(
    "broken", ["{broken json}\n", "{}\n", '{"format": "zcu.ledger"}']
)
def test_corrupt_external_input_is_not_skipped_or_rewritten(
    entry_records: tuple[ResultEntry, Path],
    broken: str,
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    path.write_text(path.read_text() + broken, encoding="utf-8")
    before = path.read_bytes()
    for operation in (
        lambda: entry.ledger.events(),
        lambda: entry.ledger.get(event.id),
        lambda: entry.ledger.append(_accepted(entry.entry_id)),
    ):
        with pytest.raises(ValueError, match="ledger") as caught:
            operation()
        assert str(path) in str(caught.value)
        assert "2" in str(caught.value)
        assert path.read_bytes() == before


def test_missing_referenced_record_reports_its_physical_path(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event, record={"fit": 1})
    physical = records / f"{event.id}.json"
    physical.unlink()
    assert entry.ledger.events()[0].id == event.id
    with pytest.raises(FileNotFoundError) as caught:
        entry.ledger.get(event.id)
    assert str(physical) in str(caught.value)


def test_invalid_record_input_leaves_ledger_and_sidecars_unchanged(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    first = _accepted(entry.entry_id)
    entry.ledger.append(first, record={"existing": True})
    before = (records / "ledger.jsonl").read_bytes()
    record_before = (records / f"{first.id}.json").read_bytes()
    event = _accepted(entry.entry_id)
    # model_validate is not involved: append owns validation of the record input.
    record: JsonObject = {"unsupported": float("nan")}
    with pytest.raises(ValueError, match="record"):
        entry.ledger.append(event, record=record)
    assert (records / "ledger.jsonl").read_bytes() == before
    assert (records / f"{first.id}.json").read_bytes() == record_before
    assert not (records / f"{event.id}.json").exists()


@pytest.mark.parametrize("failure", ["duplicate", "foreign_entry", "bad_payload"])
def test_invalid_existing_event_blocks_read_and_append_without_rewrite(
    entry_records: tuple[ResultEntry, Path],
    failure: str,
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    data = json.loads(path.read_text())
    if failure == "foreign_entry":
        data["id"] = str(uuid4())
        data["entry_id"] = str(uuid4())
    elif failure == "bad_payload":
        data["id"] = str(uuid4())
        data["payload"]["writes"] = []
    path.write_text(path.read_text() + json.dumps(data) + "\n", encoding="utf-8")
    before = path.read_bytes()
    for operation in (
        lambda: entry.ledger.events(),
        lambda: entry.ledger.get(event.id),
        lambda: entry.ledger.append(_accepted(entry.entry_id)),
    ):
        with pytest.raises(ValueError, match="ledger") as caught:
            operation()
        assert str(path) in str(caught.value)
        assert "2" in str(caught.value)
        assert path.read_bytes() == before


def test_future_minor_preserves_raw_event_and_unknown_payload_without_rewrite(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    data = json.loads(path.read_text())
    data["format_version"] = "1.2"
    data["future_header"] = {"new": True}
    data["payload"]["future_payload"] = [1, None]
    raw = "  " + json.dumps(data, ensure_ascii=False, indent=None) + "  "
    path.write_text(raw + "\n", encoding="utf-8")
    before = path.read_bytes()
    found = entry.ledger.get(event.id)
    assert found.event.id == event.id
    assert found.event.format_version == "1.2"
    assert found.event.payload == event.payload
    assert found.event_json == raw
    assert entry.ledger.events() == (found.event,)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="version"):
        entry.ledger.append(found.event)
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_payload_kind_controls_future_field_projection(
    entry_records: tuple[ResultEntry, Path], version: str
) -> None:
    entry, records = entry_records
    event = LedgerEvent(
        id=str(uuid4()),
        kind="saved",
        at=_AT,
        entry_id=entry.entry_id,
        origin="notebook",
        payload=SavedPayload(run_id="r", outputs=()),
    )
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    data = json.loads(path.read_text())
    data["format_version"] = version
    data["payload"].update(
        tab="Spectroscopy", experiment="future-spec", cfg_summary={}, roles={}
    )
    raw = json.dumps(data)
    path.write_text(raw + "\n", encoding="utf-8")
    before = path.read_bytes()
    if version == "1.0":
        with pytest.raises(ValueError, match="extra"):
            entry.ledger.events()
    else:
        found = entry.ledger.get(event.id)
        assert isinstance(found.event.payload, SavedPayload)
        assert found.event.payload == event.payload
        assert found.event_json == raw
        assert entry.ledger.events() == (found.event,)
    assert path.read_bytes() == before

    data["payload"]["run_id"] = 123
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    invalid = path.read_bytes()
    with pytest.raises(ValueError, match="run_id"):
        entry.ledger.events()
    assert path.read_bytes() == invalid


def test_invalid_utf8_ledger_reports_exact_line_without_rewrite(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    corrupt = path.read_bytes() + b"\xff\n"
    path.write_bytes(corrupt)
    for operation in (
        lambda: entry.ledger.events(),
        lambda: entry.ledger.get(event.id),
        lambda: entry.ledger.append(_accepted(entry.entry_id)),
    ):
        with pytest.raises(ValueError, match="utf-8") as caught:
            operation()
        assert f"{path}:2:" in str(caught.value)
        assert isinstance(caught.value.__cause__, UnicodeDecodeError)
        assert path.read_bytes() == corrupt


def test_invalid_utf8_record_reports_physical_path_without_rewrite(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted(entry.entry_id)
    entry.ledger.append(event, record={"sample": 1})
    path = records / f"{event.id}.json"
    path.write_bytes(b"\xff")
    ledger_bytes = (records / "ledger.jsonl").read_bytes()
    assert len(entry.ledger.events()) == 1
    with pytest.raises(ValueError, match="utf-8") as caught:
        entry.ledger.get(event.id)
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)
    assert path.read_bytes() == b"\xff"
    assert (records / "ledger.jsonl").read_bytes() == ledger_bytes


def test_record_rename_failure_cleans_only_unpublished_temp(
    entry_records: tuple[ResultEntry, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, records = entry_records
    entry.ledger.append(_accepted(entry.entry_id))
    path = records / "ledger.jsonl"
    before = path.read_bytes()
    event = _accepted(entry.entry_id)
    original_replace = Path.replace

    def fail_record_rename(source: Path, target: Path) -> Path:
        if source.parent == records and source.suffix == ".tmp":
            raise OSError("injected record rename")
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_record_rename)
        with pytest.raises(OSError, match="injected record rename"):
            entry.ledger.append(event, record={"sample": 1})
    assert not tuple(records.glob("*.tmp"))
    assert not (records / f"{event.id}.json").exists()
    assert path.read_bytes() == before
    with pytest.raises(KeyError):
        entry.ledger.get(event.id)


class _FailingLedgerStream(TextIOWrapper):
    """Write/flush collaborator that persists evidence before reporting failure."""

    def __init__(self, buffer: BufferedWriter, fail_at: Literal["write", "flush"]):
        self._fail_at: Literal["write", "flush"] | None = fail_at
        super().__init__(buffer, encoding="utf-8", newline="")

    def write(self, text: str) -> int:
        if self._fail_at == "write":
            super().write(text[:10])
            raise OSError("injected ledger write")
        return super().write(text)

    def flush(self) -> None:
        super().flush()
        if self._fail_at == "flush":
            raise OSError("injected ledger flush")

    def close(self) -> None:
        # Context cleanup must not hide the injected write/flush failure.
        self._fail_at = None
        super().close()


@pytest.mark.parametrize("fail_at", ["open", "write", "flush"])
def test_append_io_failure_retains_published_record_and_historical_bytes(
    entry_records: tuple[ResultEntry, Path],
    monkeypatch: pytest.MonkeyPatch,
    fail_at: Literal["open", "write", "flush"],
) -> None:
    entry, records = entry_records
    entry.ledger.append(_accepted(entry.entry_id))
    path = records / "ledger.jsonl"
    before = path.read_bytes()
    event = _accepted(entry.entry_id)
    record: JsonObject = {"sample": [1, "µ"]}
    original_open = Path.open

    def failing_open(
        source: Path,
        mode: str = "r",
        *,
        encoding: str | None = None,
        newline: str | None = None,
    ) -> IO[str] | IO[bytes]:
        if source == path and mode == "a":
            if fail_at == "open":
                raise OSError("injected ledger open")
            return _FailingLedgerStream(original_open(source, "ab"), fail_at)
        return original_open(source, mode, encoding=encoding, newline=newline)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match=f"injected ledger {fail_at}"):
            entry.ledger.append(event, record=record)

    attachment = records / f"{event.id}.json"
    assert json.loads(attachment.read_text()) == record
    assert not tuple(records.glob("*.tmp"))
    after = path.read_bytes()
    assert after.startswith(before)
    if fail_at == "open":
        assert after == before
        with pytest.raises(KeyError):
            entry.ledger.get(event.id)
    elif fail_at == "write":
        assert len(after) > len(before)
        for operation in (
            lambda: entry.ledger.events(),
            lambda: entry.ledger.get(event.id),
            lambda: entry.ledger.append(_accepted(entry.entry_id)),
        ):
            with pytest.raises(ValueError, match="incomplete tail"):
                operation()
            assert path.read_bytes() == after
    else:
        found = entry.ledger.get(event.id)
        assert found.record == record
        assert found.event.record == f"records/{event.id}.json"
    assert attachment.exists()
