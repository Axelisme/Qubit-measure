"""Public ledger persistence, lookup, schema, and lock contracts."""

from __future__ import annotations

import json
from io import BufferedWriter, TextIOWrapper
from pathlib import Path
from typing import IO, Literal
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest
from filelock import FileLock, Timeout
from pydantic import JsonValue, TypeAdapter
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
from zcu_tools.resources.file_lock import lock_path_for

from tests._filesystem import symlink_or_skip

_AT = "2026-10-06T02:00:00Z"


@pytest.fixture
def entry_records(tmp_path: Path) -> tuple[ResultEntry, Path]:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    return entry, tmp_path / "results/entry/records"


def _accepted(*, event_id: str | None = None) -> LedgerEvent:
    return LedgerEvent(
        id=event_id or str(uuid4()),
        kind="accepted",
        at=_AT,
        origin="notebook",
        payload=AcceptedPayload(
            writes=(AcceptedWrite(path="Q1.rate", value=5300.0),),
        ),
    )


def test_partial_history_reads_and_append_do_not_validate_unused_payload_fields(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted()
    data = event.model_dump(mode="json")
    data["payload"]["source"] = ["not a source reference"]
    path = records / "ledger.jsonl"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    before = path.read_bytes()
    ledger = entry.ledger
    added = _accepted()
    ledger.append(added)
    assert path.read_bytes().startswith(before)
    content = ledger.content(event.id)
    assert content.value("payload.writes.0.value") == 5300.0
    assert content.value("payload.run_id") is None
    with pytest.raises(ValueError, match="payload.source"):
        content.value("payload.source")
    with pytest.raises(ValueError, match="payload.source"):
        content.project()
    with pytest.raises(ValueError, match="source"):
        ledger.events()
    before_invalid = path.read_bytes()
    invalid = added.model_copy(update={"at": "yesterday"})
    with pytest.raises(ValueError, match="at"):
        ledger.append(invalid)
    assert path.read_bytes() == before_invalid


@pytest.mark.parametrize("defect", ["reference", "symlink"])
def test_partial_attachment_index_defers_reference_but_access_cannot_escape_root(
    entry_records: tuple[ResultEntry, Path],
    defect: str,
) -> None:
    entry, records = entry_records
    event = _accepted()
    ledger = entry.ledger
    ledger.append(event, record={"normal": True})
    external = records.parent / "external.json"
    external.write_text('{"private": true}', encoding="utf-8")
    path = records / "ledger.jsonl"
    if defect == "reference":
        data = json.loads(path.read_text())
        data["record"] = "../external.json"
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    else:
        attachment = records / f"{event.id}.json"
        attachment.unlink()
        symlink_or_skip(attachment, external)
    before = path.read_bytes()
    assert ledger.content(event.id).value("kind") == "accepted"
    assert len(ledger.contents()) == 1
    ledger.append(_accepted())
    assert path.read_bytes().startswith(before)
    with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
        with pytest.raises(ValueError, match="record.*records/|record escapes"):
            ledger.get(event.id)
        assert all(
            Path(call.args[0]).resolve() != external for call in opened.call_args_list
        )
    assert external.read_text(encoding="utf-8") == '{"private": true}'


def test_append_keeps_locks_out_of_records(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted()

    entry.ledger.append(event)

    assert entry.ledger.get(event.id).event == event
    assert not list(records.glob("*.lock"))


def test_large_ledger_parses_only_new_tail_after_append(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    _entry, records = entry_records
    history = tuple(_accepted() for _ in range(1000))
    path = records / "ledger.jsonl"
    path.write_text(
        "".join(event.model_dump_json() + "\n" for event in history), encoding="utf-8"
    )
    ledger = RecordsLedger(records)
    added = _accepted()

    # Observe the external JSON decoder and filesystem, not ledger internals.
    with patch.object(
        TypeAdapter,
        "validate_json",
        autospec=True,
        side_effect=TypeAdapter.validate_json,
    ) as parsed:
        assert ledger.events() == history
        assert parsed.call_count == 1000
        parsed.reset_mock()
        with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
            ledger.append(added)
            assert ledger.get(added.id).event == added
            assert tuple(
                item.event
                for item in ledger.get_many((added.id, history[0].id, added.id))
            ) == (added, history[0], added)
            assert ledger.events() == (*history, added)

        decoded_lines = [call.args[1] for call in parsed.call_args_list]
        assert len(decoded_lines) <= 1, "historical events must not be parsed again"
        assert all(line == added.model_dump_json() for line in decoded_lines)
        ledger_reads = [
            call
            for call in opened.call_args_list
            if call.args[0] == path and call.args[1] == "rb"
        ]
        assert len(ledger_reads) == 4, "each operation checks disk under its lock"


def test_accepted_complex_uses_tag_in_payload_and_persisted_ledger(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    value = complex(0.012345678901234567, -0.03456789012345678)
    tag = {"__complex__": [value.real, value.imag]}
    write = AcceptedWrite.model_validate(
        {"path": "Q1.ext.readout_cal.g_center", "value": value}
    )
    assert write.value == tag
    payload = AcceptedPayload.model_validate(
        {
            "writes": [
                {"path": "Q1.ext.readout_cal.g_center", "value": value},
                {"path": "Q1.ext.readout_cal.fidelity", "value": 0.93},
                {"path": "Q1.ext.readout_cal.confusion_matrix", "value": [[1, 0]]},
            ],
        }
    )
    assert [item.value for item in payload.writes] == [tag, 0.93, [[1, 0]]]
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at=_AT,
        origin="notebook",
        payload=payload,
    )
    entry.ledger.append(event)
    persisted = RecordsLedger(records).get(event.id).event
    assert isinstance(persisted.payload, AcceptedPayload)
    assert persisted.payload.writes == payload.writes
    assert persisted.model_dump(mode="json")["payload"]["writes"][0]["value"] == tag


@pytest.mark.parametrize(
    "value",
    [complex(float("inf"), 1), complex(1, float("nan")), [1 + 2j]],
)
def test_accepted_complex_rejects_nonfinite_and_nested_values(value: object) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        AcceptedWrite.model_validate({"path": "Q1.ext.center", "value": value})


def test_accepted_string_remains_json_string() -> None:
    write = AcceptedWrite.model_validate({"path": "Q1.ext.center", "value": "(1+2j)"})
    assert write.value == "(1+2j)"


@pytest.mark.parametrize(
    "value,expected",
    [
        pytest.param(np.float16(-0.125), -0.125, id="float16"),
        pytest.param(np.float32(0.93), 0.9300000071525574, id="float32"),
        pytest.param(np.float64(1.2345678901234567), 1.2345678901234567, id="float64"),
        pytest.param(np.longdouble("1.0000000000000000001"), 1.0, id="longdouble"),
        pytest.param(np.int8(-128), -128, id="int8"),
        pytest.param(np.int16(-32768), -32768, id="int16"),
        pytest.param(np.int32(-2147483648), -2147483648, id="int32"),
        pytest.param(np.int64(-9223372036854775808), -9223372036854775808, id="int64"),
        pytest.param(np.uint8(255), 255, id="uint8"),
        pytest.param(np.uint16(65535), 65535, id="uint16"),
        pytest.param(np.uint32(4294967295), 4294967295, id="uint32"),
        pytest.param(
            np.uint64(18446744073709551615), 18446744073709551615, id="uint64"
        ),
    ],
)
def test_accepted_numpy_scalar_normalizes_and_serializes(
    value: object, expected: int | float
) -> None:
    try:
        write = AcceptedWrite.model_validate({"path": "Q1.ext.value", "value": value})
    except ValueError as error:
        raise AssertionError(
            "AcceptedWrite must accept finite numpy scalars"
        ) from error
    assert type(write.value) is type(expected)
    assert write.value == expected
    serialized = write.model_dump_json()
    stored = json.loads(serialized)["value"]
    assert type(stored) is type(expected)
    assert stored == expected
    assert AcceptedWrite.model_validate_json(serialized) == write


@pytest.mark.parametrize(
    "scalar_type", [np.float16, np.float32, np.float64, np.longdouble]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_accepted_numpy_scalar_rejects_nonfinite(
    scalar_type: type[np.floating], value: str
) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        AcceptedWrite.model_validate(
            {"path": "Q1.ext.value", "value": scalar_type(value)}
        )


@pytest.mark.skipif(
    bool(np.finfo(np.longdouble).max <= np.finfo(np.float64).max),
    reason="longdouble has no extended range on this platform",
)
def test_accepted_numpy_scalar_rejects_finite_float_overflow() -> None:
    value = np.longdouble(np.finfo(np.longdouble).max)
    assert np.isfinite(value)
    with pytest.raises(ValueError, match="strict JSON"):
        AcceptedWrite.model_validate({"path": "Q1.ext.value", "value": value})


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.array(1.25), id="zero-dimensional-array"),
        pytest.param(np.array([1.25]), id="array"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param([np.float32(1.25)], id="nested-float"),
        pytest.param({"count": np.int64(2)}, id="nested-integer"),
        pytest.param((1, 2), id="tuple"),
        pytest.param(b"bytes", id="bytes"),
        pytest.param({1: "value"}, id="non-string-key"),
        pytest.param(object(), id="object"),
    ],
)
def test_accepted_numpy_scalar_does_not_relax_strict_json(value: object) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        AcceptedWrite.model_validate({"path": "Q1.ext.value", "value": value})


def test_empty_entry_and_independent_handles_observe_first_append(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    other = RecordsLedger(records)
    assert entry.ledger.events() == ()
    assert other.events() == ()
    assert not (records / "ledger.jsonl").exists()
    unknown = str(uuid4())
    with pytest.raises(KeyError, match=unknown):
        other.get(unknown)
    event = _accepted()
    entry.ledger.append(event)
    assert other.events() == (event,)
    assert other.get(event.id).event == event
    assert other.get(event.id).record is None
    payload = other.get(event.id).event.payload
    assert isinstance(payload, AcceptedPayload)
    assert payload.run_id is None
    assert payload.source is None


@pytest.mark.parametrize("legacy", [None, "not-a-uuid", {"malformed": [1]}])
def test_historical_import_projects_without_entry_identity(
    entry_records: tuple[ResultEntry, Path], legacy: JsonValue
) -> None:
    entry, records = entry_records
    source_id = str(uuid4())
    payload: JsonObject = {
        "source_event_id": source_id,
        "source_event_json": ' {"kept": true} ',
    }
    raw: JsonObject = {
        "format": "zcu.ledger",
        "format_version": "1.0",
        "id": str(uuid4()),
        "kind": "import",
        "at": _AT,
        "origin": "notebook",
        "payload": payload,
    }
    if legacy is not None:
        payload["source_entry_id"] = legacy
    path = records / "ledger.jsonl"
    original = json.dumps(raw)
    path.write_text(original + "\n", encoding="utf-8")
    before = path.read_bytes()

    (event,) = entry.ledger.events()
    stored = entry.ledger.get(event.id)
    assert stored.event == event
    assert stored.event_json == original
    assert isinstance(event.payload, ImportPayload)
    assert event.payload.source_event_id == source_id
    assert event.payload.source_event_json == ' {"kept": true} '
    assert path.read_bytes() == before


def test_all_event_kinds_round_trip_in_append_order(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, _records = entry_records
    acquired = LedgerEvent(
        id=str(uuid4()),
        kind="acquired",
        at=_AT,
        origin="gui",
        payload=AcquiredPayload(
            run_id="run-1",
            experiment="test/spectroscopy",
            roles={"qubit": "Q1"},
            point="a",
        ),
    )
    saved = LedgerEvent(
        id=str(uuid4()),
        kind="saved",
        at=_AT,
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
        origin="mcp",
        payload=AnalyzedPayload(
            analysis_kind="fit",
            run_ids=("run-1",),
            summary={"error": None},
        ),
    )
    accepted = _accepted()
    imported = LedgerEvent(
        id=str(uuid4()),
        kind="import",
        at=_AT,
        origin="notebook",
        payload=ImportPayload(
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
    original = _accepted()
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
    second = _accepted()
    entry.ledger.append(second)
    assert (records / "ledger.jsonl").read_bytes().startswith(first_bytes)
    assert tuple(event.id for event in entry.ledger.events()) == (
        original.id,
        second.id,
    )


def test_duplicate_rejection_preserves_published_bytes(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted()
    entry.ledger.append(event, record={"value": 1})
    before = (records / "ledger.jsonl").read_bytes()
    record_before = (records / f"{event.id}.json").read_bytes()
    with pytest.raises(ValueError, match="ledger"):
        entry.ledger.append(event)
    assert (records / "ledger.jsonl").read_bytes() == before
    assert (records / f"{event.id}.json").read_bytes() == record_before


def test_foreign_entry_history_can_be_read_and_appended(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    foreign = _accepted()
    path = records / "ledger.jsonl"
    historical = foreign.model_dump(mode="json")
    historical["entry_id"] = "foreign, malformed, ignored"
    original = (json.dumps(historical) + "\n").encode()
    path.write_bytes(original)
    ledger = entry.ledger

    assert ledger.events() == (foreign,)
    assert ledger.read_events_snapshot() == (foreign,)
    added = _accepted()
    ledger.append(added, record={"value": 2})

    assert entry.ledger.events() == (
        foreign,
        added.model_copy(update={"record": f"records/{added.id}.json"}),
    )
    assert ledger.get(added.id).record == {"value": 2}
    assert path.read_bytes().startswith(original)


def test_caller_record_reference_is_rejected_before_writing(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted()
    referenced = event.model_copy(update={"record": f"records/{event.id}.json"})
    with pytest.raises(ValueError, match="record"):
        entry.ledger.append(referenced, record={"value": 1})
    assert entry.ledger.events() == ()
    assert not (records / f"{event.id}.json").exists()


def test_existing_orphan_record_is_not_overwritten(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted()
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
    second = RecordsLedger(records / ".." / "records")
    events = tuple(_accepted() for _ in range(20))
    for index, event in enumerate(events):
        (first if index % 2 else second).append(event)
    assert first.events() == events
    assert second.events() == events
    assert all(second.get(event.id).event == event for event in events)


def test_lock_timeout_keeps_native_type_and_diagnostic_notes(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    _entry, records = entry_records
    ledger = RecordsLedger(records, lock_timeout=0.0)
    lock_path = lock_path_for(records / "ledger.jsonl")
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
    _entry, records = entry_records
    with pytest.raises(ValueError, match="lock_timeout"):
        RecordsLedger(records, lock_timeout=timeout)


def test_constructor_rejects_non_directory(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        RecordsLedger(file)


def test_missing_records_read_is_empty_and_first_append_creates_directory(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    opened, records = entry_records
    records.rmdir()

    ledger = opened.ledger
    assert ledger.events() == ()
    assert ledger.read_events_snapshot() == ()
    assert not records.exists()
    event = _accepted()
    ledger.append(event, record={"value": 3})

    assert records.is_dir()
    assert opened.ledger.events() == (
        event.model_copy(update={"record": f"records/{event.id}.json"}),
    )
    assert opened.ledger.get(event.id).record == {"value": 3}


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
        {"unexpected": True},
        {"kind": "saved"},
    ],
)
def test_event_model_rejects_invalid_header_origin_and_payload_pair(
    entry_records: tuple[ResultEntry, Path],
    change: dict[str, object],
) -> None:
    _entry, _records = entry_records
    data = _accepted().model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValueError, match="format|id|origin|kind|at|unexpected|payload"):
        LedgerEvent.model_validate(data)


@pytest.mark.parametrize("value", [b"bytes", (1, 2), float("nan"), float("inf")])
def test_payload_rejects_non_json_values(value: object) -> None:
    with pytest.raises(ValueError, match="value"):
        AcceptedPayload.model_validate(
            {
                "writes": [{"path": "Q1.rate", "value": value}],
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
        AcceptedPayload(writes=())
    source = SourceReference(kind="event", id=str(uuid4()))
    payload = AcceptedPayload(
        writes=(AcceptedWrite(path="Q1.rate", value=5300.0),),
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
    event = _accepted()
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    path.write_text(path.read_text() + broken, encoding="utf-8")
    before = path.read_bytes()
    for operation in (
        lambda: entry.ledger.events(),
        lambda: entry.ledger.get(event.id),
        lambda: entry.ledger.append(_accepted()),
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
    event = _accepted()
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
    first = _accepted()
    entry.ledger.append(first, record={"existing": True})
    before = (records / "ledger.jsonl").read_bytes()
    record_before = (records / f"{first.id}.json").read_bytes()
    event = _accepted()
    # model_validate is not involved: append owns validation of the record input.
    record: JsonObject = {"unsupported": float("nan")}
    with pytest.raises(ValueError, match="record"):
        entry.ledger.append(event, record=record)
    assert (records / "ledger.jsonl").read_bytes() == before
    assert (records / f"{first.id}.json").read_bytes() == record_before
    assert not (records / f"{event.id}.json").exists()


def test_duplicate_existing_event_blocks_read_and_append_without_rewrite(
    entry_records: tuple[ResultEntry, Path],
) -> None:
    entry, records = entry_records
    event = _accepted()
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    data = json.loads(path.read_text())
    path.write_text(path.read_text() + json.dumps(data) + "\n", encoding="utf-8")
    before = path.read_bytes()
    for operation in (
        lambda: entry.ledger.events(),
        lambda: entry.ledger.get(event.id),
        lambda: entry.ledger.append(_accepted()),
    ):
        with pytest.raises(ValueError, match="ledger") as caught:
            operation()
        assert str(path) in str(caught.value)
        assert "2" in str(caught.value)
        assert path.read_bytes() == before


@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_reader_preserves_raw_event_and_unknown_fields_without_rewrite(
    entry_records: tuple[ResultEntry, Path], version: str
) -> None:
    entry, records = entry_records
    event = _accepted()
    entry.ledger.append(event, record={"fit": [1, 2]})
    path = records / "ledger.jsonl"
    data = json.loads(path.read_text())
    data["format_version"] = version
    data["future_header"] = {"new": True}
    data["payload"]["future_payload"] = [1, None]
    raw = "  " + json.dumps(data, ensure_ascii=False, indent=None) + "  "
    path.write_text(raw + "\n", encoding="utf-8")
    before = path.read_bytes()
    record_path = records / f"{event.id}.json"
    record_before = record_path.read_bytes()
    found = entry.ledger.get(event.id)
    assert found.event.id == event.id
    assert found.event.format_version == version
    assert found.event.payload == event.payload
    assert found.record == {"fit": [1, 2]}
    assert found.event_json == raw
    assert entry.ledger.events() == (found.event,)
    assert path.read_bytes() == before
    entry.ledger.append(_accepted())
    assert path.read_bytes().startswith(before)
    assert entry.ledger.get(event.id).event_json == raw
    assert record_path.read_bytes() == record_before


@pytest.mark.parametrize("version", ["1.0", "1.2"])
def test_legacy_acquired_projects_evidence_and_keeps_raw_cfg(
    entry_records: tuple[ResultEntry, Path], version: str
) -> None:
    entry, records = entry_records
    event_id = str(uuid4())
    data = {
        "format": "zcu.ledger",
        "format_version": version,
        "id": event_id,
        "kind": "acquired",
        "at": _AT,
        "entry_id": "not-a-uuid",
        "origin": "notebook",
        "payload": {
            "run_id": "historical-run",
            "tab": "old-experiment",
            "experiment": "test/spectroscopy",
            "point": "p",
            "roles": {"qubit": "Q1"},
            "cfg_summary": ["retired", {"schema": "opaque"}],
        },
    }
    raw = json.dumps(data)
    path = records / "ledger.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")
    before = path.read_bytes()
    found = entry.ledger.get(event_id)
    assert isinstance(found.event.payload, AcquiredPayload)
    assert found.event.payload.run_id == "historical-run"
    assert found.event.payload.roles == {"qubit": "Q1"}
    assert found.event.payload.point == "p"
    assert entry.ledger.events() == (found.event,)
    assert found.event_json == raw
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_payload_kind_controls_unknown_field_projection(
    entry_records: tuple[ResultEntry, Path], version: str
) -> None:
    entry, records = entry_records
    event = LedgerEvent(
        id=str(uuid4()),
        kind="saved",
        at=_AT,
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
    event = _accepted()
    entry.ledger.append(event)
    path = records / "ledger.jsonl"
    corrupt = path.read_bytes() + b"\xff\n"
    path.write_bytes(corrupt)
    for operation in (
        lambda: entry.ledger.events(),
        lambda: entry.ledger.get(event.id),
        lambda: entry.ledger.append(_accepted()),
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
    event = _accepted()
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
    entry.ledger.append(_accepted())
    path = records / "ledger.jsonl"
    before = path.read_bytes()
    event = _accepted()
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
    entry.ledger.append(_accepted())
    path = records / "ledger.jsonl"
    before = path.read_bytes()
    event = _accepted()
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
            lambda: entry.ledger.append(_accepted()),
        ):
            with pytest.raises(ValueError, match="incomplete tail"):
                operation()
            assert path.read_bytes() == after
    else:
        found = entry.ledger.get(event.id)
        assert found.record == record
        assert found.event.record == f"records/{event.id}.json"
    assert attachment.exists()
