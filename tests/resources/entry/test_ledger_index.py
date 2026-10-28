"""Indexed ledger reads preserve invalidation, isolation and dependency contracts."""

import os
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import TypeAdapter
from zcu_tools.resources.entry import AnalyzedPayload, LedgerEvent, RecordsLedger


def _event() -> LedgerEvent:
    return LedgerEvent(
        id=str(uuid4()),
        kind="analyzed",
        at="2026-10-06T02:00:00Z",
        origin="notebook",
        payload=AnalyzedPayload(
            analysis_kind="fit", run_ids=(), summary={"rate": 5300.0}
        ),
    )


def _write(path: Path, events: tuple[LedgerEvent, ...]) -> None:
    path.write_text(
        "".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8"
    )


@pytest.fixture
def indexed_ledger(
    tmp_path: Path,
) -> tuple[RecordsLedger, Path, tuple[LedgerEvent, ...]]:
    history = tuple(_event() for _ in range(30))
    path = tmp_path / "ledger.jsonl"
    _write(path, history)
    ledger = RecordsLedger(tmp_path, lock_timeout=0.0)
    assert ledger.events() == history
    return ledger, path, history


@pytest.mark.parametrize(
    "change", ["replace", "truncate", "prefix", "same_size", "last_line"]
)
def test_changed_history_rebuilds_index(
    indexed_ledger: tuple[RecordsLedger, Path, tuple[LedgerEvent, ...]], change: str
) -> None:
    ledger, path, history = indexed_ledger
    changed = list(history)
    index = {"prefix": 0, "same_size": 15, "last_line": 29}.get(change, 0)
    changed[index] = history[index].model_copy(
        update={
            "payload": AnalyzedPayload(
                analysis_kind="fit", run_ids=(), summary={"rate": 5400.0}
            )
        }
    )
    expected = tuple(changed)
    before = path.stat()
    if change == "truncate":
        expected = history[:2]
        _write(path, expected)
    elif change == "replace":
        # Preserve prefix and previous last line, so only file identity changes.
        expected = (*history, _event())
        replacement = path.with_suffix(".replacement")
        _write(replacement, expected)
        replacement.replace(path)
    else:
        if change != "same_size":
            expected = (*expected, _event())
        _write(path, expected)
        if change == "same_size":
            assert path.stat().st_size == before.st_size
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    with patch.object(
        TypeAdapter,
        "validate_json",
        autospec=True,
        side_effect=TypeAdapter.validate_json,
    ) as parsed:
        assert ledger.events() == expected
    assert parsed.call_count == len(expected), (
        "changed history requires full validation"
    )
    if change == "truncate":
        with pytest.raises(KeyError):
            ledger.get(history[-1].id)
    else:
        expected_last = expected[-1] if change == "same_size" else expected[-2]
        assert ledger.get(history[-1].id).event == expected_last


def test_reload_validates_middle_rewrite_despite_unchanged_append_boundaries(
    indexed_ledger: tuple[RecordsLedger, Path, tuple[LedgerEvent, ...]],
) -> None:
    ledger, path, history = indexed_ledger
    data = path.read_bytes()
    middle = history[15].model_dump_json().encode()
    rewritten = middle.replace(b'"rate":5300.0', b'"rate":5400.0')
    added = _event()
    path.write_bytes(
        data.replace(middle, rewritten) + added.model_dump_json().encode() + b"\n"
    )
    with patch.object(
        TypeAdapter,
        "validate_json",
        autospec=True,
        side_effect=TypeAdapter.validate_json,
    ) as parsed:
        reloaded = ledger.reload()
    assert parsed.call_count == 31
    payload = reloaded[15].payload
    assert isinstance(payload, AnalyzedPayload)
    assert payload.summary == {"rate": 5400.0}
    assert reloaded[-1] == added
    assert ledger.get(history[15].id).event == reloaded[15]


def test_removed_ledger_discards_index_and_recreation_can_reuse_ids(
    indexed_ledger: tuple[RecordsLedger, Path, tuple[LedgerEvent, ...]],
) -> None:
    ledger, path, history = indexed_ledger
    path.unlink()
    assert ledger.events() == ()
    with pytest.raises(KeyError):
        ledger.get(history[0].id)
    ledger.append(history[0])
    assert ledger.events() == history[:1]


@pytest.mark.parametrize("lookup", ["events", "get", "get_many"])
def test_returned_payload_mutation_does_not_change_later_reads(
    indexed_ledger: tuple[RecordsLedger, Path, tuple[LedgerEvent, ...]], lookup: str
) -> None:
    ledger, _path, history = indexed_ledger
    event_id = history[0].id
    if lookup == "events":
        found = ledger.events()[0]
    elif lookup == "get":
        found = ledger.get(event_id).event
    else:
        found = ledger.get_many((event_id,))[0].event
    payload = found.payload
    assert isinstance(payload, AnalyzedPayload)
    payload.summary["rate"] = 9999.0
    assert ledger.get(event_id).event == history[0]


def test_valid_prefix_of_failed_tail_is_not_published_or_skipped(
    indexed_ledger: tuple[RecordsLedger, Path, tuple[LedgerEvent, ...]],
) -> None:
    ledger, path, history = indexed_ledger
    added = _event()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(added.model_dump_json() + "\n{broken}\n")
    for _ in range(2):
        with pytest.raises(ValueError, match=rf"{path}:32:"):
            ledger.get(added.id)
    _write(path, (*history, added))
    assert ledger.events() == (*history, added)
