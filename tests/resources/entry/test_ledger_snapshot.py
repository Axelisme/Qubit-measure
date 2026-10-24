"""Lock-free browsing snapshots retain the ledger owner's validation contract."""

from pathlib import Path
from uuid import uuid4

import pytest
from zcu_tools.resources.entry import AcquiredPayload, LedgerEvent, RecordsLedger


@pytest.fixture
def event() -> LedgerEvent:
    return LedgerEvent(
        id=str(uuid4()),
        entry_id=str(uuid4()),
        kind="acquired",
        at="2026-10-23T00:00:00Z",
        origin="notebook",
        payload=AcquiredPayload(
            run_id="20261023T000000Z-aaaaaa",
            tab="Synthetic",
            experiment="mux/readout",
            point="p1",
            roles={"qubit": "Q1"},
        ),
    )


def test_snapshot_reads_current_events_without_sidecar_or_cached_state(
    tmp_path: Path, event: LedgerEvent
) -> None:
    source = tmp_path / "ledger.jsonl"
    source.write_text(event.model_dump_json() + "\n", encoding="utf-8")
    ledger = RecordsLedger(tmp_path, entry_id=event.entry_id)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    snapshot = ledger.read_events_snapshot()
    assert snapshot == (event,)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    payload = snapshot[0].payload
    assert isinstance(payload, AcquiredPayload)
    payload.roles.clear()
    assert ledger.read_events_snapshot() == (event,)
    added = event.model_copy(update={"id": str(uuid4())})
    source.write_text(
        event.model_dump_json() + "\n" + added.model_dump_json() + "\n",
        encoding="utf-8",
    )
    assert ledger.read_events_snapshot() == (event, added)


@pytest.mark.parametrize("defect", ["tail", "duplicate", "foreign", "schema"])
def test_snapshot_rejects_invalid_history(
    tmp_path: Path, event: LedgerEvent, defect: str
) -> None:
    line = event.model_dump_json() + "\n"
    if defect == "tail":
        line = line.rstrip("\n")
    elif defect == "duplicate":
        line += line
    elif defect == "foreign":
        line = (
            event.model_copy(update={"entry_id": str(uuid4())}).model_dump_json() + "\n"
        )
    else:
        line = '{"format":"zcu.ledger","format_version":"1.0"}\n'
    (tmp_path / "ledger.jsonl").write_text(line, encoding="utf-8")
    ledger = RecordsLedger(tmp_path, entry_id=event.entry_id)
    with pytest.raises(ValueError, match=r"ledger\.jsonl:1|ledger\.jsonl:2"):
        ledger.read_events_snapshot()


def test_snapshot_missing_ledger_does_not_publish_files(
    tmp_path: Path, event: LedgerEvent
) -> None:
    assert RecordsLedger(tmp_path, entry_id=event.entry_id).read_events_snapshot() == ()
    assert list(tmp_path.iterdir()) == []
