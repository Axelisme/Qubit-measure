"""Cross-run analysis records through the public ResultEntry seam."""

import json
from datetime import datetime, timezone
from uuid import UUID

import pytest
from zcu_tools.resources.entry import AnalyzedPayload, JsonObject, ResultEntry

from tests._filesystem import DIRECTORY_OPEN_ERROR


@pytest.mark.parametrize(
    "record",
    [None, {}, {"custom": {"residuals": [0.2, None, -0.1], "converged": True}}],
)
def test_record_analysis_appends_evidence_and_optional_json_attachment(
    entry: ResultEntry, record: JsonObject | None
) -> None:
    before = datetime.now(timezone.utc)
    event_id = entry.record_analysis(
        "t1_curve",
        run_ids=["run-second", "run-first"],
        summary={"T1": 42.0, "fit": {"unit": "us"}},
        record=record,
    )
    after = datetime.now(timezone.utc)

    assert UUID(event_id).version == 4
    stored = entry.ledger.get(event_id)
    assert stored.event.id == event_id
    assert stored.event.entry_id == entry.entry_id
    assert stored.event.kind == "analyzed"
    assert stored.event.origin == "notebook"
    assert before <= datetime.fromisoformat(stored.event.at) <= after
    assert stored.event.payload == AnalyzedPayload(
        analysis_kind="t1_curve",
        run_ids=("run-second", "run-first"),
        summary={"T1": 42.0, "fit": {"unit": "us"}},
    )
    assert entry.ledger.events() == (stored.event,)
    assert stored.record == record
    if record is None:
        assert stored.event.record is None
        assert list((entry.result_path / "records").glob("*.json")) == []
    else:
        assert stored.event.record == f"records/{event_id}.json"
        attachment = entry.result_path / "records" / f"{event_id}.json"
        assert json.loads(attachment.read_text(encoding="utf-8")) == record

    second_id = entry.record_analysis("custom/fit", run_ids=[], summary={})
    assert second_id != event_id
    assert [event.id for event in entry.ledger.events()] == [event_id, second_id]
    assert entry.ledger.get(event_id) == stored


def test_record_analysis_propagates_ledger_io_failure(entry: ResultEntry) -> None:
    ledger_path = entry.result_path / "records" / "ledger.jsonl"
    ledger_path.mkdir()

    with pytest.raises(DIRECTORY_OPEN_ERROR) as failure:
        entry.record_analysis("t1_curve", run_ids=["unresolved-run"], summary={})

    assert failure.value.filename == str(ledger_path)
    assert failure.value.__cause__ is None


def test_record_analysis_propagates_invalid_record_without_publishing(
    entry: ResultEntry,
) -> None:
    with pytest.raises(ValueError, match="invalid record"):
        entry.record_analysis(
            "caller-defined",
            run_ids=["unresolved-run"],
            summary={},
            record={"nonfinite": float("nan")},
        )

    assert entry.ledger.events() == ()
    assert list((entry.result_path / "records").glob("*.json")) == []
