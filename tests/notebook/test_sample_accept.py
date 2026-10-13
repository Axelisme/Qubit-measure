"""Sample acceptance through real Notebook runs and entry storage owners."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.notebook import NotebookAdapter, SavedRun
from zcu_tools.plotting.plots import NonPresentingHost
from zcu_tools.resources.entry import (
    AcceptedPayload,
    JsonObject,
    PointView,
    ResultEntry,
    writeback,
)

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


@pytest.fixture
def entry(tmp_path: Path) -> ResultEntry:
    return ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )


@pytest.fixture
def point(entry: ResultEntry) -> PointView:
    return entry.new_point("sweet", description="original")


@pytest.fixture
def saved(point: PointView) -> SavedRun[SavingCfg, SavingResult]:
    exp = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(SavingCore())
    return exp.save_run(point, source=exp.run(SavingCfg()))


@pytest.mark.parametrize(
    "fields",
    [
        {"t1_us": 20.09, "dev_value": 12.5, "dev_unit": "mA", "meter": {"range": 3}},
        {"t1_us": 20.09, "heater_v": 0.35},
    ],
    ids=["caller-device-fields", "without-device-fields"],
)
def test_accept_samples_preserves_fields_and_shares_parameter_event(
    entry: ResultEntry,
    point: PointView,
    saved: SavedRun[SavingCfg, SavingResult],
    fields: JsonObject,
) -> None:
    before = len(entry.ledger.events())
    event_id = saved.accept(
        parameter={"general.description": "accepted"}, samples={"Q1": fields}
    )
    assert len(entry.ledger.events()) == before + 1
    event = entry.ledger.get(event_id).event
    assert event.kind == "accepted"
    assert event.origin == "notebook"
    assert isinstance(event.payload, AcceptedPayload)
    assert event.payload.run_id == saved.run_id
    assert event.payload.writes[0].value == "accepted"
    assert event.payload.writes[1].value == fields
    frame = entry.samples("Q1").read()
    assert len(frame) == 1
    row = frame.iloc[0].to_dict()
    assert row == fields | {
        "run_id": saved.run_id,
        "point": "sweet",
        "at": row["at"],
        "source": event_id,
    }
    assert datetime.fromisoformat(row["at"]) == datetime.fromisoformat(event.at)
    assert datetime.fromisoformat(row["at"]).utcoffset() == timedelta(0)
    point.refresh()
    assert point.description == "accepted"


def test_reaccept_appends_history_and_reads_last_whole_row(
    entry: ResultEntry, saved: SavedRun[SavingCfg, SavingResult]
) -> None:
    first_id = saved.accept(samples={"Q1": {"t1_us": 20.09, "dev_value": 1.2}})
    path = entry.result_path / "samples" / "Q1.jsonl"
    first_bytes = path.read_bytes()
    second_id = saved.accept(samples={"Q1": {"t1_us": 31.7}})
    assert second_id != first_id
    assert path.read_bytes().startswith(first_bytes)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["source"] for row in rows] == [first_id, second_id]
    assert [row["run_id"] for row in rows] == [saved.run_id, saved.run_id]
    latest = entry.samples("Q1").read()
    assert len(latest) == 1
    assert latest.iloc[0].to_dict() == rows[-1]
    assert latest.iloc[0]["t1_us"] == 31.7


def test_sample_failure_keeps_prior_parameter_and_sample_and_stops(
    entry: ResultEntry, point: PointView, saved: SavedRun[SavingCfg, SavingResult]
) -> None:
    blocked = entry.result_path / "samples" / "blocked.jsonl"
    blocked.mkdir(parents=True)
    before = len(entry.ledger.events())
    with pytest.raises(writeback.WritebackApplyError) as caught:
        saved.accept(
            parameter={"general.description": "retained"},
            samples={
                "Q1": {"t1_us": 20.09},
                "blocked": {"t1_us": 21.0},
                "Q2": {"t1_us": 22.0},
            },
        )
    assert len(entry.ledger.events()) == before + 1
    event = entry.ledger.events()[-1]
    assert event.kind == "accepted"
    assert caught.value.event_id == event.id
    assert caught.value.applied == 2
    assert isinstance(caught.value.__cause__, IsADirectoryError)
    point.refresh()
    assert point.description == "retained"
    assert entry.samples("Q1").read().iloc[0]["source"] == event.id
    assert entry.samples("Q2").read().empty


def test_parameter_failure_does_not_reach_samples(
    entry: ResultEntry, point: PointView, saved: SavedRun[SavingCfg, SavingResult]
) -> None:
    before = len(entry.ledger.events())
    with pytest.raises(writeback.WritebackApplyError) as caught:
        saved.accept(
            parameter={"general.description": 42}, samples={"Q1": {"t1_us": 20.09}}
        )
    assert len(entry.ledger.events()) == before + 1
    assert entry.ledger.events()[-1].kind == "accepted"
    assert caught.value.event_id == entry.ledger.events()[-1].id
    assert caught.value.applied == 0
    assert isinstance(caught.value.__cause__, ValueError)
    assert "description" in str(caught.value.__cause__)
    point.refresh()
    assert point.description == "original"
    assert entry.samples("Q1").read().empty
