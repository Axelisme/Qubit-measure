"""Shared accepted-writeback executor over a bound point and opaque parameter metadata."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.resources.entry import (
    AcceptedPayload,
    PointView,
    ResultEntry,
    capture_point_storage,
)
from zcu_tools.resources.entry.writeback import (
    ParameterWriteback,
    PointWritebackTarget,
    WritebackApplyError,
    WritebackEvidence,
    apply_writeback,
)


@pytest.fixture
def point(tmp_path: Path) -> PointView:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "db"
    )
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=5000.0, duration=10.0)
    point = entry.new_point("a")
    with point.edit() as draft:
        draft.set_meta("Q1.duration", {"stderr": 0.5, "comment": "kept"})
        draft.set_meta("Q1.rate", {"stderr": 9.0, "source": "old run"})
    return point


def target(point: PointView) -> PointWritebackTarget:
    return PointWritebackTarget(point, capture_point_storage(point))


def evidence() -> WritebackEvidence:
    return WritebackEvidence(kind="test/fit", origin="notebook", run_id="run-1")


def test_parameter_items_commit_caller_metadata_and_keep_omitted_metadata(
    point: PointView,
) -> None:
    writer = target(point)
    event_id = apply_writeback(
        (
            ParameterWriteback("Q1.rate", 5300.0, metadata={"stderr": 0.2}),
            ParameterWriteback("Q1.duration", 12.0),
            ParameterWriteback("general.description", "fit", metadata={}),
        ),
        target=writer,
        evidence=evidence(),
    )
    event = writer.ledger.get(event_id).event
    assert event.kind == "accepted"
    assert isinstance(event.payload, AcceptedPayload)
    assert [(write.path, write.value) for write in event.payload.writes] == [
        ("Q1.rate", 5300.0),
        ("Q1.duration", 12.0),
        ("general.description", "fit"),
    ]
    point.refresh()
    assert (point.Q1.rate, point.Q1.duration) == (5300.0, 12.0)
    assert point.meta("Q1.rate") == {"stderr": 0.2}
    assert point.meta("Q1.duration") == {"stderr": 0.5, "comment": "kept"}
    assert point.meta("general.description") == {}


def test_item_failure_keeps_prefix_value_and_metadata_and_stops(
    point: PointView,
) -> None:
    writer = target(point)
    with pytest.raises(WritebackApplyError) as caught:
        apply_writeback(
            (
                ParameterWriteback("Q1.rate", 5300.0, metadata={"stderr": 0.2}),
                ParameterWriteback("Q1.duration", "invalid", metadata={"stderr": 1}),
                ParameterWriteback("general.description", "later"),
            ),
            target=writer,
            evidence=evidence(),
        )
    assert caught.value.applied == 1
    assert isinstance(caught.value.__cause__, ValidationError)
    assert writer.ledger.get(caught.value.event_id).event.kind == "accepted"
    point.refresh()
    assert point.Q1.rate == 5300.0
    assert point.meta("Q1.rate") == {"stderr": 0.2}
    assert point.Q1.duration == 10.0
    assert point.meta("Q1.duration") == {"stderr": 0.5, "comment": "kept"}
    assert point.description is None
