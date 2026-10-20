"""Shared accepted-writeback executor over a bound point and opaque parameter metadata."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.resources.entry import (
    AcceptedPayload,
    PointView,
    ResultEntry,
    capture_point_storage,
)
from zcu_tools.resources.entry.storage import bind_sample_table
from zcu_tools.resources.entry.writeback import (
    ParameterWriteback,
    PointWritebackTarget,
    SampleWriteback,
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


def test_sample_numpy_producer_values_reach_ledger_and_sample_history(
    point: PointView,
) -> None:
    storage = capture_point_storage(point)
    writer = PointWritebackTarget(point, storage)
    fit = {
        "stderr": np.float32(0.125),
        "counts": [np.int64(-9223372036854775808), np.uint64(18446744073709551615)],
        "values": [np.float16(0.5), {"center": np.longdouble("1.25")}],
    }
    fields = {
        "freq": np.float64(5123.25),
        "fit": fit,
        "native": [None, True, "µ", 2, 0.75],
        "dev_value": np.float32(0.125),
        "dev_unit": "V",
    }
    expected = {
        "freq": 5123.25,
        "fit": {
            "stderr": 0.125,
            "counts": [-9223372036854775808, 18446744073709551615],
            "values": [0.5, {"center": 1.25}],
        },
        "native": [None, True, "µ", 2, 0.75],
        "dev_value": 0.125,
        "dev_unit": "V",
    }
    try:
        event_id = apply_writeback(
            (SampleWriteback("Q1", fields),), target=writer, evidence=evidence()
        )
    except ValueError as error:
        raise AssertionError(
            "Sample Apply must accept finite numpy producer numbers, including nested JSON"
        ) from error
    event = writer.ledger.get(event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert [(write.path, write.value) for write in event.payload.writes] == [
        ("Q1", expected)
    ]
    row = bind_sample_table(storage.result_path, "Q1").latest("run-1")
    assert row == expected | {
        "run_id": "run-1",
        "point": "a",
        "at": event.at,
        "source": event_id,
    }
    # Normalization must not replace values in the producer's original tree.
    assert isinstance(fields["freq"], np.float64)
    assert isinstance(fit["stderr"], np.float32)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.float32("nan"), id="nan"),
        pytest.param(np.float64("inf"), id="positive-infinity"),
        pytest.param(np.longdouble("-inf"), id="negative-infinity"),
        pytest.param(float("nan"), id="native-nan"),
        pytest.param(
            np.longdouble(np.finfo(np.longdouble).max),
            id="finite-float-overflow",
            marks=pytest.mark.skipif(
                bool(np.finfo(np.longdouble).max <= np.finfo(np.float64).max),
                reason="longdouble has no extended range on this platform",
            ),
        ),
        pytest.param(1 + 2j, id="complex"),
        pytest.param(np.complex128(1 + 2j), id="numpy-complex"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param(np.array(1.25), id="zero-dimensional-array"),
        pytest.param(np.array([1.25]), id="array"),
        pytest.param((1, 2), id="tuple"),
        pytest.param(b"bytes", id="bytes"),
        pytest.param({1: "value"}, id="non-string-key"),
        pytest.param(object(), id="object"),
    ],
)
def test_sample_invalid_producer_value_rejects_before_any_write(
    point: PointView, value: object
) -> None:
    storage = capture_point_storage(point)
    writer = PointWritebackTarget(point, storage)
    with pytest.raises(ValueError, match="strict JSON"):
        apply_writeback(
            (
                ParameterWriteback("Q1.rate", 5300.0),
                SampleWriteback("Q1", {"fit": [{"value": value}]}),
            ),
            target=writer,
            evidence=evidence(),
        )
    assert writer.ledger.events() == ()
    assert bind_sample_table(storage.result_path, "Q1").latest("run-1") is None
    point.refresh()
    assert point.Q1.rate == 5000.0
