"""Shared accepted-writeback executor over a bound point and opaque parameter metadata."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.resources.entry import (
    AcceptedPayload,
    PointView,
    ResultEntry,
    capture_point_storage,
    document_value,
    snapshot,
)
from zcu_tools.resources.entry.storage import bind_record_table
from zcu_tools.resources.entry.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    PointWritebackTarget,
    RecordWriteback,
    WaveformWriteback,
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


@pytest.mark.parametrize("fails", [False, True])
def test_executor_returns_each_committed_value_and_failure_prefix(
    point: PointView, fails: bool
) -> None:
    writer = target(point)
    items = (
        ParameterWriteback("Q1.ext.fit", {"gain": 0.1}),
        ParameterWriteback("Q1.ext.fit.gain", 0.2),
    )
    if fails:
        items += (ParameterWriteback("Q1.duration", "invalid"),)
        with pytest.raises(WritebackApplyError) as caught:
            apply_writeback(items, target=writer, evidence=evidence())
        assert hasattr(caught.value, "result"), "Failure must retain confirmed values"
        result = caught.value.result
    else:
        result = apply_writeback(items, target=writer, evidence=evidence())
        assert hasattr(result, "committed"), "Executor must return confirmed values"
    assert [(item.path, item.value) for item in result.committed] == (
        []
        if fails
        else [
            ("Q1.ext.fit", {"gain": 0.1}),
            ("Q1.ext.fit.gain", 0.2),
        ]
    )
    events = writer.ledger.events()
    assert len(events) == 1
    assert events[0].id == result.event_id
    assert events[0].kind == "accepted"
    point.refresh()
    if fails:
        assert document_value(snapshot(point), "Q1.ext") == {}
    else:
        assert document_value(snapshot(point), "Q1.ext.fit.gain") == 0.2


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
    event = writer.ledger.get(event_id.event_id).event
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


def test_invalid_parameter_rolls_back_whole_group_and_metadata(
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
    assert caught.value.applied == 0
    assert isinstance(caught.value.__cause__, ValidationError)
    assert writer.ledger.get(caught.value.event_id).event.kind == "accepted"
    point.refresh()
    assert point.Q1.rate == 5000.0
    assert point.meta("Q1.rate") == {"stderr": 9.0, "source": "old run"}
    assert point.Q1.duration == 10.0
    assert point.meta("Q1.duration") == {"stderr": 0.5, "comment": "kept"}
    assert point.description is None


def test_group_order_preserves_accepted_order_and_per_item_receipts(
    point: PointView,
) -> None:
    writer = target(point)
    waveform = ConstWaveformCfg(length=0.1)
    result = apply_writeback(
        (
            RecordWriteback("Q1", {"freq": 10.0}),
            WaveformWriteback("Q1.waveforms.base", waveform),
            ParameterWriteback("Q1.ext.fit", {"gain": 0.1}),
            WaveformWriteback("Q1.waveforms.base", ConstWaveformCfg(length=0.2)),
            ParameterWriteback("Q1.ext.fit.gain", 0.2),
            RecordWriteback("Q1", {"freq": 20.0}),
        ),
        target=writer,
        evidence=evidence(),
    )
    assert [(item.path, item.value) for item in result.committed[:4]] == [
        ("Q1.ext.fit", {"gain": 0.1}),
        ("Q1.ext.fit.gain", 0.2),
        ("Q1.waveforms.base", {"style": "const", "length": 0.1}),
        ("Q1.waveforms.base", {"style": "const", "length": 0.2}),
    ]
    assert [
        item.value["freq"]
        for item in result.committed[4:]
        if isinstance(item.value, dict)
    ] == [10.0, 20.0]
    event = writer.ledger.get(result.event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert [write.path for write in event.payload.writes] == [
        "Q1",
        "Q1.waveforms.base",
        "Q1.ext.fit",
        "Q1.waveforms.base",
        "Q1.ext.fit.gain",
        "Q1",
    ]
    assert len(writer.ledger.events()) == 1
    assert [item.index for item in result.committed] == [2, 4, 1, 3, 0, 5]


def test_invalid_library_rolls_back_mixed_family_group_and_keeps_point_prefix(
    point: PointView,
) -> None:
    writer = target(point)
    with pytest.raises(WritebackApplyError) as caught:
        apply_writeback(
            (
                WaveformWriteback("Q1.waveforms.base", ConstWaveformCfg(length=0.1)),
                RecordWriteback("Q1", {"freq": 10.0}),
                ParameterWriteback("Q1.rate", 5300.0),
                ModuleWriteback(
                    "Q1.waveforms.wrong",
                    PulseSpec(
                        ch=0, nqz=1, freq=100.0, gain=0.1, waveform="Q1.waveforms.base"
                    ),
                ),
                ParameterWriteback("Q1.duration", 12.0),
            ),
            target=writer,
            evidence=evidence(),
        )
    assert caught.value.applied == 2
    assert [(item.path, item.value) for item in caught.value.result.committed] == [
        ("Q1.rate", 5300.0),
        ("Q1.duration", 12.0),
    ]
    point.refresh()
    assert (point.Q1.rate, point.Q1.duration) == (5300.0, 12.0)
    storage = capture_point_storage(point)
    library = ResultEntry.open(
        "entry",
        result_root=storage.result_path.parent,
        database_root=storage.database_path.parent,
    ).module_library("a")
    assert library.snapshot().document().components == {}
    assert bind_record_table(storage.result_path, "Q1").latest("run-1") is None
    assert len(writer.ledger.events()) == 1


def test_record_numpy_producer_values_reach_ledger_and_record_history(
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
            (RecordWriteback("Q1", fields),), target=writer, evidence=evidence()
        )
    except ValueError as error:
        raise AssertionError(
            "Record Apply must accept finite numpy producer numbers, including nested JSON"
        ) from error
    event = writer.ledger.get(event_id.event_id).event
    assert isinstance(event.payload, AcceptedPayload)
    assert [(write.path, write.value) for write in event.payload.writes] == [
        ("Q1", expected)
    ]
    row = bind_record_table(storage.result_path, "Q1").latest("run-1")
    assert row is not None
    assert datetime.fromisoformat(str(row["at"])) == datetime.fromisoformat(event.at)
    assert row == expected | {
        "run_id": "run-1",
        "point": "a",
        "at": row["at"],
        "source": event_id.event_id,
    }
    # Normalization must not replace values in the producer's original tree.
    assert isinstance(fields["freq"], np.float64)
    assert type(fit["stderr"]) is np.float32


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
def test_record_invalid_producer_value_rejects_before_any_write(
    point: PointView, value: object
) -> None:
    storage = capture_point_storage(point)
    writer = PointWritebackTarget(point, storage)
    with pytest.raises(ValueError, match="strict JSON"):
        apply_writeback(
            (
                ParameterWriteback("Q1.rate", 5300.0),
                RecordWriteback("Q1", {"fit": [{"value": value}]}),
            ),
            target=writer,
            evidence=evidence(),
        )
    assert writer.ledger.events() == ()
    assert bind_record_table(storage.result_path, "Q1").latest("run-1") is None
    point.refresh()
    assert point.Q1.rate == 5000.0
