from __future__ import annotations

from typing import Any

import numpy as np
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime.schedule import Schedule

from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import (
    RecordingBuffer,
    RootUpdateRecordingBuffer,
)


def test_schedule_child_buffer_syncs_result_buffer_without_throttling() -> None:
    root = {"task": {"signals": np.full((1,), np.nan)}}
    updates: list[tuple[tuple[Any, ...], np.ndarray]] = []
    result_buffer = RecordingBuffer(
        root,
        on_update=lambda step: updates.append(
            (step.path, root["task"]["signals"].copy())
        ),
    )

    def child(step):
        signals_step = step.child("signals")
        buffer = signals_step.buffer((1,), dtype=np.float64)
        buffer.set(np.array([1.0]))
        buffer.set(np.array([2.0]))

    with Schedule(_cfg(), result_buffer, scope=Scope("run")) as sched:
        sched.batch({"task": child})

    np.testing.assert_allclose(root["task"]["signals"], np.array([2.0]))
    assert [path for path, _ in updates] == [("task", "signals"), ("task", "signals")]
    np.testing.assert_allclose(updates[0][1], np.array([1.0]))
    np.testing.assert_allclose(updates[1][1], np.array([2.0]))


def test_schedule_rejects_multiple_root_buffers_at_construction() -> None:
    with np.testing.assert_raises_regex(ValueError, "at most one root result buffer"):
        Schedule(
            _cfg(),
            RecordingBuffer(np.zeros((1,))),
            RecordingBuffer(np.zeros((1,))),
            scope=Scope("run"),
        )


def test_schedule_rejects_registering_multiple_root_buffers() -> None:
    first = RecordingBuffer(np.zeros((1,)))
    second = RecordingBuffer(np.zeros((1,)))

    with (
        Schedule(_cfg(), first, scope=Scope("run")) as sched,
        np.testing.assert_raises_regex(ValueError, "at most one root result buffer"),
    ):
        sched.register_buffer(second)


def test_schedule_allows_registering_one_root_buffer_later() -> None:
    buffer = RecordingBuffer(np.zeros((1,)))

    with Schedule(_cfg(), scope=Scope("run")) as sched:
        sched.register_buffer(buffer)
        assert sched.data is buffer.data


def test_schedule_root_trigger_update_passes_none_step() -> None:
    buffer = RootUpdateRecordingBuffer(np.zeros((1,)))

    with Schedule(_cfg(), buffer, scope=Scope("run")) as sched:
        sched.trigger_update(flush=True)

    assert buffer.calls == [(None, True)]


def test_schedule_child_buffer_validates_target_before_acquire() -> None:
    root = {
        "child": {
            "signals": np.full((2,), np.nan),
            "metadata": {"value": 0},
        }
    }

    def child(step):
        signals_step = step.child("signals")
        try:
            signals_step.buffer((1,), dtype=np.float64)
        except ValueError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("buffer should reject shape mismatch")

        metadata_step = step.child("metadata")
        try:
            metadata_step.buffer((1,), dtype=np.float64)
        except ValueError as exc:
            assert "NDArray" in str(exc)
        else:
            raise AssertionError("buffer should reject non-array targets")

    with Schedule(_cfg(), RecordingBuffer(root), scope=Scope("run")) as sched:
        sched.batch({"child": child})
