from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer

from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import (
    RecordingBuffer,
    RootUpdateRecordingBuffer,
)


@pytest.mark.parametrize("use_slot", [False, True], ids=["buffer", "slot"])
@pytest.mark.parametrize(
    "value",
    [np.array(7.0), np.array([7.0]), np.array([[7.0, 8.0]]), np.ones(3)],
    ids=["scalar", "length-one", "extra-axis", "wrong-length"],
)
def test_signal_writes_reject_shape_mismatch_without_update(
    use_slot: bool, value: np.ndarray
) -> None:
    updates: list[np.ndarray] = []
    buffer = SignalBuffer(
        (2, 2) if use_slot else (2,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=None,
    )
    target = buffer[0] if use_slot else buffer

    with pytest.raises(ValueError, match="shape") as error:
        target.set(value)

    assert f"target shape={(2,)}" in str(error.value)
    assert f"value shape={value.shape}" in str(error.value)
    np.testing.assert_array_equal(buffer.data, np.full(buffer.data.shape, np.nan))
    assert updates == []


def test_signal_writes_accept_exact_array_and_scalar_shapes() -> None:
    updates: list[np.ndarray] = []
    buffer = SignalBuffer(
        (2, 2),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=None,
    )

    buffer.set(np.array([[1.0, 2.0], [3.0, 4.0]]))
    buffer[0].set(np.array([5.0, 6.0]))
    scalar_slot = buffer[1, 0]
    scalar_slot.set(np.array(7.0))

    assert scalar_slot.value.shape == ()
    np.testing.assert_array_equal(buffer.data, [[5.0, 6.0], [7.0, 4.0]])
    assert len(updates) == 3
    np.testing.assert_array_equal(updates[-1], [[5.0, 6.0], [7.0, 4.0]])


@pytest.mark.parametrize("use_step", [False, True], ids=["root", "child"])
@pytest.mark.parametrize(
    "value",
    [7.0, np.array(7.0), np.array([7.0]), np.array([[7.0, 8.0]])],
    ids=["number", "scalar-array", "length-one", "extra-axis"],
)
def test_schedule_array_writes_reject_shape_mismatch_without_update(
    use_step: bool, value: Any
) -> None:
    storage = np.array([1.0, 2.0])
    buffer = RootUpdateRecordingBuffer([{"signals": storage}] if use_step else storage)

    with Schedule(_cfg(), buffer, scope=Scope("run")) as sched:
        _, step = next(sched.scan("point", [0]))
        target = step.child("signals") if use_step else sched
        with pytest.raises(ValueError, match="shape") as error:
            target.set_data(value, flush=True)

        assert "target shape=(2,)" in str(error.value)
        assert f"value shape={np.shape(value)}" in str(error.value)
        np.testing.assert_array_equal(storage, [1.0, 2.0])
        assert buffer.calls == []


@pytest.mark.parametrize("use_step", [False, True], ids=["root", "child"])
@pytest.mark.parametrize("scalar", [False, True], ids=["array", "scalar"])
def test_schedule_array_writes_accept_exact_shapes(
    use_step: bool, scalar: bool
) -> None:
    storage = np.full(() if scalar else (2,), np.nan)
    buffer = RootUpdateRecordingBuffer([{"signals": storage}] if use_step else storage)
    value = 7.0 if scalar else np.array([7.0, 8.0])

    with Schedule(_cfg(), buffer, scope=Scope("run")) as sched:
        _, step = next(sched.scan("point", [0]))
        target = step.child("signals") if use_step else sched
        target.set_data(value, flush=True)

        np.testing.assert_array_equal(storage, value)
        assert len(buffer.calls) == 1
        assert buffer.calls[0][1] is True


def test_schedule_child_buffer_syncs_result_buffer_without_throttling() -> None:
    root = [{"signals": np.full((1,), np.nan)}]
    updates: list[tuple[tuple[Any, ...], np.ndarray]] = []
    result_buffer = RecordingBuffer(
        root,
        on_update=lambda step: updates.append(
            (step.path, root[0]["signals"].copy())
        ),
    )

    with Schedule(_cfg(), result_buffer, scope=Scope("run")) as sched:
        for _, step in sched.scan("point", [0]):
            signals_step = step.child("signals")
            buffer = signals_step.buffer((1,), dtype=np.float64)
            buffer.set(np.array([1.0]))
            buffer.set(np.array([2.0]))

    np.testing.assert_allclose(root[0]["signals"], np.array([2.0]))
    assert [path for path, _ in updates] == [(0, "signals"), (0, "signals")]
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
    root = [
        {
            "signals": np.full((2,), np.nan),
            "metadata": {"value": 0},
        }
    ]

    with Schedule(_cfg(), RecordingBuffer(root), scope=Scope("run")) as sched:
        for _, step in sched.scan("point", [0]):
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
