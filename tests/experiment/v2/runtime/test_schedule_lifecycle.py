from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer

from tests.experiment.v2.runtime._flow_cfg import FlowCfg
from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import RecordingBuffer


def test_schedule_uses_own_stop_signal():
    stop = StopSignal()
    stop.set()

    with Schedule(_cfg(), scope=Scope("run", stop=stop)) as sched:
        assert sched.is_stop() is True

    stop.clear_stop()
    with Schedule(_cfg(), scope=Scope("run", stop=stop)) as sched:
        assert sched.is_stop() is False
        sched.set_stop()
        assert stop.is_set() is False
        assert sched.stop.is_set() is True


@pytest.mark.parametrize("ending", ["completed", "stopped", "error", "flush_error"])
def test_schedule_attaches_only_inside_context_and_releases_on_every_exit(
    ending: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = StopSignal()
    fail_render = False
    now = 100.0

    def render(_data: np.ndarray) -> None:
        nonlocal now
        now += 1.0
        if fail_render:
            raise ValueError("flush error")

    monkeypatch.setattr(time, "time", lambda: now)
    buffer = SignalBuffer(1, on_update=render, update_interval=0.1)
    root = Scope("run", stop=stop)
    sched = Schedule(_cfg(), buffer, scope=root)
    assert sched.stop is stop
    try:
        with sched:
            assert sched.stop is not stop, "Schedule must own a child scope"
            assert root.active_path == "run/schedule#1"
            if ending == "stopped":
                sched.set_stop()
            elif ending == "error":
                raise ValueError("body error")
            elif ending == "flush_error":
                buffer.set(np.array([1.0]))
                buffer.set(np.array([2.0]))
                fail_render = True
    except ValueError:
        assert ending in {"error", "flush_error"}
    assert root.active_path == "run"
    assert not stop.is_set()
    with Schedule(_cfg(), scope=root) as following:
        assert following.scope.sequence == 2
        assert not following.is_stop()


def test_schedule_operations_require_context():
    sched = Schedule(
        _cfg(rounds=1), SignalBuffer((1,), dtype=np.float64), scope=Scope("run")
    )

    try:
        sched.prog_builder("soc", "soccfg", marker="x")
    except RuntimeError as exc:
        assert "with Schedule" in str(exc)
    else:
        raise AssertionError("Schedule operations should require context manager")


@pytest.mark.parametrize("loop", ["scan", "repeat"])
def test_schedule_body_interrupt_preserves_partial_and_marks_signal(loop: str) -> None:
    buffer = SignalBuffer(3, dtype=np.float64)
    stop = StopSignal()
    interrupt = KeyboardInterrupt("host body stopped")
    with Schedule(FlowCfg(), buffer, scope=Scope("run", stop=stop)) as sched:
        steps = (
            sched.scan("point", range(3))
            if loop == "scan"
            else sched.repeat("point", 3)
        )
        for index, step in steps:
            if index == 1:
                raise interrupt
            buffer[step].set(np.array(7.0))
    np.testing.assert_allclose(buffer.data, [7.0, np.nan, np.nan], equal_nan=True)
    assert sched.outcome.status == "interrupted"
    assert sched.outcome.exception is interrupt
    assert not stop.is_set()
    assert sched.stop.is_set()
    with pytest.raises(ScheduleOutcomeError) as caught:
        stop.raise_if_error()
    assert caught.value.status == "interrupted"
    assert caught.value.__cause__ is interrupt


@pytest.mark.parametrize("error", [ValueError("body failure"), SystemExit("exit")])
def test_schedule_body_other_exception_still_propagates(error: BaseException) -> None:
    buffer = SignalBuffer(3, dtype=np.float64)
    stop = StopSignal()
    buffer.set(np.array([7.0, np.nan, np.nan]))
    sched = Schedule(FlowCfg(), buffer, scope=Scope("run", stop=stop))
    with pytest.raises(type(error)) as caught, sched:
        raise error
    assert caught.value is error
    assert sched.outcome.status == "completed"
    assert stop.error is None
    np.testing.assert_allclose(buffer.data, [7.0, np.nan, np.nan], equal_nan=True)


def test_schedule_repeat_obeys_stop_signal():
    seen: list[int] = []

    with Schedule(_cfg(), scope=Scope("run")) as sched:
        for index, _step in sched.repeat("round", 3):
            seen.append(index)
            sched.set_stop()
        assert sched.outcome.status == "stopped"

    assert seen == [0]


@pytest.mark.parametrize("stopped", [False, True], ids=["completed", "stopped"])
def test_schedule_exit_publishes_throttled_final_data(
    monkeypatch: pytest.MonkeyPatch, stopped: bool
) -> None:
    now = 100.0
    updates: list[np.ndarray[Any, Any]] = []

    def render(data: np.ndarray[Any, Any]) -> None:
        nonlocal now
        updates.append(data.copy())
        # Rendering consumes virtual time; no wall-clock wait or race is needed.
        now += 1.0

    monkeypatch.setattr(time, "time", lambda: now)
    buffer = SignalBuffer((2, 2), on_update=render)
    partial = np.array([[1.0, 2.0], [np.nan, np.nan]])
    final = np.array([[1.0, 2.0], [3.0, 4.0]])
    with Schedule(FlowCfg(), buffer, scope=Scope("run")) as sched:
        buffer.set(partial)
        buffer.set(final)
        assert len(updates) == 1  # The second write is still throttled.
        np.testing.assert_array_equal(updates[-1], partial)
        if stopped:
            sched.set_stop()

    np.testing.assert_array_equal(buffer.data, final)
    np.testing.assert_array_equal(
        updates[-1], final, err_msg="Schedule exit must publish the final buffer"
    )
    assert sched.outcome.status == ("stopped" if stopped else "completed")


@pytest.mark.parametrize("delivery", ["flush", "elapsed"])
def test_schedule_exit_does_not_repeat_delivered_update(
    monkeypatch: pytest.MonkeyPatch, delivery: str
) -> None:
    now = 100.0
    updates: list[np.ndarray[Any, Any]] = []

    def render(data: np.ndarray[Any, Any]) -> None:
        nonlocal now
        updates.append(data.copy())
        now += 1.0

    monkeypatch.setattr(time, "time", lambda: now)
    buffer = SignalBuffer((2,), on_update=render)
    final = np.array([1.0, 2.0])
    with Schedule(FlowCfg(), buffer, scope=Scope("run")):
        buffer.set(np.array([1.0, np.nan]))
        buffer.set(final)
        assert len(updates) == 1
        if delivery == "flush":
            buffer.trigger_update(flush=True)
        else:
            now = 200.0
            buffer.trigger_update()
        assert len(updates) == 2
        np.testing.assert_array_equal(updates[-1], final)

    assert len(updates) == 2, "Schedule exit must not repeat delivered data"


@pytest.mark.parametrize("interrupted", [False, True], ids=["exception", "interrupt"])
def test_schedule_body_exception_is_not_replaced_by_final_callback_failure(
    monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    now = 100.0
    fail_render = False

    def render(data: np.ndarray[Any, Any]) -> None:
        nonlocal now
        now += 1.0
        if fail_render:
            raise RuntimeError("presentation failed")

    monkeypatch.setattr(time, "time", lambda: now)
    buffer = SignalBuffer((2,), on_update=render)
    stop = StopSignal()
    sched = Schedule(FlowCfg(), buffer, scope=Scope("run", stop=stop))
    error = (
        KeyboardInterrupt("body interrupted")
        if interrupted
        else ValueError("body failed")
    )

    def run() -> None:
        nonlocal fail_render
        with sched:
            buffer.set(np.array([1.0, np.nan]))
            buffer.set(np.array([1.0, 2.0]))
            fail_render = True
            raise error

    if interrupted:
        run()
        assert sched.outcome.status == "interrupted"
        assert sched.outcome.exception is error
        with pytest.raises(ScheduleOutcomeError) as caught:
            stop.raise_if_error()
        assert caught.value.__cause__ is error
    else:
        with pytest.raises(ValueError, match="body failed") as caught_body:
            run()
        assert caught_body.value is error
        assert sched.outcome.status == "completed"
        assert stop.error is None
    np.testing.assert_array_equal(buffer.data, [1.0, 2.0])


def test_schedule_step_data_operations_require_context() -> None:
    step = None
    root = {"child": {"signals": np.full((1,), np.nan)}}

    with Schedule(_cfg(), RecordingBuffer(root), scope=Scope("run")) as sched:
        step = next(iter(sched.batch({"child": lambda child: child}).values()))
        assert step.path == ("child",)

    assert step is not None
    for operation in (
        lambda: step.data,
        lambda: step.set_data({"signals": np.array([1.0])}),
        lambda: step.child("signals"),
        step.trigger_update,
    ):
        try:
            operation()
        except RuntimeError as exc:
            assert "with Schedule" in str(exc)
        else:
            raise AssertionError(
                "ScheduleStep operations should require active context"
            )
