from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer

from tests.experiment.v2.runtime._flow_cfg import FlowCfg
from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg


def test_schedule_uses_own_stop_signal():
    stop = StopSignal()
    stop.set()

    with Schedule(_cfg(), stop=stop) as sched:
        assert sched.is_stop() is True

    stop.clear_stop()
    with Schedule(_cfg(), stop=stop) as sched:
        assert sched.is_stop() is False
        sched.set_stop()
        assert stop.is_set() is True


def test_schedule_operations_require_context():
    sched = Schedule(
        _cfg(rounds=1), SignalBuffer((1,), dtype=np.float64), stop=StopSignal()
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
    with Schedule(FlowCfg(), buffer, stop=stop) as sched:
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
    assert stop.is_set()
    with pytest.raises(ScheduleOutcomeError) as caught:
        stop.raise_if_error()
    assert caught.value.status == "interrupted"
    assert caught.value.__cause__ is interrupt


@pytest.mark.parametrize("error", [ValueError("body failure"), SystemExit("exit")])
def test_schedule_body_other_exception_still_propagates(error: BaseException) -> None:
    buffer = SignalBuffer(3, dtype=np.float64)
    stop = StopSignal()
    buffer.set(np.array([7.0, np.nan, np.nan]))
    sched = Schedule(FlowCfg(), buffer, stop=stop)
    with pytest.raises(type(error)) as caught, sched:
        raise error
    assert caught.value is error
    assert sched.outcome.status == "completed"
    assert stop.error is None
    np.testing.assert_allclose(buffer.data, [7.0, np.nan, np.nan], equal_nan=True)


def test_schedule_repeat_obeys_stop_signal():
    seen: list[int] = []

    with Schedule(_cfg(), stop=StopSignal()) as sched:
        for index, _step in sched.repeat("round", 3):
            seen.append(index)
            sched.set_stop()
        assert sched.outcome.status == "stopped"

    assert seen == [0]
