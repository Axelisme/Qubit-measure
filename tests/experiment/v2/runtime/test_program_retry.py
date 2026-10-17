from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.program.v2 import Module

from tests.experiment.v2.runtime._flow_cfg import FlowCfg
from tests.experiment.v2.runtime._flow_cfg import make_flow_cfg as _cfg
from tests.experiment.v2.runtime._flow_fakes import (
    AlwaysFailingProgram,
    FakeModule,
    FakeProgram,
    FlakyBuildProgram,
    InterruptingProgram,
    StopBeforeDataProgram,
    identity_array,
)


def test_build_and_acquire_stop_condition_stops_program_without_schedule_stop():
    result = None
    signals_buffer = SignalBuffer((1,), dtype=np.float64)
    run_stop = StopSignal()

    with Schedule(_cfg(reps=4, rounds=2), signals_buffer, stop=run_stop) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FakeProgram, marker="kw")
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array,
                stop_condition=lambda: True,
            )
        )
        assert sched.outcome.status == "completed"
        assert run_stop.is_set() is False
        assert run_stop.error is None
        run_stop.raise_if_error()

    assert result is not None
    np.testing.assert_allclose(result, np.array([2.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))


def test_build_and_acquire_rebuilds_program_on_retry():
    result = None
    FlakyBuildProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=StopSignal()) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FlakyBuildProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=1)
        )
        assert sched.outcome.status == "completed"

    assert result is not None
    assert np.allclose(result, [4.0])
    assert np.allclose(signals_buffer.array, [4.0])
    assert len(FlakyBuildProgram.instances) == 2
    assert FlakyBuildProgram.instances[0].acquire_count == 1
    assert FlakyBuildProgram.instances[1].acquire_count == 1


def test_build_and_acquire_retry_clears_shared_failure_cause():
    result = None
    FlakyBuildProgram.instances.clear()
    run_stop = StopSignal()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=run_stop) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FlakyBuildProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=1)
        )
        assert sched.outcome.status == "completed"
        assert run_stop.error is None
        run_stop.raise_if_error()

    assert result is not None
    np.testing.assert_allclose(result, np.array([4.0]))


def test_build_and_acquire_returns_partial_on_keyboard_interrupt_without_retrying():
    InterruptingProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=StopSignal()) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=InterruptingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=3)
        )
        assert sched.is_stop() is True
        assert sched.outcome.status == "interrupted"
        assert isinstance(sched.outcome.exception, KeyboardInterrupt)
        np.testing.assert_allclose(result, np.array([np.nan]), equal_nan=True)

    assert len(InterruptingProgram.instances) == 1
    assert InterruptingProgram.instances[0].acquire_count == 1
    np.testing.assert_allclose(signals_buffer.array, np.array([np.nan]), equal_nan=True)


def test_build_and_acquire_first_round_stop_returns_nan_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = None
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=2), signals_buffer, stop=StopSignal()) as sched:
        monkeypatch.setattr(StopBeforeDataProgram, "external_stop", sched.stop)
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=StopBeforeDataProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=3)
        )
        assert sched.is_stop() is True
        assert sched.outcome.status == "stopped"
        assert sched.outcome.reason == "stop requested"

    assert result is not None
    np.testing.assert_allclose(result, np.array([np.nan]), equal_nan=True)
    np.testing.assert_allclose(signals_buffer.array, np.array([np.nan]), equal_nan=True)
    assert sched.outcome.exception is None
    assert sched.stop.error is None


def test_build_and_acquire_returns_last_partial_after_retry_exhaustion():
    result = None
    AlwaysFailingProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=StopSignal()) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=AlwaysFailingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=1)
        )
        assert sched.is_stop() is True
        assert sched.outcome.status == "failed"
        assert isinstance(sched.outcome.exception, RuntimeError)
        assert sched.outcome.reason == "RuntimeError: permanent failure"

    assert result is not None
    np.testing.assert_allclose(result, np.array([2.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))
    assert len(AlwaysFailingProgram.instances) == 2
    assert AlwaysFailingProgram.instances[0].acquire_count == 1
    assert AlwaysFailingProgram.instances[1].acquire_count == 1


def test_build_and_acquire_records_failed_outcome_on_run_stop_signal():
    result = None
    AlwaysFailingProgram.instances.clear()
    run_stop = StopSignal()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=run_stop) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=AlwaysFailingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=1)
        )
        assert sched.outcome.status == "failed"
        assert run_stop.error is not None
        assert run_stop.error.status == "failed"

    with pytest.raises(
        ScheduleOutcomeError, match="RuntimeError: permanent failure"
    ) as exc_info:
        run_stop.raise_if_error()

    assert exc_info.value.exception is sched.outcome.exception
    assert result is not None
    np.testing.assert_allclose(result, np.array([2.0]))


def test_build_and_acquire_does_not_retry_when_stop_set_after_failed_attempt():
    result = None
    stop = StopSignal()
    cause = RuntimeError("failure with stop")
    attempts: list[int] = []
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    class StopAfterFailedAcquireProgram:
        def __init__(
            self,
            soccfg: Any,
            cfg: FlowCfg,
            *,
            modules: list[Module],
            sweep: list[tuple[str, Any]] | None,
        ) -> None:
            self.cfg_model = cfg
            self.modules = modules
            self.sweep = sweep

        def acquire(
            self, *_args: Any, round_hook, cancel_flag, **_kwargs: Any
        ) -> np.ndarray:
            attempts.append(1)
            raw = np.array([float(len(attempts))])
            round_hook(1, raw, cancel_flag)
            stop.set()
            raise cause

        def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
            raise NotImplementedError

    with Schedule(_cfg(rounds=1), signals_buffer, stop=stop) as sched:
        result = (
            sched.prog_builder(
                "soc",
                "soccfg",
                program_cls=StopAfterFailedAcquireProgram,
            )
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=3)
        )
        assert sched.outcome.exception is cause
        assert sched.outcome.status == "stopped"
        assert sched.outcome.reason == "stop requested"
        assert stop.error is None

    assert attempts == [1]
    assert result is not None
    np.testing.assert_allclose(result, np.array([1.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([1.0]))


def test_build_program_failure_does_not_retry_when_stop_requested():
    result = None
    stop = StopSignal()
    cause = RuntimeError("build failure with stop")
    attempts: list[int] = []
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    class StopAfterBuildFailureProgram:
        cfg_model: FlowCfg
        modules: list[Module]
        sweep: list[tuple[str, Any]] | None

        def __init__(
            self,
            soccfg: Any,
            cfg: FlowCfg,
            *,
            modules: list[Module],
            sweep: list[tuple[str, Any]] | None,
        ) -> None:
            attempts.append(1)
            stop.set()
            raise cause

        def acquire(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
            raise NotImplementedError

        def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
            raise NotImplementedError

    with Schedule(_cfg(rounds=1), signals_buffer, stop=stop) as sched:
        result = (
            sched.prog_builder(
                "soc",
                "soccfg",
                program_cls=StopAfterBuildFailureProgram,
            )
            .add(FakeModule("readout"))
            .build_and_acquire(raw2signal_fn=identity_array, retry=3)
        )
        assert sched.outcome.exception is cause
        assert sched.outcome.status == "stopped"
        assert sched.outcome.reason == "stop requested"
        assert stop.error is None

    assert attempts == [1]
    assert result is not None
    np.testing.assert_allclose(result, np.array([np.nan]), equal_nan=True)
    np.testing.assert_allclose(signals_buffer.array, np.array([np.nan]), equal_nan=True)
