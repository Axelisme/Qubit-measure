from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.program.acquisition import CancelFlagProtocol
from zcu_tools.program.v2 import Module, ProgramV2Cfg

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
                options=AcquireOptions(stop_condition=lambda: True),
            )
        )
        assert sched.outcome.status == "completed"
        assert run_stop.is_set() is False
        assert run_stop.error is None
        run_stop.raise_if_error()

    assert result is not None
    np.testing.assert_allclose(result, np.array([2.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))


@pytest.mark.parametrize("decimated", [False, True])
def test_data_stop_keeps_first_completed_round_and_cancels_later_rounds(
    decimated: bool,
) -> None:
    result = None
    produced_rounds: list[int] = []
    checked_signals: list[NDArray[np.float64]] = []
    signals_buffer = SignalBuffer((1,), dtype=np.float64)
    run_stop = StopSignal()

    class CooperativeProgram:
        cfg_model = ProgramV2Cfg(rounds=3)

        def acquire(
            self,
            soc: object,
            *,
            progress: bool,
            round_hook: Callable[
                [int, NDArray[np.float64], CancelFlagProtocol], object
            ],
            cancel_flag: CancelFlagProtocol,
            **_kwargs: object,
        ) -> NDArray[np.float64]:
            raw = np.array([np.nan])
            for ir in range(1, self.cfg_model.rounds + 1):
                if cancel_flag.is_set():
                    break
                produced_rounds.append(ir)
                raw = np.array([float(ir)])
                round_hook(ir, raw, cancel_flag)
            return raw

        def acquire_decimated(
            self,
            soc: object,
            *,
            progress: bool,
            round_hook: Callable[
                [int, NDArray[np.float64], CancelFlagProtocol], object
            ],
            cancel_flag: CancelFlagProtocol,
            **kwargs: object,
        ) -> NDArray[np.float64]:
            return self.acquire(
                soc,
                progress=progress,
                round_hook=round_hook,
                cancel_flag=cancel_flag,
                **kwargs,
            )

    def stop_after_written_round() -> bool:
        checked_signals.append(signals_buffer.array.copy())
        return True

    with Schedule(_cfg(rounds=3), signals_buffer, stop=run_stop) as sched:
        builder = sched.prog_builder("soc", "soccfg")
        runner = builder.run_program_decimated if decimated else builder.run_program
        result = runner(
            CooperativeProgram(),
            raw2signal_fn=identity_array,
            options=AcquireOptions(stop_condition=stop_after_written_round),
        )
        assert sched.outcome.status == "completed"
        assert sched.outcome.reason is None
        assert sched.outcome.exception is None
        assert not run_stop.is_set()
        assert run_stop.error is None
        run_stop.raise_if_error()

    assert result is not None
    assert produced_rounds == [1]
    np.testing.assert_allclose(checked_signals, [[1.0]])
    np.testing.assert_allclose(result, [1.0])
    np.testing.assert_allclose(signals_buffer.array, [1.0])


def test_build_and_acquire_retries_acquire_on_the_same_program():
    result = None
    FlakyBuildProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=StopSignal()) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FlakyBuildProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=1)
            )
        )
        assert sched.outcome.status == "completed"

    assert result is not None
    assert np.allclose(result, [4.0])
    assert np.allclose(signals_buffer.array, [4.0])
    assert len(FlakyBuildProgram.instances) == 1
    assert FlakyBuildProgram.instances[0].acquire_count == 2


def test_build_and_acquire_retry_clears_shared_failure_cause():
    result = None
    FlakyBuildProgram.instances.clear()
    run_stop = StopSignal()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=run_stop) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=FlakyBuildProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=1)
            )
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
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=3)
            )
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
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=3)
            )
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
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=1)
            )
        )
        assert sched.is_stop() is True
        assert sched.outcome.status == "failed"
        assert isinstance(sched.outcome.exception, RuntimeError)
        assert sched.outcome.reason == "RuntimeError: permanent failure"

    assert result is not None
    np.testing.assert_allclose(result, np.array([2.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))
    assert len(AlwaysFailingProgram.instances) == 1
    assert AlwaysFailingProgram.instances[0].acquire_count == 2


def test_build_and_acquire_records_failed_outcome_on_run_stop_signal():
    result = None
    AlwaysFailingProgram.instances.clear()
    run_stop = StopSignal()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, stop=run_stop) as sched:
        result = (
            sched.prog_builder("soc", "soccfg", program_cls=AlwaysFailingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=1)
            )
        )
        assert sched.outcome.status == "failed"
        assert run_stop.error is None
        assert sched.stop.error is not None
        assert sched.stop.error.status == "failed"

    with pytest.raises(
        ScheduleOutcomeError, match="RuntimeError: permanent failure"
    ) as exc_info:
        run_stop.raise_if_error()

    assert exc_info.value.exception is sched.outcome.exception
    assert not run_stop.is_set()
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
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=3)
            )
        )
        assert sched.outcome.exception is cause
        assert sched.outcome.status == "stopped"
        assert sched.outcome.reason == "stop requested"
        assert stop.error is None

    assert attempts == [1]
    assert result is not None
    np.testing.assert_allclose(result, np.array([1.0]))
    np.testing.assert_allclose(signals_buffer.array, np.array([1.0]))
