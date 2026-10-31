from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.acquisition_facts import StopDecision
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.program.acquisition import AcquisitionContractError, CancelFlagProtocol
from zcu_tools.program.v2 import Module
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

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
from tests.program._acquisition_soc import make_acquisition_soc


def _overrun_acquisition(monkeypatch: pytest.MonkeyPatch, *, recover: bool):
    soc, soccfg = make_acquisition_soc()
    program = ModularProgramV2(
        soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_ch=0,
                ro_length=1.0,
                freq=100.0,
            ).build("readout")
        ],
        reps=3,
    )
    length = program.ro_chs[0]["length"]

    def chunk(shots: int, value: int):
        return (shots, ([np.tile([value * length, -value * length], (shots, 1))], {}))

    # Complete one round, then store a partial chunk before the offending one.
    failed = [[chunk(3, 91)], [chunk(1, 99)], [chunk(3, 99)]]
    batches = iter(failed + ([[chunk(3, 4)], [chunk(3, 6)]] if recover else failed))
    polls: list[int] = []

    def poll():
        polls.append(1)
        return next(batches)

    monkeypatch.setattr(soc, "poll_data", poll)
    return soc, program, polls


@pytest.mark.parametrize("retry", [0, 1])
@pytest.mark.parametrize("mode", ["builder", "integrated", "summed"])
def test_shot_overrun_obeys_explicit_retry_budget(
    monkeypatch: pytest.MonkeyPatch, retry: int, mode: str
) -> None:
    soc, program, polls = _overrun_acquisition(monkeypatch, recover=True)
    buffer = SignalBuffer((), dtype=np.complex128)
    buffer.set(np.array(7 + 8j))
    with Schedule(_cfg(), buffer, scope=Scope("run")) as sched:
        with nullcontext() if retry else pytest.raises(AcquisitionContractError):
            if mode == "builder":
                sched.prog_builder(soc, "unused").run_program(
                    program, rounds=2, options=AcquireOptions(retry=retry)
                )
            elif mode == "integrated":
                sched.run_acquire(
                    program, soc, rounds=2, options=RunOptions(retry=retry)
                )
            else:
                sched.run_acquire_summed(
                    program,
                    soc,
                    rounds=2,
                    processor=lambda iq: iq[0][..., 0, 0].sum(axis=1),
                    options=RunOptions(retry=retry),
                )

        assert len(polls) == (5 if retry else 3)
        assert sched.outcome.status == ("completed" if retry else "failed")
        if retry:
            np.testing.assert_allclose(buffer.array, 30 if mode == "summed" else 5 - 5j)
        else:
            assert isinstance(sched.outcome.exception, AcquisitionContractError)
            np.testing.assert_array_equal(buffer.array, 7 + 8j)


def test_shot_overrun_restarts_all_rounds_with_one_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    result = None
    soc, program, _ = _overrun_acquisition(monkeypatch, recover=True)
    buffer = SignalBuffer((), dtype=np.complex128)
    with Schedule(_cfg(), buffer, scope=Scope("run")) as sched:
        result = sched.prog_builder(soc, "unused").run_program(
            program, rounds=2, options=AcquireOptions(retry=1)
        )

    assert sched.outcome.status == "completed"
    assert result is not None
    np.testing.assert_allclose(result, 5 - 5j)
    np.testing.assert_allclose(buffer.array, 5 - 5j)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "expected_shots=3, received_shots=1, chunk_shots=3" in warnings[0].message
    assert "round_index=1" in warnings[0].message
    assert "attempt 2" in warnings[0].message
    raw = program.get_raw()
    assert raw is not None
    np.testing.assert_array_equal(
        raw[0], np.tile([6, -6], (3, 1, 1)) * program.ro_chs[0]["length"]
    )


def test_repeated_shot_overrun_exhausts_explicit_retry_and_discards_failed_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soc, program, _ = _overrun_acquisition(monkeypatch, recover=False)
    buffer = SignalBuffer((), dtype=np.complex128)
    with (
        pytest.raises(AcquisitionContractError) as caught,
        Schedule(_cfg(), buffer, scope=Scope("run")) as sched,
    ):
        sched.prog_builder(soc, "unused").run_program(
            program, rounds=2, options=AcquireOptions(retry=1)
        )

    assert sched.outcome.status == "failed"
    assert sched.outcome.exception is caught.value
    assert sched.stop.error is not None
    assert np.isnan(buffer.array)


def test_build_and_acquire_stop_condition_stops_program_without_schedule_stop():
    result = None
    signals_buffer = SignalBuffer((1,), dtype=np.float64)
    run_stop = StopSignal()

    with Schedule(
        _cfg(reps=4, rounds=2), signals_buffer, scope=Scope("run", stop=run_stop)
    ) as sched:
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
@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("stop_value", [12.5, float("inf")])
def test_data_stop_keeps_first_completed_round_and_cancels_later_rounds(
    decimated: bool,
    rounds: int,
    stop_value: float,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="zcu_tools.experiment.v2.runtime.program_builder")
    result = None
    produced_rounds: list[int] = []
    checked_signals: list[NDArray[np.float64]] = []
    signals_buffer = SignalBuffer((1,), dtype=np.float64)
    run_stop = StopSignal()

    class CooperativeProgram:
        def acquire(
            self,
            soc: object,
            *,
            rounds: int,
            progress: bool,
            round_hook: Callable[
                [int, NDArray[np.float64], CancelFlagProtocol], object
            ],
            cancel_flag: CancelFlagProtocol,
            **_kwargs: object,
        ) -> NDArray[np.float64]:
            raw = np.array([np.nan])
            for ir in range(1, rounds + 1):
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
            rounds: int,
            progress: bool,
            round_hook: Callable[
                [int, NDArray[np.float64], CancelFlagProtocol], object
            ],
            cancel_flag: CancelFlagProtocol,
            **kwargs: object,
        ) -> NDArray[np.float64]:
            return self.acquire(
                soc,
                rounds=rounds,
                progress=progress,
                round_hook=round_hook,
                cancel_flag=cancel_flag,
                **kwargs,
            )

    def stop_after_written_round() -> StopDecision:
        checked_signals.append(signals_buffer.array.copy())
        return StopDecision(met=True, value=stop_value)

    with Schedule(
        _cfg(rounds=rounds), signals_buffer, scope=Scope("run", stop=run_stop)
    ) as sched:
        builder = sched.prog_builder("soc", "soccfg")
        runner = builder.run_program_decimated if decimated else builder.run_program
        result = runner(
            CooperativeProgram(),
            rounds=rounds,
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
    expected_value = stop_value if np.isfinite(stop_value) else None
    stops = [record for record in caplog.records if record.levelname == "INFO"]
    if rounds > 1:
        assert len(stops) == 1
        assert "completed rounds 1" in stops[0].message
        assert f"stop value {expected_value}" in stops[0].message
    else:
        assert stops == []
    np.testing.assert_allclose(checked_signals, [[1.0]])
    np.testing.assert_allclose(result, [1.0])
    np.testing.assert_allclose(signals_buffer.array, [1.0])


def test_build_and_acquire_retries_acquire_on_the_same_program(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = None
    FlakyBuildProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched:
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
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].message
    assert "attempt 2" in warnings[0].message


def test_build_and_acquire_retry_clears_shared_failure_cause():
    result = None
    FlakyBuildProgram.instances.clear()
    run_stop = StopSignal()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(
        _cfg(rounds=1), signals_buffer, scope=Scope("run", stop=run_stop)
    ) as sched:
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


def test_build_and_acquire_propagates_keyboard_interrupt_without_retrying():
    InterruptingProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with (
        pytest.raises(KeyboardInterrupt),
        Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched,
    ):
        (
            sched.prog_builder("soc", "soccfg", program_cls=InterruptingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=3)
            )
        )

    assert len(InterruptingProgram.instances) == 1
    assert InterruptingProgram.instances[0].acquire_count == 1
    np.testing.assert_allclose(signals_buffer.array, np.array([np.nan]), equal_nan=True)


@pytest.mark.parametrize("phase", ["producer", "converter"])
def test_run_acquire_propagates_same_interrupt_without_retry(phase: str) -> None:
    interrupt = KeyboardInterrupt("acquisition interrupted")
    attempts = 0

    class Program:
        def acquire(
            self, *_args: object, **_kwargs: object
        ) -> list[NDArray[np.float64]]:
            nonlocal attempts
            attempts += 1
            if phase == "producer":
                raise interrupt
            return [np.array([[[1.0, 0.0]]])]

    def convert(_raw: list[NDArray[np.float64]]) -> NDArray[np.complex128]:
        raise interrupt

    with (
        pytest.raises(KeyboardInterrupt) as caught,
        Schedule(_cfg(), SignalBuffer((1,)), scope=Scope("run")) as sched,
    ):
        sched.run_acquire(
            Program(), object(), options=RunOptions(retry=3, converter=convert)
        )
    assert caught.value is interrupt
    assert attempts == 1


def test_build_and_acquire_first_round_stop_returns_nan_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = None
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with Schedule(_cfg(rounds=2), signals_buffer, scope=Scope("run")) as sched:
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


def test_build_and_acquire_raises_after_retry_exhaustion_and_retains_buffer():
    AlwaysFailingProgram.instances.clear()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with (
        pytest.raises(RuntimeError, match="permanent failure") as caught,
        Schedule(_cfg(rounds=1), signals_buffer, scope=Scope("run")) as sched,
    ):
        (
            sched.prog_builder("soc", "soccfg", program_cls=AlwaysFailingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=1)
            )
        )
    assert sched.is_stop() is True
    assert sched.outcome.status == "failed"
    assert sched.outcome.exception is caught.value
    assert sched.outcome.reason == "RuntimeError: permanent failure"
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))
    assert len(AlwaysFailingProgram.instances) == 1
    assert AlwaysFailingProgram.instances[0].acquire_count == 2


def test_build_and_acquire_records_failed_outcome_on_run_stop_signal():
    AlwaysFailingProgram.instances.clear()
    run_stop = StopSignal()
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    with (
        pytest.raises(RuntimeError, match="permanent failure") as caught,
        Schedule(
            _cfg(rounds=1), signals_buffer, scope=Scope("run", stop=run_stop)
        ) as sched,
    ):
        (
            sched.prog_builder("soc", "soccfg", program_cls=AlwaysFailingProgram)
            .add(FakeModule("readout"))
            .build_and_acquire(
                raw2signal_fn=identity_array, options=AcquireOptions(retry=1)
            )
        )
    assert sched.outcome.status == "failed"
    assert sched.outcome.exception is caught.value
    assert sched.stop.error is not None
    assert sched.stop.error.status == "failed"

    with pytest.raises(
        ScheduleOutcomeError, match="RuntimeError: permanent failure"
    ) as exc_info:
        run_stop.raise_if_error()

    assert exc_info.value.exception is sched.outcome.exception
    assert not run_stop.is_set()
    np.testing.assert_allclose(signals_buffer.array, np.array([2.0]))


def test_build_and_acquire_raises_without_retry_when_stop_set_after_failed_attempt():
    stop = StopSignal()
    cause = RuntimeError("failure with stop")
    attempts: list[int] = []
    signals_buffer = SignalBuffer((1,), dtype=np.float64)

    class StopAfterFailedAcquireProgram:
        def __init__(
            self,
            soccfg: Any,
            *,
            modules: list[Module],
            reps: int = 1,
            initial_delay: float = 1.0,
            relax_delay: float = 1.0,
            sweep: list[tuple[str, Any]] | None,
        ) -> None:
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

    with (
        pytest.raises(RuntimeError) as caught,
        Schedule(
            _cfg(rounds=1), signals_buffer, scope=Scope("run", stop=stop)
        ) as sched,
    ):
        (
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
    assert caught.value is cause
    assert sched.outcome.exception is cause
    assert sched.outcome.status == "stopped"
    assert sched.outcome.reason == "stop requested"
    assert stop.error is None
    assert attempts == [1]
    np.testing.assert_allclose(signals_buffer.array, np.array([1.0]))
