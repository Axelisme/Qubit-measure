"""Engine effect boundary: failure source, Completed pairing, files, and live."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Never

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal
from zcu_tools.experiment.workflows import (
    Aborted,
    Actor,
    Completed,
    DeviceSetup,
    DeviceSnapshot,
    Done,
    Failed,
    JsonParameters,
    Live1D,
    Live2D,
    Live2DRow,
    Next,
    Run,
    Step,
    TunableChange,
    WorkflowEnv,
)
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import ProgramV2Cfg
from zcu_tools.program.v2.modules import SoftDelay

from ._engine_fakes import Cfg, Plan, Rig, State, Tunables


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def save_integer(source: Completed[Cfg, int], path: Path) -> None:
    path.write_text(f"{source.cfg.value}:{source.result}", encoding="utf-8")


def run_value(run: Run[Cfg]) -> int:
    return run.cfg.value


def start_effect(
    rig: Rig,
    experiment: Callable[[Run[Cfg]], int],
    *,
    saver: Callable[[Completed[Cfg, int], Path], None] = save_integer,
) -> list[Completed[Cfg, int] | Failed]:
    observed: list[Completed[Cfg, int] | Failed] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        outcome = yield from env.run(experiment, Cfg(), save=saver)
        observed.append(outcome)
        state.index += 1
        return Next(outcome.result if isinstance(outcome, Completed) else 0, state)

    rig.start(step, plan=Plan(count=1))
    return observed


def test_cfg_at_experiment_return_is_saved_before_typed_response(rig: Rig) -> None:
    original = Cfg()
    observed: list[Completed[Cfg, int] | Failed] = []
    saved: list[Completed[Cfg, int]] = []

    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))
        with run.schedule(buffer) as first:
            first.cfg.value = 10
            buffer.set(np.array([2 + 0j]))
        with run.schedule(buffer) as second:
            second.cfg.value = 20
        run.cfg.value = 30
        return 9

    def save(source: Completed[Cfg, int], path: Path) -> None:
        saved.append(source)
        save_integer(source, path)

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        outcome = yield from env.run(acquire, original, save=save)
        observed.append(outcome)
        assert len(saved) == 1
        assert tuple(env.iter_dir.parent.glob("runs/*.h5"))
        state.index = 1
        return Next(1, state)

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"
    assert isinstance(observed[0], Completed)
    assert observed[0].cfg == Cfg(value=30)
    assert saved[0] is observed[0]
    assert original == Cfg()
    assert rig.engine.records("run")[0].run_files == ("runs/01-acquire.h5",)
    final = rig.paths.data_root / "iter/000001/runs/01-acquire.h5"
    assert final.read_text(encoding="utf-8") == "30:9"


@pytest.mark.parametrize("rethrow", [False, True])
def test_schedule_failure_remains_recoverable_and_next_effect_has_fresh_signal(
    rig: Rig, rethrow: bool
) -> None:
    cause = OSError("acquire disconnected")
    signals: list[StopSignal] = []
    observed: list[Completed[Cfg, int] | Failed] = []

    def fail(run: Run[Cfg]) -> int:
        signals.append(run.cancel_signal)
        buffer = run.buffer((1,), axes=(np.array([0.0]),))

        def acquire(_step: object) -> None:
            raise cause

        with run.schedule(buffer) as schedule:
            schedule.batch({"acquire": acquire})
        if rethrow:
            run.cancel_signal.raise_if_error()
        return 99

    def recover(run: Run[Cfg]) -> int:
        signals.append(run.cancel_signal)
        assert not run.cancel_signal.is_set()
        return 7

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        observed.append((yield from env.run(fail, Cfg(), save=save_integer)))
        observed.append((yield from env.run(recover, Cfg(), save=save_integer)))
        state.index = 1
        return Next(7, state)

    rig.start(step)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done"
    assert isinstance(observed[0], Failed)
    assert isinstance(observed[1], Completed)
    assert signals[0] is not signals[1]
    assert rig.engine.records("run")[0].run_files == ("runs/02-recover.h5",)
    failed = next(
        event for event in rig.events() if event["kind"] == "experiment_failed"
    )
    assert failed["run_no"] == 1
    assert isinstance(failed["error"], dict)
    assert failed["error"]["type"] == "builtins.OSError"
    assert failed["error"]["message"] == "acquire disconnected"
    assert "raise cause" in str(failed["error"]["traceback"])


@pytest.mark.parametrize("source", ["other_signal", "equal_wrapper", "outside"])
def test_other_schedule_wrappers_and_experiment_errors_fail_run(
    rig: Rig, source: str
) -> None:
    cause = OSError("disconnected")

    def fail(run: Run[Cfg]) -> int:
        if source == "outside":
            raise cause
        if source == "other_signal":
            other = StopSignal()
            other.set_error("failed", "same message", cause)
            other.raise_if_error()
        else:
            buffer = run.buffer((1,), axes=(np.array([0.0]),))

            def acquire(_step: object) -> None:
                raise cause

            with run.schedule(buffer) as schedule:
                schedule.batch({"acquire": acquire})
            error = run.cancel_signal.error
            assert error is not None
            raise ScheduleOutcomeError("failed", error.reason, error.exception)
        return 0

    observed = start_effect(rig, fail)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert observed == []
    assert "experiment_failed" not in rig.kinds()
    assert "step_failed" in rig.kinds()
    assert rig.engine.records("run") == ()


@pytest.mark.parametrize("source", ["outside", "schedule", "schedule_rethrow"])
def test_keyboard_interrupt_is_stop_not_failed(rig: Rig, source: str) -> None:
    def interrupted(run: Run[Cfg]) -> int:
        if source == "outside":
            raise KeyboardInterrupt
        buffer = run.buffer((1,), axes=(np.array([0.0]),))

        def acquire(_step: object) -> None:
            raise KeyboardInterrupt

        with run.schedule(buffer) as schedule:
            schedule.batch({"acquire": acquire})
        if source == "schedule_rethrow":
            run.cancel_signal.raise_if_error()
        return 99

    observed = start_effect(rig, interrupted)
    status = rig.engine.execute("run")
    assert status.lifecycle == "stopped"
    assert observed == []
    assert rig.kinds().count("step_discarded") == 1
    assert "step_failed" not in rig.kinds()
    assert "experiment_failed" not in rig.kinds()
    assert not tuple(rig.paths.data_root.glob("iter/*/runs/*.h5"))


@pytest.mark.parametrize("control", ["pause", "stop"])
def test_cancel_during_experiment_never_saves_partial_or_returns_failed(
    rig: Rig, control: str
) -> None:
    def interrupted(run: Run[Cfg]) -> int:
        if control == "pause":
            rig.engine.pause("run")
        else:
            rig.engine.stop("run")
        assert run.cancel_signal.is_set()
        return 99

    observed = start_effect(rig, interrupted)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert observed == []
    assert rig.engine.records("run") == ()
    assert not tuple(rig.paths.data_root.glob("iter/*/runs/*.h5"))


def test_file_saved_before_pause_is_retained_and_resume_allocates_new_call(
    rig: Rig,
) -> None:
    observed: list[int] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        yield from env.run(run_value, Cfg(), save=save_integer)
        observed.append(1)
        yield from env.set_device("flux", 0.2)
        state.index = 1
        return Next(1, state)

    rig.start(step)
    rig.devices.on_set = lambda _signal: rig.engine.pause("run")
    assert rig.engine.execute("run").lifecycle == "paused"
    discarded = next(
        event for event in rig.events() if event["kind"] == "step_discarded"
    )
    assert discarded["run_files"] == ["runs/01-run_value.h5"]
    assert rig.engine.records("run") == ()
    rig.devices.on_set = None
    rig.engine.resume("run", devices=DeviceSnapshot(()))
    assert rig.engine.execute("run").lifecycle == "done"
    assert observed == [1, 1]
    assert len(tuple(rig.paths.data_root.glob("iter/*/runs/*.h5"))) == 2
    assert rig.engine.records("run")[0].call_seq == 2


def test_saver_failure_after_pause_keeps_true_failure_cause(rig: Rig) -> None:
    def fail_save(_source: Completed[Cfg, int], path: Path) -> None:
        path.write_text("incomplete", encoding="utf-8")
        rig.engine.pause("run")
        raise OSError("writer failed")

    observed = start_effect(rig, run_value, saver=fail_save)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "writer failed" in str(status.reason)
    assert observed == []
    assert "step_discarded" not in rig.kinds()
    assert not tuple(rig.paths.data_root.glob("iter/*/runs/[0-9]*.h5"))
    assert len(tuple(rig.paths.data_root.glob("iter/*/runs/.*.tmp.h5"))) == 1


def test_unlink_failure_keeps_final_but_does_not_deliver_completed(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.unlink

    def denied(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.endswith(".tmp.h5"):
            raise OSError("unlink denied")
        original(path, missing_ok=missing_ok)

    observed = start_effect(rig, run_value)
    monkeypatch.setattr(Path, "unlink", denied)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert observed == []
    failed = next(event for event in rig.events() if event["kind"] == "step_failed")
    assert failed["run_files"] == ["runs/01-run_value.h5"]
    assert len(tuple(rig.paths.data_root.glob("iter/*/runs/[0-9]*.h5"))) == 1
    assert len(tuple(rig.paths.data_root.glob("iter/*/runs/.*.tmp.h5"))) == 1


@pytest.mark.parametrize("setup", [True, False])
def test_cfg_device_setup_precedes_experiment_and_can_be_explicitly_skipped(
    rig: Rig, setup: bool
) -> None:
    settings = (DeviceSetup("flux", JsonParameters({"value": 0.1})),)

    def acquire(_run: Run[Cfg]) -> int:
        assert rig.devices.settings == ([settings] if setup else [])
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        yield from env.run(
            acquire, Cfg(dev=settings), save=save_integer, setup_devices=setup
        )
        state.index = 1
        return Next(1, state)

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"


@pytest.mark.parametrize("rethrow", [False, True])
def test_live_callback_caught_by_schedule_is_still_run_failure(
    rig: Rig, rethrow: bool
) -> None:
    observed: list[Completed[Cfg, int] | Failed] = []

    def projection(_values: NDArray[np.complex128]) -> NDArray[np.float64]:
        raise ValueError("live projection failed")

    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))

        def write(_step: object) -> None:
            buffer.set(np.array([2 + 0j]))

        with run.schedule(buffer) as schedule:
            schedule.batch({"write": write})
        if rethrow:
            run.cancel_signal.raise_if_error()
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        (line,) = env.axes("live").plot([], [])
        observed.append(
            (
                yield from env.run(
                    acquire, Cfg(), save=save_integer, live=Live1D(line, projection)
                )
            )
        )
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "live projection failed" in str(status.reason)
    assert observed == []
    assert "experiment_failed" not in rig.kinds()
    failed = next(event for event in rig.events() if event["kind"] == "step_failed")
    assert isinstance(failed["error"], dict)
    assert failed["error"]["type"] == "builtins.ValueError"


@pytest.mark.parametrize("display", ["line", "image", "row"])
def test_live_projects_without_overwriting_other_rows_or_changing_shape(
    rig: Rig, display: str
) -> None:
    def acquire(run: Run[Cfg]) -> int:
        if display == "image":
            buffer = run.buffer(
                (2, 2), axes=(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
            )
            buffer.set(np.array([[1 + 1j, 2 + 1j], [3 + 1j, 4 + 1j]]))
        else:
            buffer = run.buffer((2,), axes=(np.array([2.0, 3.0]),))
            buffer.set(np.array([1 + 1j, 2 + 1j]))
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        axes = env.axes("live")
        if display == "line":
            (line,) = axes.plot([], [])
            live = Live1D(line, lambda values: values.real)
        else:
            image = axes.imshow(np.full((2, 2), 9.0))
            live = (
                Live2D(image, lambda values: values.real)
                if display == "image"
                else Live2DRow(image, 1, lambda values: values.real)
            )
        yield from env.run(acquire, Cfg(), save=save_integer, live=live)
        if display == "line":
            np.testing.assert_array_equal(axes.lines[0].get_xdata(), [2.0, 3.0])
            np.testing.assert_array_equal(axes.lines[0].get_ydata(), [1.0, 2.0])
        else:
            expected = (
                [[1.0, 2.0], [3.0, 4.0]]
                if display == "image"
                else [[9.0, 9.0], [1.0, 2.0]]
            )
            np.testing.assert_array_equal(axes.images[0].get_array(), expected)
        state.index = 1
        return Next(1, state)

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"


@pytest.mark.parametrize("display", ["line", "image", "row"])
def test_live_finalization_projects_latest_values(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, display: str
) -> None:
    now = 1000.0
    at_return: list[NDArray[np.float64]] = []

    def wall_time() -> float:
        return now

    monkeypatch.setattr(time, "time", wall_time)

    def project(values: NDArray[np.complex128]) -> NDArray[np.float64]:
        nonlocal now
        # A costly projection followed immediately by another buffer update.
        now += 1.0
        return values.real

    def acquire(run: Run[Cfg]) -> int:
        shape = (2, 2) if display == "image" else (2,)
        axes = tuple(np.array([0.0, 1.0]) for _ in shape)
        buffer = run.buffer(shape, axes=axes)
        buffer.set(np.full(shape, 1.0 + 0j, dtype=np.complex128))
        buffer.set(np.full(shape, 7.0 + 0j, dtype=np.complex128))
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        axes = env.axes("live")
        if display == "line":
            (line,) = axes.plot([], [])
            live = Live1D(line, project)
        else:
            image = axes.imshow(np.full((2, 2), 9.0))
            live = (
                Live2D(image, project)
                if display == "image"
                else Live2DRow(image, 1, project)
            )
        yield from env.run(acquire, Cfg(), save=save_integer, live=live)
        values = (
            axes.lines[0].get_ydata()
            if display == "line"
            else axes.images[0].get_array()
        )
        at_return.append(np.asarray(values, dtype=np.float64).copy())
        state.index = 1
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "done"
    assert status.committed_seq == 1
    if display == "line":
        expected = np.array([7.0, 7.0])
    elif display == "image":
        expected = np.array([[7.0, 7.0], [7.0, 7.0]])
    else:
        expected = np.array([[9.0, 9.0], [7.0, 7.0]])
    np.testing.assert_array_equal(at_return[0], expected)
    snapshots = rig.plots.snapshots if display == "line" else rig.plots.image_snapshots
    np.testing.assert_array_equal(snapshots[-1][0], expected)


def test_workflow_can_abort_after_a_recoverable_failed_effect(rig: Rig) -> None:
    def failed(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))

        def acquire(_step: object) -> None:
            raise ValueError("calibration bad")

        with run.schedule(buffer) as schedule:
            schedule.batch({"acquire": acquire})
        return 0

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        outcome = yield from env.run(failed, Cfg(), save=save_integer)
        assert isinstance(outcome, Failed)
        return Aborted("calibration unavailable")

    rig.start(step)
    status = rig.engine.execute("run")
    assert (status.lifecycle, status.reason, status.committed_seq) == (
        "aborted",
        "calibration unavailable",
        0,
    )
    assert rig.engine.stop("run") == status


@pytest.mark.parametrize("control", ["pause", "stop"])
def test_real_schedule_failure_is_delivered_before_pending_control(
    rig: Rig, control: str
) -> None:
    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))

        def fail(_step: object) -> None:
            if control == "pause":
                rig.engine.pause("run")
            else:
                rig.engine.stop("run")
            raise OSError("acquire failed with control")

        with run.schedule(buffer) as schedule:
            schedule.batch({"acquire": fail})
        return 99

    observed = start_effect(rig, acquire)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert isinstance(observed[0], Failed)
    assert status.committed_seq == 1
    assert "experiment_failed" in rig.kinds()
    assert "step_discarded" not in rig.kinds()


@pytest.mark.parametrize("control", ["pause", "stop"])
@pytest.mark.parametrize("source", ["acquire", "build"])
@pytest.mark.parametrize("native_stop", [False, True])
def test_program_error_beats_control(
    rig: Rig, control: str, source: str, native_stop: bool
) -> None:
    message = f"program {source} failed with control"
    cause = OSError(message)
    attempts: list[str] = []

    def fail(schedule_stop: Callable[[], None]) -> Never:
        attempts.append(source)
        if control == "pause":
            rig.engine.pause("run")
        else:
            rig.engine.stop("run")
        if native_stop:
            schedule_stop()
        raise cause

    class FailingProgram:
        cfg_model = ProgramV2Cfg(rounds=1)

        def __init__(
            self,
            *_args: object,
            schedule_stop: Callable[[], None],
            **_kwargs: object,
        ) -> None:
            self.schedule_stop = schedule_stop
            if source == "build":
                fail(schedule_stop)

        def acquire(self, *_args: object, **_kwargs: object) -> NDArray[np.complex128]:
            fail(self.schedule_stop)

        def acquire_decimated(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("unexpected decimated acquire")

    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))
        with run.schedule(buffer) as schedule:
            builder = schedule.prog_builder(
                "soc",
                "soccfg",
                cfg=ProgramV2Cfg(),
                program_cls=FailingProgram,
                schedule_stop=schedule.set_stop,
            )
            if source == "build":
                builder.add(SoftDelay("wait", 0.0)).build_and_acquire(
                    raw2signal_fn=lambda raw: raw, retry=3
                )
            else:
                builder.run_program(
                    FailingProgram(schedule_stop=schedule.set_stop),
                    raw2signal_fn=lambda raw: raw,
                    retry=3,
                )
        return 99

    observed = start_effect(rig, acquire)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert len(observed) == 1
    assert isinstance(observed[0], Failed)
    assert message in observed[0].reason
    assert attempts == [source]
    assert status.committed_seq == 1
    failed = next(
        event for event in rig.events() if event["kind"] == "experiment_failed"
    )
    assert isinstance(failed["error"], dict)
    assert failed["error"]["type"] == "builtins.OSError"
    assert failed["error"]["message"] == message
    assert "raise cause" in str(failed["error"]["traceback"])
    assert "step_discarded" not in rig.kinds()
    assert not tuple(rig.paths.data_root.rglob("*.h5"))


@pytest.mark.parametrize("control", ["pause", "stop"])
@pytest.mark.parametrize("first_round_stop", [False, True])
@pytest.mark.parametrize("acquire_mode", ["integrated", "decimated"])
def test_program_cancel_protocol_never_returns_failed(
    rig: Rig, control: str, first_round_stop: bool, acquire_mode: str
) -> None:
    class CancelledProgram:
        cfg_model = ProgramV2Cfg(rounds=1)

        def acquire(self, *_args: object, **_kwargs: object) -> NDArray[np.complex128]:
            if control == "pause":
                rig.engine.pause("run")
            else:
                rig.engine.stop("run")
            if first_round_stop:
                raise StoppedPartialAcquireError(
                    "acquire stopped before the first round completed"
                )
            return np.array([1.0], dtype=np.complex128)

        def acquire_decimated(
            self, *_args: object, **_kwargs: object
        ) -> NDArray[np.complex128]:
            return self.acquire()

    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((1,), axes=(np.array([0.0]),))
        with run.schedule(buffer) as schedule:
            builder = schedule.prog_builder("soc", "soccfg", cfg=ProgramV2Cfg())
            if acquire_mode == "integrated":
                builder.run_program(CancelledProgram(), raw2signal_fn=lambda raw: raw)
            else:
                builder.run_program_decimated(
                    CancelledProgram(), raw2signal_fn=lambda raw: raw
                )
        return 99

    observed = start_effect(rig, acquire)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert not observed
    assert status.committed_seq == 0
    assert "step_discarded" in rig.kinds()
    assert "experiment_failed" not in rig.kinds()
    assert not tuple(rig.paths.data_root.rglob("*.h5"))


def test_pause_during_successful_saver_retains_final_without_delivering_completed(
    rig: Rig,
) -> None:
    def save(source: Completed[Cfg, int], path: Path) -> None:
        save_integer(source, path)
        rig.engine.pause("run")

    observed = start_effect(rig, run_value, saver=save)
    status = rig.engine.execute("run")
    assert status.lifecycle == "paused"
    assert observed == []
    assert rig.engine.records("run") == ()
    discarded = next(
        event for event in rig.events() if event["kind"] == "step_discarded"
    )
    assert discarded["run_files"] == ["runs/01-run_value.h5"]
    assert len(tuple(rig.paths.data_root.glob("iter/*/runs/[0-9]*.h5"))) == 1


def test_data_driven_early_stop_with_nan_slots_remains_completed(rig: Rig) -> None:
    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((3,), axes=(np.array([0.0, 1.0, 2.0]),))
        with run.schedule(buffer) as schedule:
            for _value, step in schedule.scan("sample", (0, 1, 2)):
                step.set_data(np.complex128(2.0))
                break
        assert np.isnan(buffer.array[1:]).all()
        return 7

    observed = start_effect(rig, acquire)
    assert rig.engine.execute("run").lifecycle == "done"
    assert isinstance(observed[0], Completed)
    assert observed[0].result == 7
    assert len(rig.engine.records("run")[0].run_files) == 1


def test_analysis_error_after_completed_effect_retains_file_but_never_commits(
    rig: Rig,
) -> None:
    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        outcome = yield from env.run(run_value, Cfg(), save=save_integer)
        assert isinstance(outcome, Completed)
        raise ValueError("analysis failed")

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "analysis failed" in str(status.reason)
    assert rig.engine.records("run") == ()
    failed = next(event for event in rig.events() if event["kind"] == "step_failed")
    assert failed["run_files"] == ["runs/01-run_value.h5"]
    assert isinstance(failed["error"], dict)
    assert failed["error"]["type"] == "builtins.ValueError"
    assert "raise ValueError" in str(failed["error"]["traceback"])
    assert len(tuple(rig.paths.data_root.glob("iter/*/runs/[0-9]*.h5"))) == 1


def test_device_setup_error_is_not_recoverable_schedule_failure(rig: Rig) -> None:
    settings = (DeviceSetup("flux", JsonParameters({"value": 0.1})),)
    observed: list[int] = []

    def setup(_signal: StopSignal) -> None:
        rig.engine.pause("run")
        raise OSError("setup failed")

    def acquire(_run: Run[Cfg]) -> int:
        observed.append(1)
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.run(acquire, Cfg(dev=settings), save=save_integer)
        return Next(1, state)

    rig.devices.on_setup = setup
    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "setup failed" in str(status.reason)
    assert observed == []
    assert "experiment_failed" not in rig.kinds()
    assert "step_discarded" not in rig.kinds()


@pytest.mark.parametrize("source", ["refresh", "ambiguous", "shape", "row"])
def test_invalid_live_binding_or_snapshot_fails_run_at_display_source(
    rig: Rig, source: str
) -> None:
    def acquire(run: Run[Cfg]) -> int:
        buffer = run.buffer((2,), axes=(np.array([0.0, 1.0]),))
        if source == "refresh":
            rig.plots.error = OSError("snapshot failed")
        if source == "ambiguous":
            run.buffer((2,), axes=(np.array([0.0, 1.0]),))
        buffer.set(np.array([1 + 0j, 2 + 0j]))
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        axes = env.axes("live")
        if source == "row":
            live = Live2DRow(
                axes.imshow(np.ones((2, 2))), 2, lambda values: values.real
            )
        else:
            (line,) = axes.plot([], [])
            live = Live1D(
                line,
                lambda values: values.real[:1] if source == "shape" else values.real,
            )
        yield from env.run(acquire, Cfg(), save=save_integer, live=live)
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "experiment_failed" not in rig.kinds()
    assert not tuple(rig.paths.data_root.glob("iter/*/runs/*.h5"))


def test_active_update_journal_failure_cancels_then_reports_original_cause(
    rig: Rig,
) -> None:
    def acquire(run: Run[Cfg]) -> int:
        journal = rig.paths.metadata_root / "journal.jsonl"
        journal.unlink()
        journal.mkdir()
        with pytest.raises(IsADirectoryError):
            rig.engine.update_tunables(
                "run",
                (TunableChange("value", 2),),
                expected_revision=0,
                actor=Actor("user", "test"),
            )
        assert run.cancel_signal.is_set()
        return 1

    observed = start_effect(rig, acquire)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "IsADirectoryError" in str(status.reason)
    assert "unusable" not in str(status.reason)
    assert status.revision == 0
    assert observed == []


def test_cancelled_generator_cleanup_failure_is_real_run_failure(rig: Rig) -> None:
    def acquire(_run: Run[Cfg]) -> int:
        rig.engine.pause("run")
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        try:
            yield from env.run(acquire, Cfg(), save=save_integer)
        finally:
            raise OSError("workflow cleanup failed")
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "workflow cleanup failed" in str(status.reason)
    assert "step_discarded" not in rig.kinds()
