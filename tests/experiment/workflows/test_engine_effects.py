"""Operation outcomes, cooperative cancellation and author-owned persistence."""

from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import TypeAdapter
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.v2.runtime import AcquireOptions, Schedule, SignalBuffer
from zcu_tools.experiment.workflows import (
    Actor,
    Completed,
    Failed,
    Live,
    OperationFacts,
    WorkflowEnv,
)
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2 import ProgramV2Cfg
from zcu_tools.program.v2.modules import SoftDelay

from tests._filesystem import DIRECTORY_OPEN_ERROR

from ._engine_fakes import Cfg, Plan, Point, Rig, Tunables, completed, paused, running


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path, native=True)
    yield rig
    rig.close()


def start_call(
    rig: Rig, operation: Callable[[CallContext], int]
) -> list[Completed[int] | Failed]:
    observed: list[Completed[int] | Failed] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        observed.append(await env.call(operation))

    rig.start(script)
    return observed


def test_full_device_cfg_reaches_port_with_operation_stop_signal(rig: Rig) -> None:
    cfg = FakeDeviceInfo(address="memory", value=0.5, rampstep=0.01, output="on")
    signals: list[StopSignal] = []
    rig.devices.on_set = lambda signal: signals.append(signal)

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        completed(await env.setup_device("flux", cfg))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert rig.devices.configs == [cfg]
    assert signals == rig.devices.signals
    assert len(signals) == 1 and not signals[0].is_set()


def test_cfg_at_return_is_detached_and_saver_receives_run_record(rig: Rig) -> None:
    original = Cfg()
    saved: list[object] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        cfg.value = 30
        cfg.nested.append(3)
        return 9

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        result = completed(await env.run(acquire, original))
        saved.append(result.value)
        assert result.value.cfg == Cfg(value=30, nested=[2, 3])
        path = env.new_commit_dir() / "measurement.txt"
        returned_cfg = result.value.cfg
        assert returned_cfg is not None
        path.write_text(f"{returned_cfg.value}:{result.value.result}")

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(saved) == 1
    assert original == Cfg()
    assert (
        rig.paths.data_root / "commits/000001/measurement.txt"
    ).read_text() == "30:9"


def test_schedule_acquire_failure_is_recoverable(rig: Rig) -> None:
    cause = OSError("acquire disconnected")

    def acquire(ctx: CallContext) -> int:
        buffer = SignalBuffer((1,))

        def fail(_step: object) -> None:
            raise cause

        with Schedule(Cfg(), buffer, stop=ctx.stop_signal) as schedule:
            schedule.batch({"acquire": fail})
        return 99

    observed = start_call(rig, acquire)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(observed) == 1
    assert isinstance(observed[0], Failed)
    assert "acquire disconnected" in observed[0].reason
    assert rig.engine.records("run") == ()
    facts = [
        TypeAdapter(OperationFacts).validate_python(event["operation"])
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert len(facts) == 1
    assert (facts[0].iteration, facts[0].attempt, facts[0].outcome) == (1, 1, "failed")
    assert facts[0].stopped is True
    assert facts[0].reason == observed[0].reason
    assert facts[0].device_snapshot is None


@pytest.mark.parametrize("source", ["outside", "other_signal"])
def test_arbitrary_exception_or_foreign_schedule_error_fails_run(
    rig: Rig, source: str
) -> None:
    def acquire(ctx: CallContext) -> int:
        cause = OSError("disconnected")
        if source == "outside":
            raise cause
        other = StopSignal()
        other.set_error("failed", "foreign failure", cause)
        other.raise_if_error()
        return 0

    observed = start_call(rig, acquire)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert observed == []
    reason = "disconnected" if source == "outside" else "foreign failure"
    assert reason in str(final.reason)
    facts = [
        TypeAdapter(OperationFacts).validate_python(event["operation"])
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert len(facts) == 1
    assert (facts[0].iteration, facts[0].attempt, facts[0].outcome) == (1, 1, "raised")
    assert facts[0].stopped is False
    assert reason in str(facts[0].reason)
    assert facts[0].device_snapshot is None


def test_schedule_interrupt_returns_partial_not_acquire_failure(rig: Rig) -> None:
    def acquire(ctx: CallContext) -> int:
        with Schedule(Cfg(), stop=ctx.stop_signal):
            raise KeyboardInterrupt
        return 7

    observed = start_call(rig, acquire)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    outcome = completed(observed[0])
    assert (outcome.value, outcome.partial_result) == (7, True)


def test_stop_keeps_partial_result_and_later_calls_see_stopped_tokens(rig: Rig) -> None:
    entered = Event()
    outcomes: list[Completed[int]] = []
    stops: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        def first(ctx: CallContext) -> int:
            entered.set()
            assert ctx.stop_signal.event.wait(10), "Stop reaches active operation"
            return 7

        outcomes.append(completed(await env.call(first)))

        def later(ctx: CallContext) -> int:
            stops.append(ctx.stop_signal.is_set())
            return 0

        outcomes.append(completed(await env.call(later)))

    rig.start(script)
    with running(rig) as future:
        assert entered.wait(10), "Worker reaches operation"
        rig.engine.stop("run")
        final = future.result(timeout=10)
    assert final.lifecycle == "cancelled"
    assert [(outcome.value, outcome.partial_result) for outcome in outcomes] == [
        (7, True),
        (0, True),
    ]
    assert stops == [True]
    assert [
        (outcome.facts.iteration, outcome.facts.attempt) for outcome in outcomes
    ] == [
        (1, 1),
        (1, 2),
    ]
    assert all(outcome.facts.stopped is outcome.partial_result for outcome in outcomes)
    assert all(outcome.facts.outcome == "partial" for outcome in outcomes)


def test_wait_pause_retries_original_deadline_without_restarting_author(
    rig: Rig,
) -> None:
    deadline = rig.clock.now() + timedelta(seconds=10)
    entered = Event()
    authors: list[int] = []
    outcomes: list[Completed[None]] = []

    def wait(signal: StopSignal) -> None:
        if len(rig.clock.targets) == 1:
            entered.set()
            assert signal.event.wait(10), "Pause stops first wait"

    rig.clock.on_wait = wait

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        authors.append(1)
        outcomes.append(completed(await env.wait_until(deadline)))

    rig.start(script)
    with running(rig) as future:
        assert entered.wait(10), "Clock announces first wait"
        rig.engine.pause("run")
        paused(rig)
        assert authors == [1]
        rig.engine.resume("run")
        final = future.result(timeout=10)
    assert final.lifecycle == "done", final.reason
    assert rig.clock.targets == [deadline, deadline]
    assert authors == [1]
    facts = [
        TypeAdapter(OperationFacts).validate_python(event["operation"])
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert [(op.iteration, op.attempt, op.stopped, op.outcome) for op in facts] == [
        (1, 1, True, "discarded"),
        (1, 2, False, "completed"),
    ]
    assert facts[0].reason == "Pause interrupted this attempt"
    assert outcomes[0].facts == facts[1]
    assert outcomes[0].partial_result is False


def test_setup_pause_retries_captured_full_cfg_with_fresh_operation_token(
    rig: Rig,
) -> None:
    entered = Event()
    cfg = FakeDeviceInfo(address="memory", value=0.25, rampstep=0.05, output="on")
    original = cfg.model_copy(deep=True)

    def setup(signal: StopSignal) -> None:
        if len(rig.devices.configs) == 1:
            entered.set()
            assert signal.event.wait(10), "Pause stops first setup"

    rig.devices.on_set = setup

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        completed(await env.setup_device("flux", cfg))

    rig.start(script)
    with running(rig) as future:
        assert entered.wait(10), "Port announces first setup"
        rig.engine.pause("run")
        paused(rig)
        cfg.value = 99
        rig.engine.resume("run")
        final = future.result(timeout=10)
    assert final.lifecycle == "done", final.reason
    assert rig.devices.configs == [original, original]
    assert rig.devices.signals[0].is_set()
    assert not rig.devices.signals[1].is_set()
    assert rig.devices.signals[0] is not rig.devices.signals[1]


def test_file_saved_before_pause_is_retained_and_author_continues(rig: Rig) -> None:
    entered = Event()
    saved: list[Path] = []
    calls: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        directory = env.new_commit_dir()
        path = directory / "measurement.txt"
        path.write_text("retained")
        saved.append(path)

        def acquire(ctx: CallContext) -> int:
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                assert ctx.stop_signal.event.wait(10), "Pause stops operation"
            return 7

        await env.call(acquire)
        await env.commit(Point(value=7), directory)

    rig.start(script)
    with running(rig) as future:
        assert entered.wait(10), "Worker reaches operation"
        rig.engine.pause("run")
        paused(rig)
        assert saved[0].read_text() == "retained"
        rig.engine.resume("run")
        assert future.result(timeout=10).lifecycle == "done"
    assert len(saved) == 1 and saved[0].read_text() == "retained"


@pytest.mark.parametrize("failure", ["saver", "analysis", "cleanup"])
def test_author_failure_retains_files_without_committing(
    rig: Rig, failure: str
) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        directory = env.new_commit_dir()
        path = directory / "measurement.txt"
        path.write_text("raw")
        completed(await env.call(lambda ctx: 7))
        if failure == "cleanup":
            try:
                rig.engine.stop("run")
            finally:
                raise OSError("cleanup failed")
        if failure == "saver":
            rig.engine.pause("run")
        raise OSError(f"{failure} failed")

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == f"OSError: {failure} failed"
    assert rig.engine.records("run") == ()
    assert (rig.paths.data_root / "commits/000001/measurement.txt").read_text() == "raw"


def test_data_driven_early_return_with_nan_slots_is_not_partial(rig: Rig) -> None:
    def acquire(ctx: CallContext) -> int:
        buffer = SignalBuffer((3,))
        with Schedule(Cfg(), buffer, stop=ctx.stop_signal) as schedule:
            for _value, step in schedule.scan("sample", (0, 1, 2)):
                step.set_data(np.complex128(2))
                break
        assert np.isnan(buffer.array[1:]).all()
        return 7

    observed = start_call(rig, acquire)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(observed) == 1
    outcome = completed(observed[0])
    assert (outcome.value, outcome.partial_result) == (7, False)


@pytest.mark.parametrize("source", ["build", "acquire", "conversion"])
def test_program_failures_keep_their_classification_and_do_not_repeat_build_or_conversion(
    rig: Rig,
    source: str,
) -> None:
    attempts: list[str] = []

    class Program:
        cfg_model = ProgramV2Cfg(rounds=1)

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            attempts.append("build")
            if source == "build":
                raise OSError("build failed")

        def acquire(self, *_args: object, **_kwargs: object) -> NDArray[np.complex128]:
            attempts.append("acquire")
            if source == "acquire":
                raise OSError("acquire failed")
            return np.array([1.0], dtype=np.complex128)

        def acquire_decimated(
            self, *_args: object, **_kwargs: object
        ) -> NDArray[np.complex128]:
            raise AssertionError("unexpected decimated acquire")

    def convert(raw: NDArray[np.complex128]) -> NDArray[np.complex128]:
        attempts.append("conversion")
        if source == "conversion":
            raise OSError("conversion failed")
        return raw

    def operation(ctx: CallContext) -> int:
        buffer = SignalBuffer((1,))
        with Schedule(Cfg(), buffer, stop=ctx.stop_signal) as schedule:
            builder = schedule.prog_builder(
                "soc", "soccfg", cfg=ProgramV2Cfg(), program_cls=Program
            )
            builder.add(SoftDelay("wait", 0.0)).build_and_acquire(
                raw2signal_fn=convert, options=AcquireOptions(retry=1)
            )
        return 99

    observed = start_call(rig, operation)
    final = rig.engine.execute("run")
    if source == "acquire":
        assert final.lifecycle == "done", final.reason
        assert isinstance(observed[0], Failed)
        assert "acquire failed" in observed[0].reason
        assert attempts == ["build", "acquire", "acquire"]
    else:
        assert final.lifecycle == "failed"
        assert final.reason == f"OSError: {source} failed"
        assert observed == []
        assert attempts == (
            ["build"] if source == "build" else ["build", "acquire", "conversion"]
        )


@pytest.mark.parametrize("first_round_stop", [False, True])
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_program_stop_keeps_partial_delivery_without_failed_outcome(
    rig: Rig,
    first_round_stop: bool,
    mode: str,
) -> None:
    class Program:
        cfg_model = ProgramV2Cfg(rounds=1)

        def acquire(self, *_args: object, **_kwargs: object) -> NDArray[np.complex128]:
            rig.engine.stop("run")
            if first_round_stop:
                raise StoppedPartialAcquireError("stopped before first completed round")
            return np.array([1.0], dtype=np.complex128)

        def acquire_decimated(
            self, *_args: object, **_kwargs: object
        ) -> NDArray[np.complex128]:
            return self.acquire()

    def operation(ctx: CallContext) -> int:
        buffer = SignalBuffer((1,))
        with Schedule(Cfg(), buffer, stop=ctx.stop_signal) as schedule:
            builder = schedule.prog_builder("soc", "soccfg", cfg=ProgramV2Cfg())
            if mode == "integrated":
                builder.run_program(Program(), raw2signal_fn=lambda raw: raw)
            else:
                builder.run_program_decimated(Program(), raw2signal_fn=lambda raw: raw)
        return 99

    observed = start_call(rig, operation)
    final = rig.engine.execute("run")
    assert final.lifecycle == "cancelled", final.reason
    assert len(observed) == 1
    outcome = completed(observed[0])
    assert (outcome.value, outcome.partial_result) == (99, True)


def test_active_tunables_journal_failure_stops_operation_and_retains_original_cause(
    rig: Rig,
) -> None:
    causes: list[OSError] = []
    stopped: list[bool] = []
    delivered: list[object] = []

    def operation(ctx: CallContext) -> int:
        journal = rig.paths.metadata_root / "journal.jsonl"
        contents = journal.read_bytes()
        journal.unlink()
        journal.mkdir()
        try:
            with pytest.raises(DIRECTORY_OPEN_ERROR) as caught:
                rig.engine.update_tunables(
                    "run",
                    Tunables(value=2),
                    expected_revision=0,
                    actor=Actor("user", "test"),
                )
            causes.append(caught.value)
        finally:
            journal.rmdir()
            journal.write_bytes(contents)
        stopped.append(ctx.stop_signal.is_set())
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        delivered.append(await env.call(operation))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert len(causes) == 1
    assert final.reason == f"{type(causes[0]).__name__}: {causes[0]}"
    assert final.revision == 0
    assert stopped == [True]
    assert delivered == []


def test_device_operation_error_is_not_a_recoverable_schedule_failure(rig: Rig) -> None:
    def operate(signal: StopSignal) -> None:
        raise OSError("Device 'flux' failed")

    rig.devices.on_set = operate
    observed: list[object] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        observed.append(
            await env.setup_device("flux", FakeDeviceInfo(address="memory", value=0.1))
        )

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == "OSError: Device 'flux' failed"
    assert observed == []
