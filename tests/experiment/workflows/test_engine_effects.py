"""Operation outcomes, cooperative cancellation and author-owned persistence."""

from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import TypeAdapter
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.run_core import run_core
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
from zcu_tools.program.v2.modules import SoftDelay

from tests._filesystem import DIRECTORY_OPEN_ERROR

from ._engine_fakes import Cfg, Plan, Rig, Tunables, completed, running


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
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            observed.append(await iteration.call(operation))
        finally:
            iterations.close()

    rig.start(script)
    return observed


def test_full_device_cfg_reaches_port_with_operation_stop_signal(rig: Rig) -> None:
    cfg = FakeDeviceInfo(address="memory", value=0.5, rampstep=0.01, output="on")
    signals: list[StopSignal] = []
    rig.devices.on_set = lambda signal: signals.append(signal)

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            completed(await iteration.setup_device("flux", cfg))
        finally:
            iterations.close()

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
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            result = completed(await run_core(iteration, acquire, original))
            saved.append(result.value)
            assert result.value.cfg == Cfg(value=30, nested=[2, 3])
            path = iteration.dir / "measurement.txt"
            returned_cfg = result.value.cfg
            assert returned_cfg is not None
            path.write_text(f"{returned_cfg.value}:{result.value.result}")
        finally:
            iterations.close()

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(saved) == 1
    assert original == Cfg()
    assert (
        rig.paths.data_root / "iteration-000001/measurement.txt"
    ).read_text() == "30:9"


def test_call_delivers_the_original_experiment_exception_to_the_author(
    rig: Rig,
) -> None:
    cause = OSError("acquire disconnected")

    def acquire(ctx: CallContext) -> int:
        buffer = SignalBuffer((1,))

        with Schedule( buffer, scope=ctx.scope):
            raise cause
        return 99

    caught: list[OSError] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            await env.call(acquire)
        except OSError as error:
            caught.append(error)
        assert (await env.call(lambda ctx: 99)).value == 99

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert caught == [cause]
    assert caught[0] is cause
    assert rig.engine.records("run") == ()
    facts = [
        TypeAdapter(OperationFacts).validate_python(event["operation"])
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert [fact.outcome for fact in facts] == ["raised", "completed"]
    assert (facts[0].iteration, facts[0].attempt) == (0, 1)
    assert facts[0].stopped is False
    assert facts[0].reason == "OSError: acquire disconnected"


@pytest.mark.parametrize(
    "failed", [False, True], ids=["schedule_cancel", "acquire_failure"]
)
def test_nested_scope_completion_preserves_outcome_and_next_operation_is_clean(
    rig: Rig,
    failed: bool,
) -> None:
    observed: list[Completed[int] | Failed] = []
    attempts: list[int] = []
    cause = OSError("readout disconnected")

    class Program:
        def acquire(self, *_args: object, **_kwargs: object) -> NDArray[np.complex128]:
            attempts.append(1)
            raise cause

        acquire_decimated = acquire

    def operation(ctx: CallContext) -> int:
        with Schedule( SignalBuffer(1), scope=ctx.scope) as schedule:
            assert schedule.stop is not ctx.scope.stop, "Schedule must own a child"
            if failed:
                schedule.prog_builder("soc", "soccfg").run_program(
                    Program(),
                    raw2signal_fn=lambda raw: raw,
                    options=AcquireOptions(retry=1),
                )
            else:
                schedule.set_stop()
            assert not ctx.scope.stop.is_set()
        return 99

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            if failed:
                with pytest.raises(OSError, match="readout disconnected") as caught:
                    await iteration.call(operation)
                assert caught.value is cause
            else:
                observed.append(await iteration.call(operation))
            observed.append(
                await iteration.call(lambda ctx: 0 if ctx.scope.cancelled else 7)
            )
        finally:
            iterations.close()

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    if failed:
        assert len(observed) == 1
        assert attempts == [1, 1]
    else:
        first = completed(observed[0])
        assert (first.value, first.facts.stopped) == (99, False)
        assert first.facts.stopped is False
        assert first.facts.outcome == "completed"
    second = completed(observed[-1])
    assert (second.value, second.facts.stopped) == (7, False)
    assert second.facts.stopped is False
    assert second.facts.outcome == "completed"
    facts = [
        TypeAdapter(OperationFacts).validate_python(event["operation"])
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert [
        (fact.iteration, fact.attempt, fact.outcome, fact.stopped) for fact in facts
    ] == [
        (1, 1, "raised" if failed else "completed", False),
        (1, 2, "completed", False),
    ]
    if not failed:
        assert facts[0] == completed(observed[0]).facts
    assert facts[1] == second.facts


def test_unhandled_exception_fails_run(rig: Rig) -> None:
    def acquire(ctx: CallContext) -> int:
        cause = OSError("disconnected")
        raise cause

    observed = start_call(rig, acquire)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert observed == []
    reason = "disconnected"
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


def test_schedule_interrupt_reaches_the_author_unchanged(rig: Rig) -> None:
    def acquire(ctx: CallContext) -> int:
        with Schedule( scope=ctx.scope):
            raise KeyboardInterrupt
        return 7

    caught: list[KeyboardInterrupt] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            await env.call(acquire)
        except KeyboardInterrupt as error:
            caught.append(error)

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(caught) == 1


def test_stop_preserves_completion_and_later_calls_see_stopped_tokens(rig: Rig) -> None:
    entered = Event()
    outcomes: list[Completed[int]] = []
    stops: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)

            def first(ctx: CallContext) -> int:
                entered.set()
                assert ctx.scope.stop.event.wait(10), "Stop reaches active operation"
                return 7

            outcomes.append(completed(await iteration.call(first)))

            def later(ctx: CallContext) -> int:
                stops.append(ctx.scope.stop.is_set())
                return 0

            outcomes.append(completed(await iteration.call(later)))
        finally:
            iterations.close()

    rig.start(script)
    with running(rig) as future:
        assert entered.wait(10), "Worker reaches operation"
        rig.engine.cancel("run", scope="iteration")
        final = future.result(timeout=10)
    assert final.lifecycle == "done"
    assert [(outcome.value, outcome.facts.stopped) for outcome in outcomes] == [
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

    assert all(outcome.facts.outcome == "completed" for outcome in outcomes)
    journal_facts = [
        TypeAdapter(OperationFacts).validate_python(event["operation"])
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert journal_facts == [outcome.facts for outcome in outcomes]


@pytest.mark.parametrize("failure", ["saver", "analysis", "cleanup"])
def test_author_failure_retains_files_without_committing(
    rig: Rig, failure: str
) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            directory = iteration.dir
            path = directory / "measurement.txt"
            path.write_text("raw")
            completed(await iteration.call(lambda ctx: 7))
            if failure == "cleanup":
                try:
                    rig.engine.stop("run")
                finally:
                    raise OSError("cleanup failed")
            raise OSError(f"{failure} failed")
        finally:
            iterations.close()

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == f"OSError: {failure} failed"
    assert rig.engine.records("run") == ()
    assert (
        rig.paths.data_root / "iteration-000001/measurement.txt"
    ).read_text() == "raw"


def test_data_driven_early_return_with_nan_slots_is_not_partial(rig: Rig) -> None:
    def acquire(ctx: CallContext) -> int:
        buffer = SignalBuffer((3,))
        with Schedule( buffer, scope=ctx.scope) as schedule:
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
    assert (outcome.value, outcome.facts.stopped) == (7, False)


@pytest.mark.parametrize("source", ["build", "acquire", "conversion"])
def test_program_failures_keep_their_classification_and_do_not_repeat_build_or_conversion(
    rig: Rig,
    source: str,
) -> None:
    attempts: list[str] = []

    class Program:
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
        with Schedule( buffer, scope=ctx.scope) as schedule:
            builder = schedule.prog_builder("soc", "soccfg", program_cls=Program)
            builder.add(SoftDelay("wait", 0.0)).build_and_acquire(
                raw2signal_fn=convert, options=AcquireOptions(retry=1)
            )
        return 99

    observed = start_call(rig, operation)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == f"OSError: {source} failed"
    assert observed == []
    assert (
        attempts
        == {
            "build": ["build"],
            "acquire": ["build", "acquire", "acquire"],
            "conversion": ["build", "acquire", "conversion"],
        }[source]
    )


@pytest.mark.parametrize("first_round_stop", [False, True])
@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_program_stop_keeps_partial_delivery_without_failed_outcome(
    rig: Rig,
    first_round_stop: bool,
    mode: str,
) -> None:
    class Program:
        def acquire(self, *_args: object, **_kwargs: object) -> NDArray[np.complex128]:
            rig.engine.cancel("run", scope="iteration")
            if first_round_stop:
                raise StoppedPartialAcquireError("stopped before first completed round")
            return np.array([1.0], dtype=np.complex128)

        def acquire_decimated(
            self, *_args: object, **_kwargs: object
        ) -> NDArray[np.complex128]:
            return self.acquire()

    def operation(ctx: CallContext) -> int:
        buffer = SignalBuffer((1,))
        with Schedule( buffer, scope=ctx.scope) as schedule:
            builder = schedule.prog_builder("soc", "soccfg")
            if mode == "integrated":
                builder.run_program(Program(), raw2signal_fn=lambda raw: raw)
            else:
                builder.run_program_decimated(Program(), raw2signal_fn=lambda raw: raw)
        return 99

    observed = start_call(rig, operation)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert len(observed) == 1
    outcome = completed(observed[0])
    assert (outcome.value, outcome.facts.stopped) == (99, True)


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
        stopped.append(ctx.scope.stop.is_set())
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            delivered.append(await iteration.call(operation))
        finally:
            iterations.close()

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
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            observed.append(
                await iteration.setup_device(
                    "flux", FakeDeviceInfo(address="memory", value=0.1)
                )
            )
        finally:
            iterations.close()

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == "OSError: Device 'flux' failed"
    assert observed == []
