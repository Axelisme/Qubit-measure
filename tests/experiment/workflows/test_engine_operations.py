"""Cooperative async operation, Pause and request-bound Ask contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event, Thread, Timer

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.metadata import SoftwareProvenance
from zcu_tools.experiment import workflows as w
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import saved_run
from zcu_tools.plotting.plots import NonPresentingHost

from ._engine_ports import ManualClock, RecordingBar, RecordingDevices


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Tunables(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = 3


class Summary(BaseModel):
    value: int


class Reply(Answer):
    approved: bool


class Approval(Question[Reply]):
    label: str


class Cfg(ExpCfgModel):
    value: int = 3


type Rig = tuple[w.Engine[int], w.RunPaths, RecordingDevices]


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    plots = w.WorkflowPlots(NonPresentingHost())
    devices = RecordingDevices()
    board = white_noise()
    engine = w.Engine(
        w.EnginePorts(
            plots,
            RecordingBar,
            ManualClock(),
            context=7,
            devices=devices,
            soc=board.soc,
            soccfg=board.soccfg,
        )
    )
    paths = w.RunPaths(tmp_path / "metadata", tmp_path / "data")
    yield engine, paths, devices
    plots.finish()
    plots.release()


def start(
    engine: w.Engine[int],
    paths: w.RunPaths,
    script: w.WorkflowScript[Plan, Tunables, Summary | None, int],
) -> None:
    engine.start(
        script,
        plan=Plan(),
        tunables=Tunables(),
        paths=paths,
        identity=w.RunIdentity(
            "run",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="offline",
            ),
        ),
    )


def worker(engine: w.Engine[int], ready: Event) -> tuple[Thread, list[w.RunStatus]]:
    statuses: list[w.RunStatus] = []

    def execute() -> None:
        try:
            statuses.append(engine.execute("run"))
        finally:
            ready.set()

    thread = Thread(target=execute)
    thread.start()
    return thread, statuses


def until(predicate: Callable[[], bool], label: str) -> None:
    # Only a public terminal/pending condition succeeds; this timer is a deadlock watchdog.
    expired = Event()
    timer = Timer(30, expired.set)
    timer.start()
    try:
        while not predicate():
            assert not expired.is_set(), f"Timed out awaiting {label}"
            expired.wait(0.001)
    finally:
        timer.cancel()
        timer.join()


def test_stop_delivers_partial_then_starts_stopped_operations_and_skips_ask(
    rig: Rig,
) -> None:
    engine, paths, _devices = rig
    ready, entered = Event(), Event()
    outcomes: list[tuple[int, bool]] = []
    answers: list[Reply | None] = []
    later_stops: list[bool] = []

    @w.workflow("stop", "Preserve cooperative partial results")
    async def script(
        env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Tunables]
    ) -> Summary:
        def first(ctx: CallContext) -> int:
            entered.set()
            ready.set()
            assert ctx.stop_signal.event.wait(30), (
                "Stop must reach the active operation"
            )
            return 4

        first_result = await env.call(first)
        assert isinstance(first_result, w.Completed)
        outcomes.append((first_result.value, first_result.partial_result))

        def later(ctx: CallContext) -> int:
            later_stops.append(ctx.stop_signal.is_set())
            return 0

        second_result = await env.call(later)
        assert isinstance(second_result, w.Completed)
        outcomes.append((second_result.value, second_result.partial_result))
        answers.append(await env.ask(Approval(label="after stop")))
        return Summary(value=4)

    start(engine, paths, script)
    thread, statuses = worker(engine, ready)
    try:
        assert ready.wait(30), "The worker must enter its operation or settle"
        assert entered.is_set(), statuses
        engine.stop("run")
    finally:
        engine.stop("run")
        thread.join(30)
    assert not thread.is_alive()
    assert statuses[-1].lifecycle == "cancelled"
    assert outcomes == [(4, True), (0, True)]
    assert later_stops == [True]
    assert answers == [None]
    assert saved_run.read_summary(paths.metadata_root) is None


def test_pause_retries_an_interrupted_call_without_returning_partial_to_author(
    rig: Rig,
) -> None:
    engine, paths, _devices = rig
    ready, entered = Event(), Event()
    calls: list[int] = []
    delivered: list[int] = []

    @w.workflow("pause", "Resume the same suspended request")
    async def script(
        env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Tunables]
    ) -> None:
        def acquire(ctx: CallContext) -> int:
            calls.append(ctx.epoch)
            if len(calls) == 1:
                entered.set()
                ready.set()
                assert ctx.stop_signal.event.wait(30), (
                    "Pause must reach the active operation"
                )
                return -1
            assert not ctx.stop_signal.is_set()
            return 42

        result = await env.call(acquire)
        assert isinstance(result, w.Completed)
        delivered.append(result.value)
        assert not result.partial_result

    start(engine, paths, script)
    thread, statuses = worker(engine, ready)
    try:
        assert ready.wait(30), "The worker must enter its operation or settle"
        assert entered.is_set(), statuses
        engine.pause("run")
        until(
            lambda: (
                (status := engine.status()) is not None and status.lifecycle == "paused"
            ),
            "paused operation",
        )
        assert delivered == []
        engine.resume("run")
        thread.join(30)
    finally:
        if thread.is_alive():
            engine.stop("run")
            thread.join(30)
    assert not thread.is_alive()
    assert delivered == [42]
    assert calls == [0, 1]
    assert statuses[-1].lifecycle == "done"


def test_answers_are_bound_to_the_original_request_id(rig: Rig) -> None:
    engine, paths, _devices = rig
    ready = Event()
    answers: list[Reply | None] = []

    @w.workflow("questions", "Require an explicit answer for each question")
    async def script(
        env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Tunables]
    ) -> None:
        answers.append(await env.ask(Approval(label="first")))
        answers.append(await env.ask(Approval(label="second")))

    start(engine, paths, script)
    thread, statuses = worker(engine, ready)
    try:
        until(
            lambda: engine.pending_question("run") is not None or bool(statuses),
            "first question",
        )
        first = engine.pending_question("run")
        assert first is not None, statuses
        before = engine.status()
        with pytest.raises(w.QuestionMismatch) as rejected:
            engine.answer("run", Reply(approved=True), request_id="stale")
        assert rejected.value.pending_request_id == first.request_id
        assert engine.status() == before
        with pytest.raises(w.QuestionMismatch) as stale_parse:
            engine.parse_answer("run", {"approved": "invalid"}, request_id="stale")
        assert stale_parse.value.pending_request_id == first.request_id
        parsed = engine.parse_answer(
            "run", {"approved": True}, request_id=first.request_id
        )
        assert parsed == Reply(approved=True)
        assert engine.status() == before
        engine.answer("run", Reply(approved=True), request_id=first.request_id)
        until(
            lambda: (
                (pending := engine.pending_question("run")) is not None
                and pending.request_id != first.request_id
            ),
            "second question",
        )
        second = engine.pending_question("run")
        assert second is not None
        with pytest.raises(w.QuestionMismatch) as duplicate:
            engine.answer("run", Reply(approved=False), request_id=first.request_id)
        assert duplicate.value.pending_request_id == second.request_id
        assert engine.pending_question("run") == second
        engine.answer("run", Reply(approved=False), request_id=second.request_id)
        thread.join(30)
    finally:
        if thread.is_alive():
            engine.stop("run")
            thread.join(30)
    assert not thread.is_alive()
    assert statuses[-1].lifecycle == "done"
    assert answers == [Reply(approved=True), Reply(approved=False)]
    with pytest.raises(w.QuestionMismatch) as terminal:
        engine.answer("run", Reply(approved=True), request_id=second.request_id)
    assert terminal.value.pending_request_id is None


def test_recoverable_acquire_failure_does_not_poison_the_next_operation(
    rig: Rig,
) -> None:
    engine, paths, _devices = rig
    outcomes: list[w.Completed[RunRecord[Cfg, int]] | w.Failed] = []
    tokens: list[bool] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        tokens.append(context.cancel_signal.is_set())
        if len(tokens) == 1:
            cause = OSError("first acquire failed")
            context.cancel_signal.set_error("failed", str(cause), cause)
        return 42

    @w.workflow("recover", "A recoverable acquire failure permits an authored retry")
    async def script(
        env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Tunables]
    ) -> None:
        outcomes.append(await env.run(acquire, Cfg()))
        outcomes.append(await env.run(acquire, Cfg()))

    start(engine, paths, script)
    final = engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert isinstance(outcomes[0], w.Failed)
    assert outcomes[0].reason == "first acquire failed"
    assert isinstance(outcomes[1], w.Completed), (
        "A new operation must not inherit handled acquire failure"
    )
    assert tokens == [False, False]


def test_run_sugar_captures_devices_and_returns_the_modified_cfg_record(
    rig: Rig,
) -> None:
    engine, paths, devices = rig
    records: list[tuple[int, int]] = []
    cfg = Cfg(value=3)

    @w.workflow("experiment", "Return a native RunRecord from async run")
    async def script(
        env: w.WorkflowEnv[int], plan: Plan, tun: w.Live[Tunables]
    ) -> None:
        def experiment(cfg: Cfg, *, context: RunContext) -> int:
            assert context.plots is not env.plots
            assert not context.cancel_signal.is_set()
            cfg.value = 8
            return cfg.value + 1

        outcome = await env.run(experiment, cfg)
        assert isinstance(outcome, w.Completed)
        assert outcome.value.cfg is not None
        records.append((outcome.value.cfg.value, outcome.value.result))

    start(engine, paths, script)
    final = engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert records == [(8, 9)]
    assert cfg.value == 3
    assert devices.captures == 1
