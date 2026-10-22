"""Public async Engine lifecycle, isolation and journal-authoritative commits."""

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest
from pydantic import ValidationError
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import (
    Actor,
    Engine,
    EnginePorts,
    InvalidRunState,
    Live,
    RevisionConflict,
    RunMismatch,
    RunPaths,
    WorkflowEnv,
    workflow,
)

from tests._filesystem import DIRECTORY_OPEN_ERROR

from ._engine_fakes import (
    Plan,
    Point,
    RecordingBar,
    Rig,
    Tunables,
    completed,
    identity,
    paused,
    running,
)


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


async def points(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
    iterations = env.iterations()
    try:
        iteration = next(iterations)
        for _ in range(plan.count):
            iteration.commit(Point(value=tun.current.value))
            iteration = next(iterations)
    finally:
        iterations.close()


def test_start_is_deferred_and_commits_precede_done(rig: Rig) -> None:
    entered: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        entered.append(env.context)
        await points(env, plan, tun)

    assert rig.engine.status() is None
    rig.start(script)
    assert entered == []
    assert rig.kinds() == ("run_started",)
    initial = rig.engine.status()
    assert initial is not None
    assert (initial.lifecycle, initial.call_seq, initial.committed_seq) == (
        "running",
        None,
        0,
    )
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert entered == [7]
    assert final.committed_seq == 2
    assert rig.kinds() == (
        "run_started",
        "record_committed",
        "record_committed",
        "run_ended",
    )
    assert [r.commit_seq for r in rig.engine.records("run", since=1)] == [2]
    assert rig.engine.records("run", since=2) == ()
    manifest = json.loads((rig.paths.metadata_root / "manifest.json").read_text())
    assert manifest["lifecycle"] == "done"
    assert manifest["ended_at"] is not None
    assert rig.engine.stop("run") == final


def test_nested_inputs_and_published_records_are_detached(rig: Rig) -> None:
    caller_plan = Plan()
    observed: list[list[int]] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            observed.append(plan.values.copy())
            plan.values.append(99)
            record = Point(value=tun.current.value)
            iteration.commit(record)
            iteration = next(iterations)
            record.value = 100
        finally:
            iterations.close()

    rig.start(script, plan=caller_plan)
    caller_plan.values.append(42)
    assert rig.engine.execute("run").lifecycle == "done"
    assert observed == [[10]]
    records = rig.engine.records("run")
    assert records[0].encoded_record.value == {"value": 1}
    value = records[0].encoded_record.value
    assert isinstance(value, dict)
    value["value"] = 9
    assert rig.engine.records("run")[0].encoded_record.value == {"value": 1}


def test_resume_keeps_commits_progress_and_author_locals(rig: Rig) -> None:
    entered = Event()
    calls: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            bar = env.pbar("points", plan.count)
            iteration.commit(Point(value=tun.current.value))
            iteration = next(iterations)
            bar.set_progress(1)

            def operation(ctx: CallContext) -> int:
                calls.append(1)
                if len(calls) == 1:
                    entered.set()
                    assert ctx.scope.stop.event.wait(10), (
                        "Pause reaches active operation"
                    )
                return 7

            value = completed(await iteration.call(operation)).value
            assert env.pbar("points", plan.count) is bar
            iteration.commit(Point(value=value + tun.current.value))
            iteration = next(iterations)
            bar.set_progress(2)
        finally:
            iterations.close()

    rig.start(script)
    with running(rig) as future:
        assert entered.wait(10), "The worker reaches the operation"
        rig.engine.pause("run")
        paused(rig)
        records = rig.engine.records("run")
        assert len(records) == 1
        assert not future.done()
        assert len(rig.bars) == 1 and not rig.bars[0].closed
        rig.engine.update_tunables(
            "run", Tunables(value=3), expected_revision=0, actor=Actor("user", "test")
        )
        rig.engine.resume("run")
        final = future.result(timeout=10)
    assert final.lifecycle == "done", final.reason
    assert calls == [1, 1]
    assert [r.encoded_record.value for r in rig.engine.records("run")] == [
        {"value": 1},
        {"value": 10},
    ]
    assert final.progress[0].completed == 2
    assert rig.bars[0].closed


def test_thread_safe_requests_do_not_wait_and_stop_upgrades_pause(rig: Rig) -> None:
    entered, release = Event(), Event()
    stops: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)

            def operation(ctx: CallContext) -> int:
                entered.set()
                assert release.wait(10), "Controller releases the operation in finally"
                stops.append(ctx.scope.stop.is_set())
                return 4

            result = completed(await iteration.call(operation))
            assert result.partial_result
        finally:
            iterations.close()

    rig.start(script)
    with running(rig) as future:
        try:
            assert entered.wait(10), "Worker enters the injected operation"
            assert rig.engine.pause("run").lifecycle == "pausing"
            assert rig.engine.pause("run").lifecycle == "pausing"
            assert not future.done()
            with pytest.raises(InvalidRunState):
                rig.engine.execute("run")
            with pytest.raises(InvalidRunState):
                rig.engine.resume("run")
            assert rig.engine.stop("run").lifecycle == "stopping"
            assert rig.engine.stop("run").lifecycle == "stopping"
            with pytest.raises(InvalidRunState):
                rig.engine.pause("run")
            with pytest.raises(InvalidRunState):
                rig.engine.update_tunables(
                    "run",
                    Tunables(value=2),
                    expected_revision=0,
                    actor=Actor("user", "test"),
                )
        finally:
            release.set()
        assert future.result(timeout=10).lifecycle == "cancelled"
    assert stops == [True]
    assert rig.kinds().count("pause_requested") == 1
    assert rig.kinds().count("stop_requested") == 1


def test_bad_revision_or_values_do_not_publish(rig: Rig) -> None:
    rig.start(points)
    with pytest.raises(RevisionConflict) as caught:
        rig.engine.update_tunables(
            "run", Tunables(value=2), expected_revision=1, actor=Actor("user", "test")
        )
    assert (caught.value.expected, caught.value.actual) == (1, 0)
    with pytest.raises(ValidationError):
        rig.engine.update_tunables(
            "run",
            Tunables.model_construct(value=-1),
            expected_revision=0,
            actor=Actor("user", "test"),
        )
    assert rig.engine.tunables("run").revision == 0
    assert rig.kinds() == ("run_started",)
    assert rig.engine.execute("run").lifecycle == "done"


def test_wrong_run_invalid_lifecycle_and_terminal_replacement(rig: Rig) -> None:
    with pytest.raises(RunMismatch) as caught:
        rig.engine.execute("missing")
    assert (caught.value.expected, caught.value.actual) == ("missing", None)
    rig.start(points)
    with pytest.raises(RunMismatch):
        rig.engine.stop("other")
    with pytest.raises(InvalidRunState):
        rig.start(points)
    with pytest.raises(InvalidRunState):
        rig.engine.resume("run")
    old = rig.engine.execute("run")
    with pytest.raises(InvalidRunState):
        rig.engine.execute("run")
    with pytest.raises(InvalidRunState):
        rig.engine.pause("run")
    with pytest.raises(ValueError, match="since"):
        rig.engine.records("run", since=-1)

    async def replacement(
        env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]
    ) -> None:
        await points(env, plan, tun)

    rig.engine.start(
        workflow("replacement", "New run")(replacement),
        plan=Plan(count=1),
        tunables=Tunables(),
        paths=RunPaths(rig.root / "new-meta", rig.root / "new-data"),
        identity=identity("new"),
    )
    with pytest.raises(RunMismatch):
        rig.engine.records("run")
    assert rig.engine.execute("new").committed_seq == 1
    assert old.run_id == "run" and old.committed_seq == 2


def test_start_revalidates_plan_before_creating_paths(rig: Rig) -> None:
    with pytest.raises(ValidationError):
        rig.start(points, plan=Plan.model_construct(count=-1, values=[]))
    assert not rig.paths.data_root.exists()


@pytest.mark.parametrize("port", ["soc", "devices", "context"])
def test_missing_port_fails_only_when_used_during_execute(rig: Rig, port: str) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        if port == "soc":
            _ = env.soc
        elif port == "devices":
            _ = env.devices
        else:
            _ = env.context

    rig.engine = Engine(EnginePorts(rig.plots, rig.progress, rig.clock))
    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert f"Missing workflow capability: {port}" in str(final.reason)
    assert rig.engine.records("run") == ()


def test_author_startup_failure_never_commits(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        raise OSError("author startup failed")

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == "OSError: author startup failed"
    assert rig.engine.records("run") == ()
    assert rig.kinds() == ("run_started", "run_ended")


@pytest.mark.parametrize("producer_fails", [False, True])
def test_progress_snapshot_error_does_not_replace_producer_cause_or_retain_worker_reservation(
    rig: Rig,
    producer_fails: bool,
) -> None:
    class ReadCallbackBar(RecordingBar):
        fail_next_read = False

        @property
        def n(self) -> int | float:
            if self.fail_next_read:
                self.fail_next_read = False
                raise LookupError("capture failed")
            return super().n

    bar = ReadCallbackBar(total=2, desc="work")

    def progress(**kwargs: object) -> ReadCallbackBar:
        return bar

    rig.engine = Engine(
        EnginePorts(rig.plots, progress, rig.clock, context=7, devices=rig.devices)
    )

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        env.pbar("work", 2).set_progress(1)
        try:
            if producer_fails:
                raise RuntimeError("producer failed")
        finally:
            bar.fail_next_read = True

    rig.start(script)
    with pytest.raises(LookupError, match="capture failed"):
        rig.engine.execute("run")
    status = rig.engine.status()
    assert status is not None
    assert status.lifecycle == ("failed" if producer_fails else "done")
    assert status.reason == (
        "RuntimeError: producer failed" if producer_fails else None
    )
    assert status.progress[0].completed == 1
    assert bar.closed
    with pytest.raises(InvalidRunState):
        rig.engine.execute("run")


def test_commit_journal_failure_does_not_publish_record(rig: Rig) -> None:
    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        iterations = env.iterations()
        try:
            iteration = next(iterations)
            journal = rig.paths.metadata_root / "journal.jsonl"
            journal.unlink()
            journal.mkdir()
            iteration.commit(Point(value=1))
            iteration = next(iterations)
        finally:
            iterations.close()

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert DIRECTORY_OPEN_ERROR.__name__ in str(final.reason)
    assert final.committed_seq == 0
    assert rig.engine.records("run") == ()
