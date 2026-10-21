"""Public Engine question suspension, worker delivery and cancellation contracts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Event, get_ident

import pytest
from pydantic import ValidationError
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InvalidRunState,
    Next,
    Run,
    RunMismatch,
    Step,
    WorkflowEnv,
)

from tests._filesystem import DIRECTORY_OPEN_ERROR

from ._engine_fakes import START, Cfg, Plan, Rig, State, Tunables


class Decision(Answer):
    """The synthetic operator's chosen point value."""

    value: int


class OtherAnswer(Answer):
    """An unrelated model, even though its fields match Decision."""

    value: int


class Choose(Question[Decision]):
    """Evidence identifying the point awaiting a decision."""

    point: int


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_pending_answer_type_is_readonly_and_absent_outside_ask(rig: Rig) -> None:
    received: list[Decision | None] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        received.append((yield from env.ask(Choose(point=4))))
        return Done()

    rig.start(step)
    assert rig.engine.pending_answer_type("run") is None
    rig.engine.execute("run")
    before = rig.kinds()
    answer_type = rig.engine.pending_answer_type("run")
    assert answer_type is Decision
    assert answer_type.model_json_schema()["properties"]["value"]["type"] == "integer"
    assert received == []
    assert rig.kinds() == before
    with pytest.raises(RunMismatch):
        rig.engine.pending_answer_type("stale")
    rig.engine.answer("run", Decision(value=7))
    assert rig.engine.pending_answer_type("run") is None


def test_parse_answer_validates_without_resuming_and_preserves_retry(rig: Rig) -> None:
    received: list[Decision | None] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        received.append((yield from env.ask(Choose(point=4))))
        return Done()

    rig.start(step)
    with pytest.raises(RuntimeError, match="pending"):
        rig.engine.parse_answer("run", {"value": 7})
    rig.engine.execute("run")
    before = rig.kinds()
    with pytest.raises(ValidationError, match="Choose"):
        rig.engine.parse_answer("run", {"value": "not an integer"})
    with pytest.raises(RunMismatch):
        rig.engine.parse_answer("stale", {"value": 7})
    answer = rig.engine.parse_answer("run", {"value": 7})
    assert answer == Decision(value=7)
    assert received == []
    status = rig.engine.status()
    assert status is not None and status.lifecycle == "paused"
    assert rig.kinds() == before
    rig.engine.answer("run", answer)
    assert received == [Decision(value=7)]
    with pytest.raises(RuntimeError, match="pending"):
        rig.engine.parse_answer("run", {"value": 8})


def test_answer_continues_same_invocation_with_original_capture(rig: Rig) -> None:
    entered: list[tuple[int, Path]] = []
    observed: list[tuple[int, int]] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index == 1:
            return Done()
        entered.append((tun.value, env.iteration_dir))
        yield from env.wait_until(START)
        decision = yield from env.ask(Choose(point=state.index))
        assert decision is not None
        observed.append((decision.value, tun.value))
        yield from env.wait_until(START)
        state.index += 1
        return Next(decision.value, state)

    rig.start(step)
    paused = rig.engine.execute("run")
    assert (paused.lifecycle, paused.call_seq, paused.committed_seq) == ("paused", 1, 0)
    question = rig.engine.pending_question("run")
    assert question == Choose(point=0)
    assert rig.engine.pending_question("run") == question
    assert rig.engine.records("run") == ()
    assert not (entered[0][1] / "record.json").exists()
    with pytest.raises(InvalidRunState):
        rig.engine.resume("run")
    with pytest.raises(InvalidRunState):
        rig.engine.execute("run")
    rig.engine.update_tunables(
        "run", Tunables(value=9), expected_revision=0, actor=Actor("agent", "test")
    )
    final = rig.engine.answer("run", Decision(value=23))
    assert final.lifecycle == "done"
    assert observed == [(23, 1)]
    assert len(entered) == 1
    assert rig.initialized == [START]
    records = rig.engine.records("run")
    assert [(record.call_seq, record.revision) for record in records] == [(1, 0)]
    assert rig.engine.pending_question("run") is None
    assert "step_discarded" not in rig.kinds()


def test_wrong_answer_preserves_question_and_allows_retry(rig: Rig) -> None:
    received: list[Decision] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        decision = yield from env.ask(Choose(point=4))
        assert decision is not None
        received.append(decision)
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    before = rig.kinds()
    with pytest.raises(TypeError, match="Decision"):
        rig.engine.answer("run", OtherAnswer(value=7))
    assert rig.engine.pending_question("run") == Choose(point=4)
    assert rig.kinds() == before
    assert received == []
    assert rig.engine.answer("run", Decision(value=7)).lifecycle == "done"
    assert received == [Decision(value=7)]


def test_each_answer_runs_until_next_question_without_restart(rig: Rig) -> None:
    received: list[int] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        for point in (1, 2):
            decision = yield from env.ask(Choose(point=point))
            assert decision is not None
            received.append(decision.value)
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    second = rig.engine.answer("run", Decision(value=11))
    assert (second.lifecycle, second.call_seq) == ("paused", 1)
    assert rig.engine.pending_question("run") == Choose(point=2)
    assert received == [11]
    assert rig.engine.answer("run", Decision(value=22)).lifecycle == "done"
    assert received == [11, 22]
    with pytest.raises(InvalidRunState):
        rig.engine.answer("run", Decision(value=33))


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_stop_is_control_only_and_drain_resumes_cancelled_ask_on_worker(
    rig: Rig, cleanup_fails: bool
) -> None:
    closed: list[int] = []
    received: list[Decision | None] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        try:
            received.append((yield from env.ask(Choose(point=0))))
            from zcu_tools.experiment.workflows import Cancelled

            def acquisition(_run: Run[Cfg]) -> int:
                pytest.fail("Cancelled invocation must not start new work")

            cancelled = yield from env.run(acquisition, Cfg())
            assert cancelled == Cancelled(run_id="run")
            yield from env.setup_device("unused", FakeDeviceInfo(address="unused"))
            yield from env.wait_until(START + timedelta(days=1))
            received.append((yield from env.ask(Choose(point=1))))
            if cleanup_fails:
                raise ValueError("cancel cleanup failed")
        finally:
            closed.append(get_ident())
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    assert closed == []
    assert rig.engine.pause("run").lifecycle == "paused"
    assert rig.engine.stop("run").lifecycle == "stopping"
    assert closed == []
    with ThreadPoolExecutor(max_workers=1) as pool:
        # The submitted drain has no blocked collaborator and must return.
        stopped = pool.submit(rig.engine.drain, "run").result(timeout=30)
    assert stopped.lifecycle == ("failed" if cleanup_fails else "stopped")
    if cleanup_fails:
        assert stopped.reason == "ValueError: cancel cleanup failed"
    assert len(closed) == 1 and closed[0] != get_ident()
    assert received == [None, None]
    assert rig.devices.values == []
    assert rig.devices.captures == 0
    assert rig.clock.targets == []
    assert rig.engine.pending_question("run") is None
    assert rig.kinds().count("step_failed" if cleanup_fails else "step_discarded") == 1
    assert rig.engine.records("run") == ()


def test_stop_journal_failure_defers_author_teardown_to_unlocked_worker(
    rig: Rig,
) -> None:
    owner = get_ident()
    cleanup_threads: list[int] = []
    statuses: list[object] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        try:
            yield from env.ask(Choose(point=0))
        finally:
            cleanup_threads.append(get_ident())
            if get_ident() != owner:
                with ThreadPoolExecutor(max_workers=1) as probe:
                    # A public status read must finish while author cleanup runs;
                    # no control lock may surround generator teardown.
                    statuses.append(probe.submit(rig.engine.status).result(timeout=30))
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    journal = rig.paths.metadata_root / "journal.jsonl"
    contents = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(DIRECTORY_OPEN_ERROR) as caught:
            rig.engine.stop("run")
    finally:
        journal.rmdir()
        journal.write_bytes(contents)
    assert cleanup_threads == [], "Control I/O failure must not run author cleanup"
    with ThreadPoolExecutor(max_workers=1) as worker:
        # The failed journal request has settled; drain owns only local teardown.
        failed = worker.submit(rig.engine.drain, "run").result(timeout=30)
    assert failed.lifecycle == "failed"
    assert failed.reason == f"{type(caught.value).__name__}: {caught.value}"
    assert len(cleanup_threads) == 1 and cleanup_threads[0] != owner
    assert len(statuses) == 1


def test_answer_reserves_worker_segment_and_pause_discards_on_next_effect(
    rig: Rig,
) -> None:
    entered, release = Event(), Event()
    closed: list[bool] = []

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        try:
            yield from env.ask(Choose(point=0))
            entered.set()
            # The test releases the continuation in its finally block.
            assert release.wait(30)
            yield from env.wait_until(START)
        finally:
            closed.append(True)
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rig.engine.answer, "run", Decision(value=1))
        try:
            # The submitted answer resumes directly into entered.set().
            assert entered.wait(30)
            with pytest.raises(InvalidRunState):
                rig.engine.answer("run", Decision(value=2))
            with pytest.raises(InvalidRunState):
                rig.engine.execute("run")
            assert rig.engine.pause("run").lifecycle == "pausing"
        finally:
            release.set()
        # Release above lets the synchronous segment reach its pause boundary.
        assert future.result(timeout=30).lifecycle == "paused"
    assert closed == [True]
    assert rig.engine.pending_question("run") is None
    assert rig.engine.resume("run").lifecycle == "running"
    assert rig.engine.execute("run").lifecycle == "paused"
    rig.engine.stop("run")
    rig.engine.drain("run")


def test_answer_failure_uses_execution_failure_boundary(rig: Rig) -> None:
    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        yield from env.ask(Choose(point=0))
        raise ValueError("analysis after answer failed")

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    failed = rig.engine.answer("run", Decision(value=1))
    assert failed.lifecycle == "failed"
    assert failed.reason == "ValueError: analysis after answer failed"
    assert rig.engine.pending_question("run") is None
    assert "step_failed" in rig.kinds()


def test_question_controls_reject_stale_run_ids(rig: Rig) -> None:
    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        yield from env.ask(Choose(point=0))
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "paused"
    with pytest.raises(RunMismatch):
        rig.engine.pending_question("stale")
    with pytest.raises(RunMismatch):
        rig.engine.answer("stale", Decision(value=1))
    assert rig.engine.pending_question("run") == Choose(point=0)
    rig.engine.stop("run")
    rig.engine.drain("run")
