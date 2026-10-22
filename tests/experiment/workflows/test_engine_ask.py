"""Questions suspend one async author and accept request-bound answers."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from threading import Event, get_ident

import pytest
from pydantic import ValidationError
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.device import FakeDeviceInfo
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import (
    Actor,
    InvalidRunState,
    Live,
    QuestionMismatch,
    RunMismatch,
    WorkflowEnv,
)

from tests._filesystem import DIRECTORY_OPEN_ERROR

from ._engine_fakes import (
    START,
    Plan,
    Point,
    Rig,
    Tunables,
    completed,
    pending,
    question,
    running,
    until,
)


class Decision(Answer):
    value: int


class OtherAnswer(Answer):
    text: str


class Choose(Question[Decision]):
    point: int


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path)
    yield rig
    rig.close()


async def ask_once(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
    answer = await env.ask(Choose(point=0))
    if answer is not None:
        await env.commit(Point(value=answer.value))


def test_pending_answer_type_and_parse_are_readonly_and_allow_retry(rig: Rig) -> None:
    rig.start(ask_once)
    assert rig.engine.pending_answer_type("run") is None
    with pytest.raises(QuestionMismatch) as absent:
        rig.engine.parse_answer("run", {"value": 7}, request_id="no-question")
    assert absent.value.pending_request_id is None
    with running(rig) as future:
        request_id = pending(rig)
        assert rig.engine.pending_answer_type("run") is Decision
        before = rig.events()
        with pytest.raises(RunMismatch):
            rig.engine.parse_answer("stale", {"value": 7}, request_id=request_id)
        with pytest.raises(ValidationError):
            rig.engine.parse_answer("run", {"unknown": 3}, request_id=request_id)
        assert rig.engine.parse_answer(
            "run", {"value": 7}, request_id=request_id
        ) == Decision(value=7)
        assert rig.events() == before
        assert question(rig).request_id == request_id
        assert not future.done()
        with pytest.raises(InvalidRunState):
            rig.engine.resume("run")
        rig.engine.answer("run", Decision(value=7), request_id=request_id)
        assert future.result(timeout=10).lifecycle == "done"
    assert rig.engine.pending_answer_type("run") is None
    assert rig.engine.records("run")[0].encoded_record.value == {"value": 7}
    with pytest.raises(QuestionMismatch) as ended:
        rig.engine.parse_answer("run", {"value": 8}, request_id=request_id)
    assert ended.value.pending_request_id is None


def test_answer_continues_same_author_with_live_tunables(rig: Rig) -> None:
    entered: list[int] = []
    seen: list[tuple[int, int]] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        entered.append(get_ident())
        captured = tun.current
        answer = await env.ask(Choose(point=0))
        assert answer is not None
        seen.append((captured.value, tun.current.value))
        await env.commit(Point(value=answer.value))

    rig.start(script)
    with running(rig) as future:
        request_id = pending(rig)
        rig.engine.update_tunables(
            "run", Tunables(value=8), expected_revision=0, actor=Actor("agent", "test")
        )
        rig.engine.answer("run", Decision(value=4), request_id=request_id)
        assert future.result(timeout=10).lifecycle == "done"
    assert len(entered) == 1
    assert entered[0] != get_ident()
    assert seen == [(1, 8)]
    assert rig.engine.records("run")[0].revisions == (0, 1)


def test_wrong_answer_and_request_preserve_pending_question(rig: Rig) -> None:
    rig.start(ask_once)
    with running(rig) as future:
        request_id = pending(rig)
        before = rig.events()
        with pytest.raises(TypeError):
            rig.engine.answer("run", OtherAnswer(text="wrong"), request_id=request_id)
        with pytest.raises(QuestionMismatch) as error:
            rig.engine.answer("run", Decision(value=1), request_id="stale")
        assert error.value.pending_request_id == request_id
        assert rig.events() == before
        assert question(rig).question == Choose(point=0)
        rig.engine.answer("run", Decision(value=2), request_id=request_id)
        assert future.result(timeout=10).lifecycle == "done"


def test_each_answer_continues_until_next_question_without_restart(rig: Rig) -> None:
    received: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        for index in range(plan.count):
            answer = await env.ask(Choose(point=index))
            assert answer is not None
            received.append(answer.value)

    rig.start(script)
    with running(rig) as future:
        first_id = pending(rig)
        rig.engine.answer("run", Decision(value=2), request_id=first_id)
        until(
            lambda: (
                (q := rig.engine.pending_question("run")) is not None
                and q.request_id != first_id
            ),
            "second question",
        )
        second_id = pending(rig)
        assert question(rig).question == Choose(point=1)
        with pytest.raises(QuestionMismatch):
            rig.engine.answer("run", Decision(value=99), request_id=first_id)
        rig.engine.answer("run", Decision(value=3), request_id=second_id)
        assert future.result(timeout=10).lifecycle == "done"
    assert received == [2, 3]


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_stop_delivers_none_and_cooperative_stopped_calls_then_runs_finally(
    rig: Rig,
    cleanup_fails: bool,
) -> None:
    closed: list[int] = []
    received: list[Decision | None] = []
    stops: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            received.append(await env.ask(Choose(point=0)))

            def operation(ctx: CallContext) -> int:
                stops.append(ctx.stop_signal.is_set())
                return 0

            result = completed(await env.call(operation))
            assert result.partial_result and result.value == 0
            assert completed(
                await env.setup_device(
                    "unused", FakeDeviceInfo(address="unused", value=0.5)
                )
            ).partial_result
            assert completed(
                await env.wait_until(START + timedelta(days=1))
            ).partial_result
            received.append(await env.ask(Choose(point=1)))
            if cleanup_fails:
                raise ValueError("cancel cleanup failed")
        finally:
            closed.append(get_ident())

    rig.start(script)
    with running(rig) as future:
        pending(rig)
        assert closed == []
        rig.engine.stop("run")
        final = future.result(timeout=10)
    assert final.lifecycle == ("failed" if cleanup_fails else "cancelled")
    if cleanup_fails:
        assert final.reason == "ValueError: cancel cleanup failed"
    assert len(closed) == 1 and closed[0] != get_ident()
    assert received == [None, None]
    assert stops == [True]
    assert rig.devices.values == [("unused", 0.5)]
    assert rig.devices.signals[0].is_set()
    assert rig.clock.targets == [START + timedelta(days=1)]
    assert rig.engine.pending_question("run") is None
    assert rig.engine.records("run") == ()


def test_stop_journal_failure_keeps_original_cause_and_cleans_up_on_worker(
    rig: Rig,
) -> None:
    cleanup_threads: list[int] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            await env.ask(Choose(point=0))
        finally:
            cleanup_threads.append(get_ident())

    rig.start(script)
    with running(rig) as future:
        request_id = pending(rig)
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
        try:
            # A control I/O failure must fail and drain the existing worker.
            final = future.result(timeout=10)
        finally:
            # If a broken engine leaves Ask pending, release it only for teardown.
            # The timeout above still fails the test; no result is accepted here.
            if not future.done() and rig.engine.pending_question("run") is not None:
                rig.engine.answer("run", Decision(value=0), request_id=request_id)
                future.result(timeout=10)
    assert final.lifecycle == "failed"
    assert final.reason == f"{type(caught.value).__name__}: {caught.value}"
    assert len(cleanup_threads) == 1 and cleanup_threads[0] != get_ident()
    assert rig.engine.records("run") == ()


def test_answer_is_nonblocking_and_continuation_errors_fail_execution(rig: Rig) -> None:
    entered, release = Event(), Event()

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        await env.ask(Choose(point=0))

        def operation(ctx: CallContext) -> None:
            entered.set()
            assert release.wait(10), "Caller releases operation in finally"
            raise ValueError("analysis after answer failed")

        await env.call(operation)

    rig.start(script)
    with running(rig) as future:
        request_id = pending(rig)
        try:
            rig.engine.answer("run", Decision(value=1), request_id=request_id)
            assert entered.wait(10), "Answer continues on retained worker"
            assert not future.done()
            with pytest.raises(InvalidRunState):
                rig.engine.execute("run")
        finally:
            release.set()
        final = future.result(timeout=10)
    assert final.lifecycle == "failed"
    assert final.reason == "ValueError: analysis after answer failed"


def test_question_controls_reject_stale_run_ids(rig: Rig) -> None:
    rig.start(ask_once)
    with running(rig):
        request_id = pending(rig)
        with pytest.raises(RunMismatch):
            rig.engine.pending_question("stale")
        with pytest.raises(RunMismatch):
            rig.engine.answer("stale", Decision(value=1), request_id=request_id)
        assert question(rig).question == Choose(point=0)
