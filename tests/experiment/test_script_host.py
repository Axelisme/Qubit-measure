"""Public async host contracts, independent of GUI and experiment definitions."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.scope import Scope, ScriptCancelled
from zcu_tools.experiment.stop_signal import ScheduleOutcomeError, StopSignal

if TYPE_CHECKING:
    from zcu_tools.experiment.script_host import CallResult


class Approval(Answer):
    accepted: bool


class Confirm(Question[Approval]):
    evidence: str


def host_module():
    """Require the production host at its public package boundary."""
    assert importlib.util.find_spec("zcu_tools.experiment.script_host") is not None, (
        "The shared production async host is not implemented"
    )
    return importlib.import_module("zcu_tools.experiment.script_host")


def test_stop_tree_inherits_stop_and_releases_completed_scopes() -> None:
    root = StopSignal()
    child = root.child()
    sibling = root.child()
    child.set()
    assert not root.is_set()
    assert not sibling.is_set()
    root.set()
    assert sibling.is_set()
    assert root.child().is_set()
    child.release()
    root.clear_stop()
    root.set()
    child.clear_stop()
    assert not child.is_set(), "Released operation scopes must be detached"


def test_child_error_stops_only_its_scope_and_preserves_first_error() -> None:
    root = StopSignal()
    iteration = root.child()
    operation = iteration.child()
    sibling = iteration.child()
    descendant = operation.child().child()
    cause = ValueError("bad point")

    operation.set_error("failed", "acquire failed", cause)
    first_error = operation.error
    operation.set_error("interrupted", "later interruption", None)

    assert operation.is_set()
    assert descendant.is_set()
    assert descendant.error is None
    assert operation.error is first_error
    with pytest.raises(ScheduleOutcomeError, match="acquire failed") as raised:
        operation.raise_if_error()
    assert raised.value is first_error
    assert raised.value.status == "failed"
    assert raised.value.exception is cause
    assert raised.value.__cause__ is cause
    inherited = operation.child()
    assert inherited.is_set()
    assert inherited.error is first_error

    next_sibling = iteration.child()
    for unaffected in (root, iteration, sibling, next_sibling):
        assert not unaffected.is_set()
        assert unaffected.error is None
        unaffected.raise_if_error()


def test_release_hands_first_error_up_without_stop_and_stops_at_operation() -> None:
    root = StopSignal()
    iteration = root.child(boundary=False)
    operation = iteration.child()
    schedule = operation.child(boundary=False)
    acquire = schedule.child(boundary=False)
    later = schedule.child(boundary=False)
    cause = OSError("acquire disconnected")
    acquire.set_error("failed", "first failure", cause)
    first = acquire.error
    assert schedule.error is None
    acquire.release()
    assert schedule.error is first
    assert not schedule.is_set()
    later.set_error("failed", "second failure", ValueError("later"))
    later.release()
    assert schedule.error is first
    schedule.release()
    assert operation.error is first
    assert not operation.is_set()
    with pytest.raises(ScheduleOutcomeError, match="first failure") as caught:
        operation.raise_if_error()
    assert caught.value.__cause__ is cause
    operation.release()
    assert iteration.error is None
    assert not iteration.is_set()
    next_operation = iteration.child()
    assert next_operation.error is None
    assert not next_operation.is_set()


def test_release_delivers_partial_without_stopping_parent_or_next_operation() -> None:
    iteration = StopSignal()
    operation = iteration.child()
    assert getattr(operation, "partial_result", None) is False
    schedule = operation.child(boundary=False)
    acquire = schedule.child(boundary=False)
    acquire.set()
    assert not schedule.partial_result
    acquire.release()
    assert schedule.partial_result
    assert not schedule.is_set()
    schedule.release()
    assert operation.partial_result
    assert not operation.is_set()
    operation.release()
    assert not iteration.partial_result
    assert not iteration.child().partial_result


def test_host_labels_calls_under_explicit_iteration_lifetimes() -> None:
    host = host_module().ScriptHost(Scope("workflow", name="test"))
    observed: list[str] = []

    def observe(ctx) -> None:
        assert ctx.scope.kind == "operation"
        observed.append(host.active_path)

    async def script() -> None:
        with host.root.branch("iteration") as iteration:
            await host.call(observe, scope=iteration)
            await host.call(observe, scope=iteration)
        with host.root.branch("iteration") as iteration:
            await host.call(observe, scope=iteration)

    host.run(script())
    assert observed == [
        "workflow:test/iteration#1/operation#1",
        "workflow:test/iteration#1/operation#2",
        "workflow:test/iteration#2/operation#1",
    ]
    assert host.active_path == "workflow:test"


def test_call_runs_in_executor_and_author_finally_stays_on_worker() -> None:
    module = host_module()
    host = module.ScriptHost(Scope("workflow", name="test"))
    author_threads: list[int] = []
    executor_threads: list[int] = []
    worker = get_ident()

    async def script() -> int:
        try:
            author_threads.append(get_ident())
            completion = await host.call(
                lambda ctx: executor_threads.append(get_ident()) or 17
            )
            return completion.result
        finally:
            author_threads.append(get_ident())

    assert host.run(script()) == 17
    assert author_threads == [worker, worker]
    assert executor_threads[0] != worker
    assert host.status == "completed"


@pytest.mark.parametrize("scope", ["run"])
def test_cooperative_cancel_never_cancels_author_and_starts_later_calls(
    scope: str,
) -> None:
    module = host_module()
    host = module.ScriptHost(Scope("workflow", name="test"))
    entered = Event()
    calls: list[bool] = []
    finished: list[int] = []

    def operation(ctx) -> bool:
        entered.set()
        assert ctx.scope.stop.event.wait(3), "cancel did not reach active operation"
        return ctx.scope.stop.is_set()

    async def script() -> str:
        try:
            calls.append((await host.call(operation)).result)
            calls.append((await host.call(lambda ctx: ctx.scope.stop.is_set())).result)
            return "author report"
        except asyncio.CancelledError:
            pytest.fail("The host cancelled the author task")
        finally:
            finished.append(get_ident())

    def worker() -> tuple[str, int]:
        return host.run(script()), get_ident()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        try:
            assert entered.wait(3)
            host.cancel(scope=scope)
        finally:
            host.cancel(scope="workflow") if not entered.is_set() else None
        report, worker_id = future.result(timeout=3)
    assert report == "author report"
    assert calls == [True, scope == "workflow"]
    assert finished == [worker_id]
    assert host.status == ("cancelled" if scope == "workflow" else "completed")


def test_question_request_ids_reject_invalid_duplicate_and_late_answers() -> None:
    module = host_module()
    changed = Event()
    host = module.ScriptHost(Scope("workflow", name="test"), on_change=changed.set)

    async def script() -> Approval | None:
        return await host.ask(Confirm(evidence="fit"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: host.run(script()))
        try:
            assert changed.wait(3)
            pending = host.pending_question
            assert pending is not None
            with pytest.raises(ValidationError):
                host.answer(pending.request_id, {"unknown": True})
            with pytest.raises(ValueError, match="request"):
                host.answer("stale", {"accepted": True})
            host.answer(pending.request_id, {"accepted": True})
            with pytest.raises((ValueError, RuntimeError)):
                host.answer(pending.request_id, {"accepted": True})
            assert future.result(timeout=3) == Approval(accepted=True)
        finally:
            if not future.done():
                host.close()


def test_close_drains_question_and_preserves_cancelled_cleanup_error() -> None:
    module = host_module()
    changed = Event()
    host = module.ScriptHost(Scope("workflow", name="test"), on_change=changed.set)
    cleaned = Event()

    async def script() -> None:
        try:
            assert await host.ask(Confirm(evidence="partial")) is None
        finally:
            cleaned.set()
            raise ValueError("cleanup failed")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: host.run(script()))
        assert changed.wait(3)
        host.close()
        with pytest.raises(ValueError, match="cleanup failed"):
            future.result(timeout=3)
    assert cleaned.is_set()
    assert host.status == "cancelled"
    assert str(host.error) == "cleanup failed"


def test_pause_stops_only_operation_and_gate_retains_iteration_until_resume() -> None:
    module = host_module()
    host = module.ScriptHost(Scope("workflow", name="test"))
    entered = Event()
    delivered = Event()
    attempts: list[CallResult[int | bool]] = []

    def blocking(ctx) -> int:
        entered.set()
        assert ctx.scope.stop.event.wait(3)
        return 11

    async def script() -> int:
        first = await host.call(blocking)
        attempts.append(first)
        delivered.set()
        await host.wait_until_resumed()
        second = await host.call(lambda ctx: ctx.scope.stop.is_set())
        attempts.append(second)
        return second.epoch

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: host.run(script()))
        try:
            assert entered.wait(3)
            host.pause()
            assert delivered.wait(3)
            # A second acknowledged pause orders before the gate can release.
            host.pause()
            assert host.epoch == 2
            host.resume()
            assert future.result(timeout=3) == 2
        finally:
            if not future.done():
                host.close()
    assert [(item.result, item.epoch, item.partial_result) for item in attempts] == [
        (11, 0, True),
        (False, 2, False),
    ]


def test_natural_completion_snapshot_is_not_retroactively_stopped_by_pause() -> None:
    module = host_module()
    changed = Event()
    host = module.ScriptHost(Scope("workflow", name="test"), on_change=changed.set)
    completions: list[CallResult[int]] = []

    async def script() -> None:
        completions.append(await host.call(lambda ctx: 19))
        await host.ask(Confirm(evidence="natural completion"))
        await host.wait_until_resumed()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: host.run(script()))
        try:
            assert changed.wait(3)
            host.pause()
            pending = host.pending_question
            assert pending is not None
            host.answer(pending.request_id, {"accepted": True})
            host.resume()
            future.result(timeout=3)
        finally:
            if not future.done():
                host.close()
    assert [(item.result, item.epoch, item.partial_result) for item in completions] == [
        (19, 0, False)
    ]


def test_public_control_serializes_on_worker_returns_value_and_transports_errors() -> (
    None
):
    module = host_module()
    changed = Event()
    host = module.ScriptHost(Scope("workflow", name="test"), on_change=changed.set)
    worker_ids: list[int] = []

    class ControlFailure(Exception):
        pass

    failure = ControlFailure("specific callback diagnostic")

    def broken() -> None:
        raise failure

    async def script() -> None:
        worker_ids.append(get_ident())
        await host.ask(Confirm(evidence="control"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: host.run(script()))
        try:
            assert changed.wait(3)
            assert hasattr(host, "control"), (
                "Host has no public memory-only control primitive"
            )
            assert host.control(lambda: (get_ident(), 23)) == (worker_ids[0], 23)
            with pytest.raises(ControlFailure) as caught:
                host.control(broken)
            assert caught.value is failure
        finally:
            host.close()
            with pytest.raises(ScriptCancelled):
                future.result(timeout=3)
    with pytest.raises(RuntimeError):
        host.control(lambda: 1)
