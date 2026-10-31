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
from zcu_tools.experiment.stop_signal import StopSignal

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
