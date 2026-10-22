"""Public async host contracts, independent of GUI and experiment definitions."""

import asyncio
import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest
from pydantic import ValidationError
from zcu_tools.core.script_protocol import Answer, Question
from zcu_tools.experiment.stop_signal import StopSignal


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


def test_stop_tree_inherits_stop_and_propagates_first_error() -> None:
    root = StopSignal()
    assert hasattr(root, "child"), "StopSignal has no child scope contract"
    child = root.child()
    sibling = root.child()
    child.set()
    assert not root.is_set()
    assert not sibling.is_set()
    root.set()
    assert sibling.is_set()
    assert root.child().is_set()
    child.set_error("failed", "acquire failed", ValueError("bad point"))
    assert root.error is not None
    assert root.error.reason == "acquire failed"
    child.release()
    root.clear_stop()
    root.set()
    child.clear_stop()
    assert not child.is_set(), "Released operation scopes must be detached"


def test_call_runs_in_executor_and_author_finally_stays_on_worker() -> None:
    module = host_module()
    host = module.ScriptHost()
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


@pytest.mark.parametrize("scope", ["script", "run"])
def test_cooperative_cancel_never_cancels_author_and_starts_later_calls(
    scope: str,
) -> None:
    module = host_module()
    host = module.ScriptHost()
    entered = Event()
    calls: list[bool] = []
    finished: list[int] = []

    def operation(ctx) -> bool:
        entered.set()
        assert ctx.stop_signal.event.wait(3), "cancel did not reach active operation"
        return ctx.stop_signal.is_set()

    async def script() -> str:
        try:
            calls.append((await host.call(operation)).result)
            calls.append((await host.call(lambda ctx: ctx.stop_signal.is_set())).result)
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
            host.cancel(scope="script") if not entered.is_set() else None
        report, worker_id = future.result(timeout=3)
    assert report == "author report"
    assert calls == [True, scope == "script"]
    assert finished == [worker_id]
    assert host.status == ("cancelled" if scope == "script" else "completed")


def test_question_request_ids_reject_invalid_duplicate_and_late_answers() -> None:
    module = host_module()
    changed = Event()
    host = module.ScriptHost(on_change=changed.set)

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
    host = module.ScriptHost(on_change=changed.set)
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
    host = module.ScriptHost()
    entered = Event()
    delivered = Event()
    attempts: list[object] = []

    def blocking(ctx) -> int:
        entered.set()
        assert ctx.stop_signal.event.wait(3)
        return 11

    async def script() -> int:
        first = await host.call(blocking)
        attempts.append(first)
        delivered.set()
        await host.commit()
        second = await host.call(lambda ctx: ctx.stop_signal.is_set())
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
    host = module.ScriptHost(on_change=changed.set)
    completions: list[object] = []

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
