"""Root acceptance is immediate and idempotent; completion still waits for author cleanup."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import pytest
from zcu_tools.experiment.scope import Scope, ScriptCancelled
from zcu_tools.experiment.script_host import ScriptHost

from ._loop_hold import LoopHold


def test_root_cancel_reuses_acceptance_while_first_loop_notification_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Scope("workflow", name="synthetic")
    host = ScriptHost(root)
    entered, cleanup_entered, release_cleanup = Event(), Event(), Event()
    second_returned = Event()
    cleanup_threads: list[int] = []
    hold = LoopHold(phase="before")
    hold.install(monkeypatch)

    async def author() -> None:
        try:

            def work(ctx) -> None:
                entered.set()
                assert ctx.scope.stop.event.wait(10), "Root stop reaches executor work"
                assert hold.release.wait(10), "Work drains after the held notification"

            await host.call(work)
        finally:
            cleanup_threads.append(get_ident())
            cleanup_entered.set()
            assert release_cleanup.wait(10), "Scenario releases author cleanup"

    def first_cancel():
        hold.thread = get_ident()
        return host.cancel()

    def second_cancel():
        try:
            return host.cancel()
        finally:
            second_returned.set()

    with ThreadPoolExecutor(max_workers=3) as pool:
        execution = pool.submit(host.run, author())
        assert entered.wait(5), "The author has admitted its executor call"
        first = pool.submit(first_cancel)
        try:
            assert hold.entered.wait(5), "First root notification is queued but held"
            second = pool.submit(second_cancel)
            assert second_returned.wait(2), (
                "A repeated root cancel must not wait for loop delivery"
            )
            assert first.done(), "First acceptance also completes at the owner latch"
            first_ack, second_ack = first.result(), second.result()
            assert first_ack == second_ack
            assert (first_ack.cancelled, first_ack.active_path, root.cancelled) == (
                True,
                "workflow:synthetic",
                True,
            )
            hold.release.set()
            assert cleanup_entered.wait(5), "Accepted Stop reaches authored finally"
            assert not execution.done(), (
                "Acknowledgement does not mean finally has drained"
            )
        finally:
            hold.release.set()
            release_cleanup.set()
        with pytest.raises(ScriptCancelled):
            execution.result(timeout=5)
    assert len(cleanup_threads) == 1 and cleanup_threads[0] != get_ident()
    assert host.error is None
