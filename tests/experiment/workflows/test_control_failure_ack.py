"""A failed answer journal cannot make its caller wait for authored finally."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, get_ident

import pytest
from zcu_tools.experiment.workflows import Live, WorkflowEnv

from tests.experiment._loop_hold import LoopHold

from ._engine_fakes import Plan, Rig, Tunables, running
from .test_engine_ask import Choose, Decision, pending


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    instance = Rig(tmp_path)
    yield instance
    instance.close()


def test_first_failure_stop_ack_does_not_wait_behind_author_finally(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold = LoopHold(phase="after", release_on_second=True)
    hold.install(monkeypatch)
    cleanup_entered, release_cleanup, caller_returned = Event(), Event(), Event()
    cleanup_threads: list[int] = []
    stopped_at_cleanup: list[bool] = []
    fault = OSError("answer journal storage failed")
    original_open = Path.open

    def open_journal(path: Path, mode: str = "r", *args, **kwargs):
        if path == rig.paths.metadata_root / "journal.jsonl" and mode == "a":
            raise fault
        return original_open(path, mode, *args, **kwargs)

    async def author(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        try:
            await env.ask(Choose(point=0))
        finally:
            cleanup_threads.append(get_ident())
            stopped_at_cleanup.append(env.cancelled)
            cleanup_entered.set()
            assert release_cleanup.wait(10), "Scenario releases author cleanup"

    def answer(request_id: str) -> None:
        hold.thread = get_ident()
        try:
            with pytest.raises(
                OSError, match="answer journal storage failed"
            ) as raised:
                rig.engine.answer("run", Decision(value=1), request_id=request_id)
            assert raised.value is fault, "Answer caller retains the original I/O cause"
        finally:
            caller_returned.set()
            hold.release.set()

    rig.start(author)
    with running(rig) as execution, ThreadPoolExecutor(max_workers=1) as pool:
        request_id = pending(rig)
        monkeypatch.setattr(Path, "open", open_journal)
        caller = pool.submit(answer, request_id)
        try:
            assert cleanup_entered.wait(5), "I/O failure reaches authored finally"
            assert caller_returned.wait(2), (
                "First root acknowledgement must not wait behind author finally; "
                f"root stopped at finally: {stopped_at_cleanup}"
            )
            assert not execution.done(), (
                "The worker remains reserved until finally drains"
            )
        finally:
            hold.release.set()
            release_cleanup.set()
            caller.result(timeout=5)
            final = execution.result(timeout=5)
    assert final.lifecycle == "failed"
    assert final.reason == "OSError: answer journal storage failed"
    assert len(cleanup_threads) == 1 and cleanup_threads[0] != hold.thread
