from __future__ import annotations

import threading
import weakref
from collections.abc import Callable

import pytest
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.adapters.thread_pool_background import (
    ThreadPoolBackgroundExecutor,
)

from tests.gui._lifecycle_waits import manual_until as _pump_until
from tests.gui._lifecycle_waits import waiting_call


class RecordingOwner(ManualOwnerScheduler):
    """Record real terminal queue arrival without running the owner callback."""

    def __init__(self) -> None:
        super().__init__()
        self.posted = threading.Event()

    def post(self, callback: Callable[[], None]) -> None:
        super().post(callback)
        self.posted.set()


@pytest.fixture
def make_executor(request: pytest.FixtureRequest):
    def make(scheduler: ManualOwnerScheduler) -> ThreadPoolBackgroundExecutor:
        executor = ThreadPoolBackgroundExecutor(scheduler)

        def close() -> None:
            # Every accepted worker is released by its test; drain terminal acks on failure too.
            _pump_until(scheduler, lambda: executor.quiesce(30))

        request.addfinalizer(close)
        return executor

    return make


@pytest.mark.parametrize("run_in_pool", [True, False])
def test_delivers_success_on_owner_thread(make_executor, run_in_pool: bool) -> None:
    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)
    owner_id = threading.get_ident()
    delivered: list[tuple[int, int]] = []

    executor.submit(
        lambda: 7,
        run_in_pool=run_in_pool,
        on_done=lambda value: delivered.append((value, threading.get_ident())),
        on_error=lambda error: pytest.fail(str(error)),
    )

    _pump_until(scheduler, lambda: bool(delivered))
    assert delivered == [(7, owner_id)]
    # The observed terminal callback has acked; only worker bookkeeping may remain.
    assert executor.quiesce(30)


@pytest.mark.parametrize("run_in_pool", [True, False])
def test_delivers_exact_error_on_owner_thread(make_executor, run_in_pool: bool) -> None:
    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)
    owner_id = threading.get_ident()
    boom = ValueError("boom")
    delivered: list[tuple[Exception, int]] = []

    def work() -> None:
        raise boom

    executor.submit(
        work,
        run_in_pool=run_in_pool,
        on_done=lambda value: pytest.fail(f"unexpected result: {value}"),
        on_error=lambda error: delivered.append((error, threading.get_ident())),
    )

    _pump_until(scheduler, lambda: bool(delivered))
    assert delivered == [(boom, owner_id)]
    # The observed terminal callback has acked; only worker bookkeeping may remain.
    assert executor.quiesce(30)


def test_quiesce_waits_for_worker_and_delivery_ack(make_executor, monkeypatch) -> None:
    scheduler = RecordingOwner()
    executor = make_executor(scheduler)
    release = threading.Event()
    delivered: list[str] = []

    executor.submit(
        lambda: release.wait(),
        run_in_pool=True,
        on_done=lambda _: delivered.append("done"),
        on_error=lambda error: pytest.fail(str(error)),
    )

    with waiting_call(monkeypatch, lambda: executor.quiesce(30)) as (arrived, result):
        try:
            # The submitted quiesce worker enters its external condition wait.
            assert arrived.wait(30)
            assert delivered == []
            assert not result.done()
        finally:
            release.set()
            try:
                # Released work posts a real terminal callback, but only owner pumping acks it.
                assert scheduler.posted.wait(30)
                assert delivered == []
                assert not result.done()
            finally:
                _pump_until(scheduler, lambda: bool(delivered))
    assert result.result() is True
    assert delivered == ["done"]


def test_owner_quiesce_does_not_deadlock_queued_delivery(make_executor) -> None:
    scheduler = RecordingOwner()
    executor = make_executor(scheduler)
    delivered: list[int] = []

    executor.submit(
        lambda: 1,
        run_in_pool=True,
        on_done=delivered.append,
        on_error=lambda error: pytest.fail(str(error)),
    )
    # submit has arranged a terminal post; record arrival after queue insertion.
    assert scheduler.posted.wait(timeout=30)

    assert executor.quiesce(30) is False
    assert delivered == []
    assert scheduler.pump_all() == 1
    assert executor.quiesce(30) is True
    assert delivered == [1]


def test_timeout_does_not_cancel_or_drop_terminal_delivery(make_executor) -> None:
    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)
    release = threading.Event()
    delivered: list[int] = []

    executor.submit(
        lambda: (release.wait(), 3)[1],
        run_in_pool=False,
        on_done=delivered.append,
        on_error=lambda error: pytest.fail(str(error)),
    )

    try:
        assert executor.quiesce(0) is False
        with pytest.raises(RuntimeError, match="quiescing"):
            executor.submit(
                lambda: None, on_done=delivered.append, on_error=delivered.append
            )
        assert delivered == []
    finally:
        release.set()
        _pump_until(scheduler, lambda: bool(delivered))
        # Released work posts its terminal callback; wait only for completion bookkeeping.
        assert executor.quiesce(30) is True
    assert delivered == [3]


def test_quiesce_rejects_new_submissions(make_executor) -> None:
    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)
    assert executor.quiesce()

    with pytest.raises(RuntimeError, match="quiescing"):
        executor.submit(
            lambda: None,
            run_in_pool=True,
            on_done=lambda _: None,
            on_error=lambda _: None,
        )


def test_terminal_callback_failure_is_acknowledged(make_executor, caplog) -> None:
    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)

    def fail_callback(_: object) -> None:
        raise RuntimeError("callback boom")

    executor.submit(
        lambda: 1,
        run_in_pool=True,
        on_done=fail_callback,
        on_error=lambda error: pytest.fail(str(error)),
    )
    _pump_until(scheduler, lambda: "background terminal callback failed" in caplog.text)

    # The observed terminal callback has acked; only worker bookkeeping may remain.
    assert executor.quiesce(30)


def test_terminal_delivery_releases_completed_future_result(make_executor) -> None:
    class Payload:
        pass

    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)
    payload = Payload()
    result_ref = weakref.ref(payload)
    delivered: list[bool] = []

    executor.submit(
        lambda value=payload: value,
        run_in_pool=True,
        on_done=lambda _: delivered.append(True),
        on_error=lambda error: pytest.fail(str(error)),
    )
    del payload
    _pump_until(scheduler, lambda: bool(delivered))

    # The observed terminal callback has acked; only worker bookkeeping may remain.
    assert executor.quiesce(30)
    assert delivered == [True]
    assert result_ref() is None


@pytest.mark.parametrize("run_in_pool", [True, False])
def test_terminal_quiesce_uses_named_admission_rejection(
    make_executor, run_in_pool: bool
) -> None:
    scheduler = ManualOwnerScheduler()
    executor = make_executor(scheduler)
    assert executor.quiesce()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="background") as caught:
            executor.submit(
                lambda: None,
                run_in_pool=run_in_pool,
                on_done=lambda _: None,
                on_error=lambda _: None,
            )
        assert type(caught.value).__name__ == "BackgroundAdmissionClosedError"
        assert executor.quiesce()
