from __future__ import annotations

import threading
import time
import weakref

import pytest
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.adapters.thread_pool_background import (
    ThreadPoolBackgroundExecutor,
)


def _pump_until(
    scheduler: ManualOwnerScheduler,
    predicate,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        scheduler.pump_once(block=True, timeout=0.02)
    assert predicate()


@pytest.mark.parametrize("run_in_pool", [True, False])
def test_delivers_success_on_owner_thread(run_in_pool: bool) -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
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
    assert executor.quiesce()


@pytest.mark.parametrize("run_in_pool", [True, False])
def test_delivers_exact_error_on_owner_thread(run_in_pool: bool) -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
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
    assert executor.quiesce()


def test_quiesce_waits_for_worker_and_delivery_ack() -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
    release = threading.Event()
    delivered: list[str] = []
    quiesced: list[bool] = []

    executor.submit(
        lambda: release.wait(timeout=1.0),
        run_in_pool=True,
        on_done=lambda _: delivered.append("done"),
        on_error=lambda error: pytest.fail(str(error)),
    )

    waiter = threading.Thread(target=lambda: quiesced.append(executor.quiesce(1.0)))
    waiter.start()
    release.set()
    _pump_until(scheduler, lambda: bool(delivered))
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert quiesced == [True]
    assert delivered == ["done"]


def test_owner_quiesce_does_not_deadlock_queued_delivery() -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
    worker_done = threading.Event()
    delivered: list[int] = []

    def work() -> int:
        worker_done.set()
        return 1

    executor.submit(
        work,
        run_in_pool=True,
        on_done=delivered.append,
        on_error=lambda error: pytest.fail(str(error)),
    )
    assert worker_done.wait(timeout=1.0)

    assert executor.quiesce(1.0) is False
    assert delivered == []
    assert scheduler.pump_all() == 1
    assert executor.quiesce(1.0) is True
    assert delivered == [1]


def test_timeout_does_not_cancel_or_drop_terminal_delivery() -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
    release = threading.Event()
    delivered: list[int] = []

    executor.submit(
        lambda: (release.wait(timeout=1.0), 3)[1],
        run_in_pool=False,
        on_done=delivered.append,
        on_error=lambda error: pytest.fail(str(error)),
    )

    assert executor.quiesce(0.001) is False
    release.set()
    _pump_until(scheduler, lambda: bool(delivered))
    assert executor.quiesce(1.0) is True
    assert delivered == [3]


def test_quiesce_rejects_new_submissions() -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
    assert executor.quiesce()

    with pytest.raises(RuntimeError, match="quiescing"):
        executor.submit(
            lambda: None,
            run_in_pool=True,
            on_done=lambda _: None,
            on_error=lambda _: None,
        )


def test_terminal_callback_failure_is_acknowledged(caplog) -> None:
    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)

    def fail_callback(_: object) -> None:
        raise RuntimeError("callback boom")

    executor.submit(
        lambda: 1,
        run_in_pool=True,
        on_done=fail_callback,
        on_error=lambda error: pytest.fail(str(error)),
    )
    _pump_until(scheduler, lambda: "background terminal callback failed" in caplog.text)

    assert executor.quiesce()


def test_terminal_delivery_releases_completed_future_result() -> None:
    class Payload:
        pass

    scheduler = ManualOwnerScheduler()
    executor = ThreadPoolBackgroundExecutor(scheduler)
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
    _pump_until(scheduler, lambda: bool(delivered) and not executor._jobs)

    assert executor.quiesce()
    assert delivered == [True]
    assert result_ref() is None
