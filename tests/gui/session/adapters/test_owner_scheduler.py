from __future__ import annotations

import threading
import time

import pytest
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.ports import OwnerScheduler


def _assert_owner_scheduler_contract(scheduler: OwnerScheduler) -> None:
    assert scheduler.is_owner_thread()


def test_owner_scheduler_adapters_expose_owner_probe(qapp) -> None:  # noqa: ARG001
    _assert_owner_scheduler_contract(ManualOwnerScheduler())
    _assert_owner_scheduler_contract(QtOwnerScheduler())


def test_manual_scheduler_posts_until_owner_pumps() -> None:
    scheduler = ManualOwnerScheduler()
    seen: list[int] = []

    thread = threading.Thread(target=lambda: scheduler.post(lambda: seen.append(1)))
    thread.start()
    thread.join()

    assert seen == []
    assert scheduler.pump_all() == 1
    assert seen == [1]


def test_manual_scheduler_call_returns_foreign_thread_result() -> None:
    scheduler = ManualOwnerScheduler()
    result: list[int] = []

    thread = threading.Thread(target=lambda: result.append(scheduler.call(lambda: 7)))
    thread.start()
    assert scheduler.pump_once(block=True, timeout=1.0)
    thread.join()

    assert result == [7]


def test_manual_scheduler_call_propagates_callback_error() -> None:
    scheduler = ManualOwnerScheduler()
    errors: list[BaseException] = []

    def callback() -> int:
        raise ValueError("boom")

    def invoke() -> None:
        try:
            scheduler.call(callback)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert scheduler.pump_once(block=True, timeout=1.0)
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert str(errors[0]) == "boom"


def test_manual_scheduler_rejects_owner_thread_call_and_foreign_pump() -> None:
    scheduler = ManualOwnerScheduler()
    with pytest.raises(RuntimeError, match="owner thread"):
        scheduler.call(lambda: None)

    errors: list[BaseException] = []

    def pump() -> None:
        try:
            scheduler.pump_all()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=pump)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_qt_scheduler_posts_and_calls_on_owner_thread(qapp) -> None:
    scheduler = QtOwnerScheduler()
    owner_thread_id = threading.get_ident()
    posted: list[int] = []
    called: list[int] = []

    def worker() -> None:
        scheduler.post(lambda: posted.append(threading.get_ident()))
        called.append(scheduler.call(threading.get_ident))

    thread = threading.Thread(target=worker)
    thread.start()
    deadline = time.monotonic() + 1.0
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.001)
    thread.join(timeout=0.1)

    assert not thread.is_alive()
    assert posted == [owner_thread_id]
    assert called == [owner_thread_id]


def test_qt_scheduler_rejects_owner_thread_call(qapp) -> None:  # noqa: ARG001
    scheduler = QtOwnerScheduler()
    with pytest.raises(RuntimeError, match="owner thread"):
        scheduler.call(lambda: None)
