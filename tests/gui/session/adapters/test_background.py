"""Tests for BackgroundRunner — the pure OffMain executor (ADR-0066).

Covers both substrates (dedicated QThread vs shared pool), the done/error
delivery on the main thread, that a None result is delivered (not swallowed),
and that work-thunk closures install progress_ambient on the worker thread.

BackgroundRunner is now scope-agnostic: it no longer accepts or reads an
``OffMainScopes`` argument. Scope entering is the caller's responsibility,
embedded in the work thunk.
"""

from __future__ import annotations

import threading

import pytest
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.ports import BackgroundAdmissionClosedError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.scopes import progress_ambient

from tests.gui._qt_waits import qt_until

# Every BackgroundRunner created by a test is registered here so that the
# autouse quiesce fixture can drain it before its QObjects are GC'd.  A queued
# cross-thread delivery dispatched onto a freed C++ object segfaults (same
# pattern as tests/gui/session/services/test_device.py).
_LIVE_BG: list[BackgroundRunner] = []


def _bg() -> BackgroundRunner:
    bg = BackgroundRunner()
    _LIVE_BG.append(bg)
    return bg


@pytest.fixture(autouse=True, name="_quiesce_bg")
def fixture_quiesce_bg():
    """Join all worker threads and flush their queued deliveries before GC."""
    yield
    for bg in _LIVE_BG:
        # Submitted Qt work has no remaining blocker; drain its worker and delivery.
        assert bg.quiesce(timeout_ms=30_000)
    _LIVE_BG.clear()


def _pump_until(qapp, predicate) -> None:
    qt_until(predicate)


def test_dedicated_runs_work_and_delivers_done(qapp):
    bg = _bg()
    got: list[object] = []
    bg.submit(
        lambda: 7,
        run_in_pool=False,
        on_done=got.append,
        on_error=lambda e: got.append(("error", e)),
    )
    _pump_until(qapp, lambda: got)
    assert got == [7]


def test_pool_runs_work_and_delivers_done(qapp):
    bg = _bg()
    got: list[object] = []
    bg.submit(
        lambda: 9,
        run_in_pool=True,
        on_done=got.append,
        on_error=lambda e: got.append(("error", e)),
    )
    _pump_until(qapp, lambda: got)
    assert got == [9]


def test_none_result_is_delivered_not_swallowed(qapp):
    # save's work returns None; on_done must still fire (NO_RESULT sentinel only
    # guards "never produced a value", which a normal None return is not).
    bg = _bg()
    got: list[object] = []
    bg.submit(
        lambda: None,
        run_in_pool=False,
        on_done=lambda r: got.append(("done", r)),
        on_error=lambda e: got.append(("error", e)),
    )
    _pump_until(qapp, lambda: got)
    assert got == [("done", None)]


def test_dedicated_delivers_error(qapp):
    bg = _bg()
    boom = RuntimeError("boom")
    errs: list[Exception] = []

    def work() -> object:
        raise boom

    bg.submit(work, run_in_pool=False, on_done=lambda r: None, on_error=errs.append)
    _pump_until(qapp, lambda: errs)
    assert errs == [boom]


def test_pool_delivers_error(qapp):
    bg = _bg()
    boom = ValueError("pool boom")
    errs: list[Exception] = []

    def work() -> object:
        raise boom

    bg.submit(work, run_in_pool=True, on_done=lambda r: None, on_error=errs.append)
    _pump_until(qapp, lambda: errs)
    assert errs == [boom]


@pytest.mark.parametrize("nested_none", [False, True])
def test_progress_ambient_routes_worker_bars_to_factory(qapp, nested_none):
    from zcu_tools.gui.session.services.progress import ProgressService
    from zcu_tools.progress_bar.interface import make_pbar

    from tests.gui._progress_fakes import DirectProgressTransport

    bg = _bg()
    progress = ProgressService(DirectProgressTransport())
    factory = progress.make_factory(1, owner_id="worker")
    done: list[object] = []

    def work() -> object:
        with (
            progress_ambient(factory),
            progress_ambient(None if nested_none else factory),
        ):
            bar = make_pbar(total=10, desc="working")
            bar.set_progress(4)
        return 1

    bg.submit(
        work,
        run_in_pool=False,
        on_done=done.append,
        on_error=lambda e: pytest.fail(str(e)),
    )
    _pump_until(qapp, lambda: done)
    ((_, model),) = progress.bars_for_owner("worker")
    assert model.n == 4
    assert model.total == 10
    assert model.label == "working"


@pytest.mark.parametrize("run_in_pool", [True, False])
def test_terminal_admission_drains_accepted_delivery_and_rejects_submit(
    qapp, run_in_pool
):
    bg = _bg()
    delivered: list[object] = []
    bg.submit(
        lambda: 7,
        run_in_pool=run_in_pool,
        on_done=delivered.append,
        on_error=delivered.append,
    )
    assert callable(getattr(bg, "stop_accepting", None)), (
        "executor needs terminal admission"
    )
    bg.stop_accepting()
    bg.stop_accepting()
    with pytest.raises(RuntimeError, match="background") as caught:
        bg.submit(
            lambda: 9,
            run_in_pool=run_in_pool,
            on_done=delivered.append,
            on_error=delivered.append,
        )
    assert type(caught.value).__name__ == "BackgroundAdmissionClosedError"
    # Accepted work returns immediately and posts its terminal result.
    assert bg.quiesce(timeout_ms=30_000)
    assert delivered == [7]
    with pytest.raises(RuntimeError, match="background") as caught:
        bg.submit(lambda: 9, on_done=delivered.append, on_error=delivered.append)
    assert type(caught.value).__name__ == "BackgroundAdmissionClosedError"


@pytest.mark.parametrize("run_in_pool", [True, False])
@pytest.mark.parametrize("owner_fails", [True, False])
def test_quiesce_pumps_owner_calls_before_joining_workers(qapp, run_in_pool, owner_fails):
    bg = _bg()
    scheduler = QtOwnerScheduler()
    reached = threading.Event()
    release = threading.Event()
    delivered: list[object] = []
    owner_calls: list[int] = []
    failure = ValueError("owner command failed")

    def owner_command() -> int:
        owner_calls.append(threading.get_ident())
        with pytest.raises(BackgroundAdmissionClosedError):
            bg.submit(lambda: 99, on_done=delivered.append, on_error=delivered.append)
        if owner_fails:
            raise failure
        return 7

    def work() -> int:
        reached.set()
        release.wait()
        return scheduler.call(owner_command)

    bg.submit(work, run_in_pool=run_in_pool, on_done=delivered.append, on_error=delivered.append)
    try:
        qt_until(reached.is_set, label="worker reached external barrier")
        bg.stop_accepting()
        assert not bg.quiesce(timeout_ms=0)
        assert delivered == [] and owner_calls == []
        release.set()
        assert bg.quiesce(timeout_ms=1000), "drain must service owner calls while work is live"
        assert delivered == [failure if owner_fails else 7]
        assert owner_calls == [threading.get_ident()]
        assert bg.quiesce(timeout_ms=0)
    finally:
        release.set()
        assert bg.quiesce(timeout_ms=30_000)


def test_repeatable_quiesce_does_not_close_qt_admission(qapp):
    bg = _bg()
    delivered: list[object] = []
    for value in (1, 2):
        bg.submit(
            lambda value=value: value,
            on_done=delivered.append,
            on_error=delivered.append,
        )
        # This iteration submitted a result-producing task with no blocker.
        assert bg.quiesce(timeout_ms=30_000)
    assert delivered == [1, 2]
