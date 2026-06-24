"""Tests for BackgroundRunner — the pure OffMain executor (ADR-0019, ADR-0026 §2).

Covers both substrates (dedicated QThread vs shared pool), the done/error
delivery on the main thread, that a None result is delivered (not swallowed),
and that work-thunk closures correctly install ambient scopes on the worker
thread (figure routing via ``figure_ambient``, pbar via ``progress_ambient``).

BackgroundRunner is now scope-agnostic: it no longer accepts or reads an
``OffMainScopes`` argument. Scope entering is the caller's responsibility,
embedded in the work thunk.
"""

from __future__ import annotations

import time

import pytest
from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.gui.app.main.services.scopes import figure_ambient
from zcu_tools.gui.plotting.routing import get_current_container
from zcu_tools.gui.session.scopes import progress_ambient

# Every BackgroundRunner created by a test is registered here so that the
# autouse quiesce fixture can drain it before its QObjects are GC'd.  A queued
# cross-thread delivery dispatched onto a freed C++ object segfaults (same
# pattern as tests/gui/services/test_device.py).
_LIVE_BG: list[BackgroundRunner] = []


def _bg() -> BackgroundRunner:
    bg = BackgroundRunner()
    _LIVE_BG.append(bg)
    return bg


@pytest.fixture(autouse=True)
def _quiesce_bg():
    """Join all worker threads and flush their queued deliveries before GC."""
    yield
    for bg in _LIVE_BG:
        bg.quiesce()
    _LIVE_BG.clear()


def _pump_until(qapp, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


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


def test_figure_ambient_routing_active_during_work(qapp):
    # Work thunk closes over figure_ambient: the routing ContextVar is set on
    # the worker thread for the duration of the thunk (ADR-0026 §2).
    # BackgroundRunner itself is unaware of the scope — only the thunk knows.
    bg = _bg()
    container = object()
    seen: list[object] = []

    def work() -> object:
        with figure_ambient(container):  # type: ignore[arg-type]
            seen.append(get_current_container())
        return 1

    bg.submit(
        work,
        run_in_pool=False,
        on_done=lambda r: None,
        on_error=lambda e: None,
    )
    _pump_until(qapp, lambda: seen)
    assert seen == [container]


def test_no_figure_ambient_leaves_routing_unset(qapp):
    # When the work thunk does not install figure_ambient the routing ContextVar
    # is absent on the worker thread (the default is None).
    bg = _bg()
    seen: list[object] = []

    def work() -> object:
        seen.append(get_current_container())
        return 1

    bg.submit(
        work,
        run_in_pool=True,
        on_done=lambda r: None,
        on_error=lambda e: None,
    )
    _pump_until(qapp, lambda: seen)
    assert seen == [None]


def test_figure_ambient_none_container_is_noop(qapp):
    # figure_ambient(None) is a no-op: routing stays unset.
    bg = _bg()
    seen: list[object] = []

    def work() -> object:
        with figure_ambient(None):
            seen.append(get_current_container())
        return 1

    bg.submit(
        work,
        run_in_pool=False,
        on_done=lambda r: None,
        on_error=lambda e: None,
    )
    _pump_until(qapp, lambda: seen)
    assert seen == [None]


def test_progress_ambient_installs_pbar_factory(qapp):
    # progress_ambient installs the pbar ContextVar so the worker thread can
    # create progress bars (ADR-0026 §2). We verify it is callable from the
    # worker by reading back the ContextVar value directly.
    from zcu_tools.progress_bar.interface import _pbar_factory as _cv

    bg = _bg()
    sentinel = object()
    seen: list[object] = []

    def work() -> object:
        with progress_ambient(sentinel):  # type: ignore[arg-type]
            seen.append(_cv.get())
        return 1

    bg.submit(
        work,
        run_in_pool=False,
        on_done=lambda r: None,
        on_error=lambda e: None,
    )
    _pump_until(qapp, lambda: seen)
    assert seen == [sentinel]


def test_progress_ambient_none_factory_is_noop(qapp):
    # progress_ambient(None) is a no-op: the pbar ContextVar stays at its default.
    from zcu_tools.progress_bar.interface import _pbar_factory as _cv

    bg = _bg()
    seen: list[object] = []

    def work() -> object:
        with progress_ambient(None):
            seen.append(_cv.get())
        return 1

    bg.submit(
        work,
        run_in_pool=False,
        on_done=lambda r: None,
        on_error=lambda e: None,
    )
    _pump_until(qapp, lambda: seen)
    assert seen == [None]
