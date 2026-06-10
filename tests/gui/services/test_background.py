"""Tests for BackgroundService — the OffMain execution mechanism (ADR-0019).

Covers both substrates (dedicated QThread vs shared pool), the done/error
delivery on the main thread, that a None result is delivered (not swallowed),
and that the opt-in OffMainScopes are entered on the worker thread (figure
routing present when requested, absent when not).
"""

from __future__ import annotations

import time

import pytest
from zcu_tools.gui.app.main.services.background import (
    BackgroundService,
    OffMainScopes,
)
from zcu_tools.gui.plotting.routing import get_current_container

# Every BackgroundService created by a test is registered here so that the
# autouse quiesce fixture can drain it before its QObjects are GC'd.  A queued
# cross-thread delivery dispatched onto a freed C++ object segfaults (same
# pattern as tests/gui/services/test_device.py).
_LIVE_BG: list[BackgroundService] = []


def _bg() -> BackgroundService:
    bg = BackgroundService()
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


def test_figure_container_scope_active_during_work(qapp):
    # The figure_container scope sets the routing ContextVar on the worker thread
    # for the duration of work (routing + liveplot are one facet, driven by this
    # single field). A plain object suffices — routing just stores it.
    bg = _bg()
    container = object()
    seen: list[object] = []

    def work() -> object:
        seen.append(get_current_container())
        return 1

    bg.submit(
        work,
        OffMainScopes(figure_container=container),  # type: ignore[arg-type]
        run_in_pool=False,
        on_done=lambda r: None,
        on_error=lambda e: None,
    )
    _pump_until(qapp, lambda: seen)
    assert seen == [container]


def test_no_scopes_leaves_routing_unset(qapp):
    # Opt-out: with no figure_container, work runs with no routing container.
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
