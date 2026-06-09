"""Tests for OperationHandles — the async Handle / Cancel facet (ADR-0019).

Token minting + the three async verbs (await_outcome / poll / cancel) +
cancel_all + live_count, independent of exclusion. A handle-only op (analyze /
interactive) uses exactly this with no OperationGate involvement.
"""

from __future__ import annotations

import threading
import time

from zcu_tools.gui.app.main.services.operation_handles import (
    OperationHandles,
    OperationOutcome,
)


def test_create_mints_unique_increasing_tokens() -> None:
    handles = OperationHandles()
    assert handles.create() == 1
    assert handles.create() == 2
    assert handles.create() == 3


# ---------------------------------------------------------------------------
# await_outcome — thread-safe wait for off-main blocking handlers
# ---------------------------------------------------------------------------


def test_await_outcome_unblocks_on_settle_with_outcome() -> None:
    handles = OperationHandles()
    token = handles.create()

    got: list[object] = []
    dt: list[float] = []

    def waiter() -> None:
        t0 = time.monotonic()
        got.append(handles.await_outcome(token, timeout=3.0))
        dt.append(time.monotonic() - t0)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.1)
    handles.settle(token, OperationOutcome("failed", "boom"))  # from "main" thread
    wt.join(timeout=2.0)

    assert got and got[0] == OperationOutcome("failed", "boom")
    assert dt[0] < 1.0  # woke promptly, not on timeout


def test_await_outcome_immediate_for_already_settled() -> None:
    handles = OperationHandles()
    token = handles.create()
    handles.settle(token, OperationOutcome("finished"))

    t0 = time.monotonic()
    assert handles.await_outcome(token, timeout=5.0) == OperationOutcome("finished")
    assert time.monotonic() - t0 < 0.5


def test_await_outcome_immediate_for_unknown_token() -> None:
    handles = OperationHandles()
    t0 = time.monotonic()
    # Unknown/evicted token is treated as already finished (never hangs).
    assert handles.await_outcome(99999, timeout=5.0) == OperationOutcome("finished")
    assert time.monotonic() - t0 < 0.5


def test_await_outcome_times_out_while_pending() -> None:
    handles = OperationHandles()
    token = handles.create()
    assert handles.await_outcome(token, timeout=0.1) is None


# ---------------------------------------------------------------------------
# poll — non-blocking status
# ---------------------------------------------------------------------------


def test_poll_pending_then_settled() -> None:
    handles = OperationHandles()
    token = handles.create()
    assert handles.poll(token) is None  # still pending
    handles.settle(token, OperationOutcome("finished"))
    assert handles.poll(token) == OperationOutcome("finished")


def test_poll_unknown_token_is_finished() -> None:
    handles = OperationHandles()
    assert handles.poll(99999) == OperationOutcome("finished")


# ---------------------------------------------------------------------------
# cancel — async stop notification (sets the worker's stop_event)
# ---------------------------------------------------------------------------


def test_cancel_sets_the_registered_stop_event() -> None:
    handles = OperationHandles()
    stop_event = threading.Event()
    token = handles.create(stop_event=stop_event)

    assert not stop_event.is_set()
    handles.cancel(token)
    assert stop_event.is_set()
    # cancel is a request, not a settle: the operation stays pending until the
    # worker self-judges and the owner settles the handle.
    assert handles.poll(token) is None


def test_cancel_is_noop_for_operation_without_stop_event() -> None:
    # A connect has no cancellation point: create passes no stop_event, so
    # cancel does nothing (shutdown falls back to a timeout force-close).
    handles = OperationHandles()
    token = handles.create()
    handles.cancel(token)  # must not raise
    assert handles.poll(token) is None


def test_cancel_unknown_token_is_noop() -> None:
    handles = OperationHandles()
    handles.cancel(99999)  # must not raise


def test_cancel_all_sets_every_live_stop_event_and_returns_tokens() -> None:
    handles = OperationHandles()
    setup_stop = threading.Event()
    setup_token = handles.create(stop_event=setup_stop)
    # A connect with no stop_event is still returned (its token must be polled
    # for shutdown to know whether it settled), but its flag stays unset.
    connect_token = handles.create()

    tokens = handles.cancel_all()

    assert set(tokens) == {setup_token, connect_token}
    assert setup_stop.is_set()


def test_cancel_all_ignores_already_settled_operations() -> None:
    handles = OperationHandles()
    token = handles.create(stop_event=threading.Event())
    handles.settle(token, OperationOutcome("finished"))

    assert handles.cancel_all() == []


# ---------------------------------------------------------------------------
# live_count — how many operations are in progress (shutdown confirm)
# ---------------------------------------------------------------------------


def test_live_count_tracks_pending_operations() -> None:
    handles = OperationHandles()
    assert handles.live_count() == 0
    t1 = handles.create()
    t2 = handles.create()
    assert handles.live_count() == 2
    handles.settle(t1, OperationOutcome("finished"))
    assert handles.live_count() == 1
    handles.settle(t2, OperationOutcome("finished"))
    assert handles.live_count() == 0
