"""Tests for OperationHandles — the async Handle / Cancel facet (ADR-0019).

Token minting + the three async verbs (await_outcome / poll / cancel) +
cancel_all + live_count, independent of exclusion. A handle-only op (analyze /
interactive) uses exactly this with no OperationGate involvement.

ADR-0023 extension: await_outcome supports FeedbackInbox as a second wakeup
source. Tests cover the three return reasons (completed / user_feedback / timeout)
and the non-terminal guarantee (feedback/timeout leave the handle unsettled and
re-awaitble).
"""

from __future__ import annotations

import threading
import time

import pytest
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    FeedbackInbox,
    OperationHandles,
    OperationOutcome,
)

# ---------------------------------------------------------------------------
# OperationHandles basics (ADR-0019)
# ---------------------------------------------------------------------------


def test_create_mints_unique_increasing_tokens() -> None:
    handles = OperationHandles()
    assert handles.create() == 1
    assert handles.create() == 2
    assert handles.create() == 3


# ---------------------------------------------------------------------------
# AwaitResult shape
# ---------------------------------------------------------------------------


def test_await_result_completed_requires_outcome() -> None:
    with pytest.raises(ValueError, match="must have outcome"):
        AwaitResult(reason="completed")


def test_await_result_user_feedback_requires_feedback() -> None:
    with pytest.raises(ValueError, match="must have feedback"):
        AwaitResult(reason="user_feedback")


def test_await_result_timeout_ok() -> None:
    r = AwaitResult(reason="timeout")
    assert r.reason == "timeout"
    assert r.outcome is None
    assert r.feedback is None


# ---------------------------------------------------------------------------
# await_outcome — completed path (ADR-0019 contract)
# ---------------------------------------------------------------------------


def test_await_outcome_unblocks_on_settle() -> None:
    handles = OperationHandles()
    token = handles.create()

    results: list[AwaitResult] = []
    dt: list[float] = []

    def waiter() -> None:
        t0 = time.monotonic()
        r = handles.await_outcome(token, timeout=5.0)
        assert r is not None
        results.append(r)
        dt.append(time.monotonic() - t0)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.05)
    handles.settle(token, OperationOutcome("failed", "boom"))
    wt.join(timeout=4.0)

    assert results
    r = results[0]
    assert r.reason == "completed"
    assert r.outcome == OperationOutcome("failed", "boom")
    assert dt[0] < 2.0  # woke promptly, not on a full tick


def test_await_outcome_immediate_for_already_settled() -> None:
    handles = OperationHandles()
    token = handles.create()
    handles.settle(token, OperationOutcome("finished"))

    t0 = time.monotonic()
    r = handles.await_outcome(token, timeout=10.0)
    elapsed = time.monotonic() - t0

    assert r is not None
    assert r.reason == "completed"
    assert r.outcome == OperationOutcome("finished")
    assert elapsed < 0.5  # fast path, no blocking


def test_await_outcome_immediate_for_unknown_token() -> None:
    handles = OperationHandles()
    t0 = time.monotonic()
    r = handles.await_outcome(99999, timeout=10.0)
    assert time.monotonic() - t0 < 0.5

    assert r is not None
    assert r.reason == "completed"
    assert r.outcome == OperationOutcome("finished")


def test_await_outcome_times_out_while_pending() -> None:
    handles = OperationHandles()
    token = handles.create()
    r = handles.await_outcome(token, timeout=0.05)
    assert r is not None
    assert r.reason == "timeout"
    # timeout is non-terminal: handle is still pending
    assert handles.poll(token) is None


def test_await_outcome_handle_still_awitable_after_timeout() -> None:
    """Timeout does not settle the handle — a re-await still blocks until settle."""
    handles = OperationHandles()
    token = handles.create()

    # First await — times out
    r1 = handles.await_outcome(token, timeout=0.05)
    assert r1 is not None
    assert r1.reason == "timeout"

    # Settle and re-await — must succeed
    handles.settle(token, OperationOutcome("finished"))
    r2 = handles.await_outcome(token, timeout=1.0)
    assert r2 is not None
    assert r2.reason == "completed"
    assert r2.outcome is not None
    assert r2.outcome.status == "finished"


# ---------------------------------------------------------------------------
# await_outcome — user_feedback path (ADR-0023)
# ---------------------------------------------------------------------------


def test_await_outcome_wakes_on_feedback() -> None:
    inbox = FeedbackInbox()
    handles = OperationHandles(feedback_inbox=inbox)
    token = handles.create()

    results: list[AwaitResult] = []
    dt: list[float] = []

    def waiter() -> None:
        t0 = time.monotonic()
        r = handles.await_outcome(token, timeout=10.0)
        assert r is not None
        results.append(r)
        dt.append(time.monotonic() - t0)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.05)  # let the thread enter the poll loop
    inbox.post("stop and recalibrate")
    wt.join(timeout=6.0)  # at most one tick + margin

    assert results
    r = results[0]
    assert r.reason == "user_feedback"
    assert r.feedback == "stop and recalibrate"
    # operation is NOT settled — still pending
    assert handles.poll(token) is None
    # returned reasonably fast (within one 2s tick + margin)
    assert dt[0] < 5.0


def test_await_outcome_handle_still_awitable_after_feedback() -> None:
    """user_feedback is non-terminal — re-await on the same token still works."""
    inbox = FeedbackInbox()
    handles = OperationHandles(feedback_inbox=inbox)
    token = handles.create()

    # Deliver feedback
    inbox.post("adjust gain")
    r1 = handles.await_outcome(token, timeout=5.0)
    assert r1 is not None
    assert r1.reason == "user_feedback"
    assert "adjust gain" in (r1.feedback or "")

    # Handle still pending
    assert handles.poll(token) is None

    # Settle and re-await — must work
    handles.settle(token, OperationOutcome("finished"))
    r2 = handles.await_outcome(token, timeout=2.0)
    assert r2 is not None
    assert r2.reason == "completed"


def test_await_outcome_pre_enter_feedback_returns_immediately() -> None:
    """Feedback posted before await_outcome is called is returned on entry."""
    inbox = FeedbackInbox()
    handles = OperationHandles(feedback_inbox=inbox)
    token = handles.create()

    # Post before entering await
    inbox.post("recalibrate now")
    t0 = time.monotonic()
    r = handles.await_outcome(token, timeout=10.0)
    elapsed = time.monotonic() - t0

    assert r is not None
    assert r.reason == "user_feedback"
    assert "recalibrate now" in (r.feedback or "")
    assert elapsed < 0.5  # pre-enter fast path, not a full tick


def test_await_outcome_multiple_feedback_joined() -> None:
    """Multiple feedback messages before drain are joined into one string."""
    inbox = FeedbackInbox()
    handles = OperationHandles(feedback_inbox=inbox)
    token = handles.create()

    inbox.post("line 1")
    inbox.post("line 2")
    r = handles.await_outcome(token, timeout=5.0)
    assert r is not None
    assert r.reason == "user_feedback"
    assert "line 1" in (r.feedback or "")
    assert "line 2" in (r.feedback or "")


def test_await_outcome_operation_win_before_feedback_tick() -> None:
    """When the operation settles before a feedback arrives on the next tick,
    reason='completed' is returned (the operation win takes priority)."""
    inbox = FeedbackInbox()
    handles = OperationHandles(feedback_inbox=inbox)
    token = handles.create()

    results: list[AwaitResult] = []

    def waiter() -> None:
        r = handles.await_outcome(token, timeout=10.0)
        if r is not None:
            results.append(r)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.05)
    # Settle before posting feedback
    handles.settle(token, OperationOutcome("finished"))
    wt.join(timeout=4.0)

    assert results
    assert results[0].reason == "completed"


def test_feedback_inbox_ignores_whitespace_only() -> None:
    inbox = FeedbackInbox()
    inbox.post("   ")
    assert not inbox.has_pending()


def test_feedback_inbox_drain_clears_event() -> None:
    inbox = FeedbackInbox()
    inbox.post("hello")
    assert inbox.notify_event.is_set()
    msgs = inbox.drain()
    assert msgs == ["hello"]
    assert not inbox.notify_event.is_set()


def test_feedback_inbox_concurrent_post_and_drain() -> None:
    """Producer/consumer thread safety: many posts, drain collects all."""
    inbox = FeedbackInbox()
    n = 50

    def producer() -> None:
        for i in range(n):
            inbox.post(f"msg{i}")

    t = threading.Thread(target=producer)
    t.start()
    t.join()

    msgs = inbox.drain()
    assert len(msgs) == n


# ---------------------------------------------------------------------------
# set_feedback_inbox wiring
# ---------------------------------------------------------------------------


def test_set_feedback_inbox_wires_after_construction() -> None:
    handles = OperationHandles()  # no inbox initially
    inbox = FeedbackInbox()
    handles.set_feedback_inbox(inbox)

    token = handles.create()
    inbox.post("late-wired feedback")
    r = handles.await_outcome(token, timeout=5.0)
    assert r is not None
    assert r.reason == "user_feedback"


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
