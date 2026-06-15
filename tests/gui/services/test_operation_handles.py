"""Tests for OperationHandles and OperationChannel (ADR-0019 / ADR-0025).

Token minting + the three async verbs (await_outcome / poll / cancel) +
cancel_all + live_count, independent of exclusion. A handle-only op (analyze /
interactive) uses exactly this with no OperationGate involvement.

ADR-0025: cross-thread interaction uses per-op OperationChannel (ordered FIFO).
Tests cover the three return reasons (completed / user_feedback / timeout),
the non-terminal guarantee (feedback/timeout leave handle unsettled),
and the folding rules (Stop+Settled, Stop-then-Message fold into reason,
idempotent settle, interactive direct-settle cancel_hook ordering).
"""

from __future__ import annotations

import threading
import time

import pytest
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    OperationChannel,
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
    # Woke promptly via Queue.get, not on a full 2s tick.
    assert dt[0] < 2.0


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
# await_outcome — user_feedback path (ADR-0025 Message event)
# ---------------------------------------------------------------------------


def test_await_outcome_wakes_on_message() -> None:
    """handles.message(token, text) is delivered as user_feedback (non-terminal)."""
    handles = OperationHandles()
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
    time.sleep(0.05)  # let the thread enter Queue.get
    handles.message(token, "stop and recalibrate")
    wt.join(timeout=4.0)

    assert results
    r = results[0]
    assert r.reason == "user_feedback"
    assert r.feedback == "stop and recalibrate"
    # operation is NOT settled — still pending
    assert handles.poll(token) is None
    # woke promptly (Queue.get, no 2s poll)
    assert dt[0] < 2.0


def test_await_outcome_handle_still_awitable_after_feedback() -> None:
    """user_feedback is non-terminal — re-await on the same token still works."""
    handles = OperationHandles()
    token = handles.create()

    # Deliver message
    handles.message(token, "adjust gain")
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


def test_await_outcome_message_returns_immediately_if_queued_before_await() -> None:
    """Message enqueued before await_outcome is called surfaces on entry."""
    handles = OperationHandles()
    token = handles.create()

    handles.message(token, "recalibrate now")
    t0 = time.monotonic()
    r = handles.await_outcome(token, timeout=10.0)
    elapsed = time.monotonic() - t0

    assert r is not None
    assert r.reason == "user_feedback"
    assert "recalibrate now" in (r.feedback or "")
    assert elapsed < 0.5


def test_await_outcome_operation_wins_before_message() -> None:
    """When the operation settles before a message arrives, reason='completed'."""
    handles = OperationHandles()
    token = handles.create()

    results: list[AwaitResult] = []

    def waiter() -> None:
        r = handles.await_outcome(token, timeout=10.0)
        if r is not None:
            results.append(r)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.05)
    # Settle before sending a message
    handles.settle(token, OperationOutcome("finished"))
    wt.join(timeout=4.0)

    assert results
    assert results[0].reason == "completed"


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
# cancel — async stop notification (invokes cancel_hook)
# ---------------------------------------------------------------------------


def test_cancel_invokes_registered_cancel_hook() -> None:
    called: list[bool] = []

    def hook() -> None:
        called.append(True)

    handles = OperationHandles()
    token = handles.create(cancel_hook=hook)

    handles.cancel(token)
    assert called == [True]
    # cancel is a request, not a settle: the operation stays pending.
    assert handles.poll(token) is None


def test_cancel_is_noop_for_operation_without_hook() -> None:
    handles = OperationHandles()
    token = handles.create()
    handles.cancel(token)  # must not raise
    assert handles.poll(token) is None


def test_cancel_unknown_token_is_noop() -> None:
    handles = OperationHandles()
    handles.cancel(99999)  # must not raise


def test_cancel_all_invokes_every_live_hook_and_returns_tokens() -> None:
    setup_called: list[bool] = []
    setup_token_box: list[int] = []
    connect_token_box: list[int] = []

    def setup_hook() -> None:
        setup_called.append(True)

    handles = OperationHandles()
    setup_token_box.append(handles.create(cancel_hook=setup_hook))
    connect_token_box.append(handles.create())  # no hook

    tokens = handles.cancel_all()

    assert set(tokens) == {setup_token_box[0], connect_token_box[0]}
    assert setup_called == [True]


def test_cancel_all_ignores_already_settled_operations() -> None:
    handles = OperationHandles()
    token = handles.create(cancel_hook=lambda: None)
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


# ---------------------------------------------------------------------------
# OperationChannel folding semantics (ADR-0025)
# ---------------------------------------------------------------------------


def test_channel_send_and_stop_folded_into_reason() -> None:
    """Stop(reason) enqueued before Settled(cancelled) — reason folds into feedback."""
    ch = OperationChannel()
    # Simulate: stop enqueued first, then worker settles cancelled.
    ch.stop("stop - wrong resonator")
    ch.settle(OperationOutcome("cancelled"))

    r = ch.consume(timeout=1.0)
    assert r.reason == "completed"
    assert r.outcome == OperationOutcome("cancelled")
    assert r.feedback == "stop - wrong resonator"


def test_channel_pure_message_nudge_is_nonterminal() -> None:
    """Message without a preceding Stop returns user_feedback (non-terminal)."""
    ch = OperationChannel()
    ch.message("recalibrate gain")

    r = ch.consume(timeout=1.0)
    assert r.reason == "user_feedback"
    assert r.feedback == "recalibrate gain"
    # Channel is not settled.
    assert ch.settled_outcome() is None


def test_channel_stop_then_settled_folds_reason_across_consume_calls() -> None:
    """Stop seen in one consume, Settled seen in the next: reason must fold."""
    ch = OperationChannel()
    ch.stop("abort reason")

    # First consume: sees Stop, returns timeout (no Settled yet).
    r1 = ch.consume(timeout=0.05)
    assert r1.reason == "timeout"

    # Settle after the first consume.
    ch.settle(OperationOutcome("cancelled"))

    # Second consume: sees Settled; _pending_stop_reason is still latched.
    r2 = ch.consume(timeout=1.0)
    assert r2.reason == "completed"
    assert r2.outcome is not None
    assert r2.outcome.status == "cancelled"
    assert r2.feedback == "abort reason"


def test_channel_terminal_idempotent_reconsume() -> None:
    """Already-settled channel: re-consuming always returns completed immediately."""
    ch = OperationChannel()
    ch.settle(OperationOutcome("finished"))

    r1 = ch.consume(timeout=1.0)
    assert r1.reason == "completed"

    # Re-consume: must return immediately (not block).
    t0 = time.monotonic()
    r2 = ch.consume(timeout=5.0)
    assert time.monotonic() - t0 < 0.5
    assert r2.reason == "completed"


def test_channel_timeout() -> None:
    """consume on an empty channel returns timeout after the deadline."""
    ch = OperationChannel()
    r = ch.consume(timeout=0.05)
    assert r.reason == "timeout"


def test_channel_cancel_hook_triggers_direct_settle() -> None:
    """Interactive cancel: Stop enqueued → hook runs → hook settles channel.

    Ordering: Stop arrives before Settled in the queue, so the consumer
    latches the reason and folds it into the subsequent Settled(cancelled).
    """
    ch: OperationChannel | None = None
    reason_text = "user said stop"

    def direct_settle_hook() -> None:
        # Simulates cancel_interactive: settles the channel as cancelled.
        assert ch is not None
        ch.settle(OperationOutcome("cancelled"))

    ch = OperationChannel(cancel_hook=direct_settle_hook)
    ch.stop(reason_text)
    # stop() enqueued Stop BEFORE calling hook; hook settled the channel.

    r = ch.consume(timeout=1.0)
    assert r.reason == "completed"
    assert r.outcome is not None
    assert r.outcome.status == "cancelled"
    # reason folded: Stop arrived before Settled in the queue.
    assert r.feedback == reason_text


def test_channel_message_during_cancel_folds_into_reason() -> None:
    """Message arriving after Stop (cancel in progress) folds into stop reason."""
    ch = OperationChannel()
    ch.stop("first stop")
    ch.message("also this info")
    ch.settle(OperationOutcome("cancelled"))

    r = ch.consume(timeout=1.0)
    assert r.reason == "completed"
    assert r.outcome is not None
    assert r.outcome.status == "cancelled"
    assert "first stop" in (r.feedback or "")
    assert "also this info" in (r.feedback or "")


def test_channel_settled_no_stop_has_no_feedback() -> None:
    """A cancelled outcome without any Stop event has no feedback."""
    ch = OperationChannel()
    ch.settle(OperationOutcome("cancelled"))

    r = ch.consume(timeout=1.0)
    assert r.reason == "completed"
    assert r.feedback is None


def test_channel_finished_outcome_no_feedback_even_with_stop() -> None:
    """feedback is ONLY attached for cancelled outcomes, not for finished."""
    ch = OperationChannel()
    ch.stop("some reason")
    ch.settle(OperationOutcome("finished"))

    r = ch.consume(timeout=1.0)
    assert r.reason == "completed"
    assert r.outcome is not None
    assert r.outcome.status == "finished"
    # feedback is not populated for non-cancelled outcomes.
    assert r.feedback is None


def test_channel_blank_message_ignored() -> None:
    """Blank/whitespace-only messages are silently ignored."""
    ch = OperationChannel()
    ch.message("   ")
    # No event in queue; consume should timeout.
    r = ch.consume(timeout=0.05)
    assert r.reason == "timeout"


def test_handles_message_method_delivers_to_live_channel() -> None:
    """handles.message(token, text) routes to the live channel."""
    handles = OperationHandles()
    token = handles.create()
    handles.message(token, "test nudge")
    r = handles.await_outcome(token, timeout=1.0)
    assert r is not None
    assert r.reason == "user_feedback"
    assert r.feedback == "test nudge"


def test_handles_stop_with_reason_folds_into_cancelled() -> None:
    """handles.stop(token, reason) + settle cancelled → feedback=reason."""
    handles = OperationHandles()
    token = handles.create()
    handles.stop(token, reason="agent aborted")
    handles.settle(token, OperationOutcome("cancelled"))

    r = handles.await_outcome(token, timeout=1.0)
    assert r is not None
    assert r.reason == "completed"
    assert r.outcome is not None
    assert r.outcome.status == "cancelled"
    assert r.feedback == "agent aborted"


def test_handles_cancel_invokes_hook() -> None:
    """handles.cancel(token) calls the registered cancel_hook via channel.stop."""
    called: list[bool] = []
    handles = OperationHandles()
    token = handles.create(cancel_hook=lambda: called.append(True))
    handles.cancel(token)
    assert called == [True]


def test_settle_window_keeps_token_reachable() -> None:
    """Registry TOCTOU regression (ADR-0025): ``settle`` publishes the channel
    to ``_done`` BEFORE retracting it from ``_live``, so a concurrent
    ``await_outcome`` / ``poll`` always finds it in at least one dict and never
    falls through to the default 'finished' (which would misreport a
    cancelled/failed terminal). This hand-constructs both in-flight windows."""
    handles = OperationHandles()
    token = handles.create()
    ch = handles._live[token]
    ch.settle(OperationOutcome("cancelled", "stopped"))

    # Window 1: published to _done, not yet retracted from _live (in BOTH).
    handles._done[token] = ch
    assert handles.poll(token) == OperationOutcome("cancelled", "stopped")
    r = handles.await_outcome(token, timeout=0.5)
    assert r is not None and r.reason == "completed"
    assert r.outcome is not None and r.outcome.status == "cancelled"

    # Window 2: retracted from _live, in _done only.
    handles._live.pop(token, None)
    assert handles.poll(token) == OperationOutcome("cancelled", "stopped")
    r2 = handles.await_outcome(token, timeout=0.5)
    assert r2 is not None and r2.reason == "completed"
    assert r2.outcome is not None and r2.outcome.status == "cancelled"


# ---------------------------------------------------------------------------
# OperationChannel.can_cancel (ADR-0025 §Stop-gating)
# ---------------------------------------------------------------------------


def test_channel_can_cancel_true_when_hook_registered() -> None:
    """A channel with a cancel hook reports can_cancel=True."""
    ch = OperationChannel(cancel_hook=lambda: None)
    assert ch.can_cancel is True


def test_channel_can_cancel_false_when_no_hook() -> None:
    """A channel with no cancel hook (e.g. connect) reports can_cancel=False."""
    ch = OperationChannel(cancel_hook=None)
    assert ch.can_cancel is False


# ---------------------------------------------------------------------------
# OperationHandles.has_cancel_hook (ADR-0025 §Stop-gating)
# ---------------------------------------------------------------------------


def test_has_cancel_hook_live_token_with_hook() -> None:
    """A live token with a cancel hook returns True."""
    handles = OperationHandles()
    token = handles.create(cancel_hook=lambda: None)
    assert handles.has_cancel_hook(token) is True


def test_has_cancel_hook_live_token_without_hook() -> None:
    """A live token without a cancel hook (e.g. connect) returns False."""
    handles = OperationHandles()
    token = handles.create(cancel_hook=None)
    assert handles.has_cancel_hook(token) is False


def test_has_cancel_hook_settled_token_with_hook() -> None:
    """A settled (done) token with a hook is still reachable via _done."""
    handles = OperationHandles()
    token = handles.create(cancel_hook=lambda: None)
    handles.settle(token, OperationOutcome("finished"))
    # Token moved to _done but must still be queryable.
    assert handles.has_cancel_hook(token) is True


def test_has_cancel_hook_settled_token_without_hook() -> None:
    """A settled token without a hook returns False."""
    handles = OperationHandles()
    token = handles.create(cancel_hook=None)
    handles.settle(token, OperationOutcome("finished"))
    assert handles.has_cancel_hook(token) is False


def test_has_cancel_hook_unknown_token_returns_false() -> None:
    """Unknown token (never created) returns False — no hook by definition."""
    handles = OperationHandles()
    assert handles.has_cancel_hook(99999) is False


def test_has_cancel_hook_is_pure_read_does_not_trigger_hook() -> None:
    """has_cancel_hook must never call the cancel hook."""
    fired: list[bool] = []

    def hook() -> None:
        fired.append(True)

    handles = OperationHandles()
    token = handles.create(cancel_hook=hook)
    result = handles.has_cancel_hook(token)
    assert result is True
    assert fired == []  # hook must NOT have been called


def test_pure_nudge_between_awaits_is_delivered_not_dropped() -> None:
    """A pure nudge enqueued while the op is settling (or between awaits) is
    delivered as user_feedback on the next consume, not silently folded away
    (ADR-0025 in-order drain). The subsequent settle then completes."""
    handles = OperationHandles()
    token = handles.create()
    handles.message(token, "also check the readout freq")
    handles.settle(token, OperationOutcome("finished"))

    # First consume surfaces the queued nudge (non-terminal) ...
    r1 = handles.await_outcome(token, timeout=0.5)
    assert r1 is not None and r1.reason == "user_feedback"
    assert r1.feedback == "also check the readout freq"
    # ... and the next consume returns the terminal outcome.
    r2 = handles.await_outcome(token, timeout=0.5)
    assert r2 is not None and r2.reason == "completed"
    assert r2.outcome is not None and r2.outcome.status == "finished"
