"""Tests for NotifyChannel — per-prompt FIFO (ADR-0025 invariants, Stage 4b).

Covers:
  - reply / dismiss / timeout event folding
  - set-once latch: second producer call is silently ignored
  - drain-first: pre-queued events returned without blocking
  - producer non-blocking: put() never stalls (instantaneous in tests)
  - idempotent re-consume: a settled channel returns same result on every call
  - backstop timeout: returns NotifyResult("timeout") when no event arrives
"""

from __future__ import annotations

import threading
import time

from zcu_tools.gui.session.notify_handles import (
    NotifyChannel,
    NotifyResult,
)

# ---------------------------------------------------------------------------
# Event folding: each terminal event maps to the correct NotifyResult
# ---------------------------------------------------------------------------


def test_reply_event_folds_to_reply_reason() -> None:
    ch = NotifyChannel()
    ch.reply("hello")
    result = ch.consume(timeout=1.0)
    assert result.reason == "reply"
    assert result.reply == "hello"


def test_reply_empty_string_is_valid() -> None:
    ch = NotifyChannel()
    ch.reply("")
    result = ch.consume(timeout=1.0)
    assert result.reason == "reply"
    assert result.reply == ""


def test_dismiss_event_folds_to_dismiss_reason() -> None:
    ch = NotifyChannel()
    ch.dismiss()
    result = ch.consume(timeout=1.0)
    assert result.reason == "dismiss"
    assert result.reply is None


def test_timeout_event_folds_to_timeout_reason() -> None:
    ch = NotifyChannel()
    ch.timeout()
    result = ch.consume(timeout=1.0)
    assert result.reason == "timeout"
    assert result.reply is None


# ---------------------------------------------------------------------------
# Set-once latch: second producer call is silently ignored
# ---------------------------------------------------------------------------


def test_set_once_dismiss_then_reply_keeps_dismiss() -> None:
    ch = NotifyChannel()
    ch.dismiss()
    ch.reply("late reply")  # should be ignored
    result = ch.consume(timeout=1.0)
    assert result.reason == "dismiss"


def test_set_once_reply_then_timeout_keeps_reply() -> None:
    ch = NotifyChannel()
    ch.reply("first")
    ch.timeout()  # should be ignored
    result = ch.consume(timeout=1.0)
    assert result.reason == "reply"
    assert result.reply == "first"


def test_set_once_timeout_then_dismiss_keeps_timeout() -> None:
    ch = NotifyChannel()
    ch.timeout()
    ch.dismiss()  # should be ignored
    result = ch.consume(timeout=1.0)
    assert result.reason == "timeout"


# ---------------------------------------------------------------------------
# Drain-first: pre-queued event returned without blocking
# ---------------------------------------------------------------------------


def test_drain_first_reply_before_consumer_starts() -> None:
    ch = NotifyChannel()
    ch.reply("pre-queued")
    # No blocking should occur; drain-first path.
    start = time.monotonic()
    result = ch.consume(timeout=5.0)
    elapsed = time.monotonic() - start
    assert result.reason == "reply"
    assert result.reply == "pre-queued"
    # Should return almost instantly, not wait for timeout.
    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Producer non-blocking: put() returns instantly
# ---------------------------------------------------------------------------


def test_producer_put_is_non_blocking() -> None:
    ch = NotifyChannel()
    start = time.monotonic()
    ch.dismiss()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05  # must return well under 50ms


# ---------------------------------------------------------------------------
# Idempotent re-consume: settled channel returns same result every call
# ---------------------------------------------------------------------------


def test_idempotent_consume_after_reply() -> None:
    ch = NotifyChannel()
    ch.reply("once")
    r1 = ch.consume(timeout=1.0)
    r2 = ch.consume(timeout=1.0)
    assert r1 == r2
    assert r1.reason == "reply"
    assert r1.reply == "once"


def test_idempotent_consume_after_dismiss() -> None:
    ch = NotifyChannel()
    ch.dismiss()
    r1 = ch.consume(timeout=1.0)
    r2 = ch.consume(timeout=1.0)
    assert r1 == r2
    assert r1.reason == "dismiss"


# ---------------------------------------------------------------------------
# Backstop timeout: no event → returns timeout
# ---------------------------------------------------------------------------


def test_backstop_timeout_when_no_event() -> None:
    ch = NotifyChannel()
    start = time.monotonic()
    result = ch.consume(timeout=0.1)
    elapsed = time.monotonic() - start
    assert result.reason == "timeout"
    assert 0.05 < elapsed < 0.5


# ---------------------------------------------------------------------------
# Cross-thread: consumer unblocks when producer fires from another thread
# ---------------------------------------------------------------------------


def test_consumer_unblocks_on_reply_from_thread() -> None:
    ch = NotifyChannel()
    results: list[NotifyResult] = []

    def consumer() -> None:
        results.append(ch.consume(timeout=5.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    ch.reply("from thread")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert len(results) == 1
    assert results[0].reason == "reply"
    assert results[0].reply == "from thread"


def test_consumer_unblocks_on_dismiss_from_thread() -> None:
    ch = NotifyChannel()
    results: list[NotifyResult] = []

    def consumer() -> None:
        results.append(ch.consume(timeout=5.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    ch.dismiss()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert results[0].reason == "dismiss"
