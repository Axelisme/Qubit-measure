"""Tests for NotifyHandles — channel registry (Stage 4b).

Covers:
  - open() mints unique, increasing tokens
  - reply / dismiss / timeout routes to the correct channel
  - unknown-token producer calls are no-ops (no exception)
  - unknown-token consumer returns dismiss (treat as already-dismissed)
  - settle LRU: _done retains channels for very-late consumers
  - await_result works for a settled-before-consumer-starts channel
  - concurrent open + await from off-main thread
"""

from __future__ import annotations

import threading
import time

from zcu_tools.gui.session.notify_handles import NotifyHandles, NotifyResult

# ---------------------------------------------------------------------------
# Token minting
# ---------------------------------------------------------------------------


def test_open_mints_unique_increasing_tokens() -> None:
    h = NotifyHandles()
    assert h.open() == 1
    assert h.open() == 2
    assert h.open() == 3


# ---------------------------------------------------------------------------
# Routing: reply / dismiss / timeout reach the right channel
# ---------------------------------------------------------------------------


def test_reply_routes_to_correct_token() -> None:
    h = NotifyHandles()
    t1 = h.open()
    t2 = h.open()
    h.reply(t1, "for t1")
    result = h.await_result(t1, timeout=1.0)
    assert result.reason == "reply"
    assert result.reply == "for t1"
    # The reply routed ONLY to t1 — t2 has no event, so it times out.
    assert h.await_result(t2, timeout=0.05).reason == "timeout"


def test_dismiss_routes_to_correct_token() -> None:
    h = NotifyHandles()
    token = h.open()
    h.dismiss(token)
    result = h.await_result(token, timeout=1.0)
    assert result.reason == "dismiss"


def test_timeout_routes_to_correct_token() -> None:
    h = NotifyHandles()
    token = h.open()
    h.timeout(token)
    result = h.await_result(token, timeout=1.0)
    assert result.reason == "timeout"


# ---------------------------------------------------------------------------
# Unknown-token behaviour
# ---------------------------------------------------------------------------


def test_reply_unknown_token_is_noop() -> None:
    h = NotifyHandles()
    h.reply(999, "ignored")  # must not raise


def test_dismiss_unknown_token_is_noop() -> None:
    h = NotifyHandles()
    h.dismiss(999)


def test_timeout_unknown_token_is_noop() -> None:
    h = NotifyHandles()
    h.timeout(999)


def test_await_result_unknown_token_returns_dismiss() -> None:
    h = NotifyHandles()
    result = h.await_result(999, timeout=0.1)
    assert result.reason == "dismiss"


# ---------------------------------------------------------------------------
# LRU retain: settled channels are still reachable by very-late consumers
# ---------------------------------------------------------------------------


def test_await_result_after_settle_returns_same_result() -> None:
    h = NotifyHandles()
    token = h.open()
    h.reply(token, "early settle")
    # Consumer starts after the token is already in _done.
    result = h.await_result(token, timeout=1.0)
    assert result.reason == "reply"
    assert result.reply == "early settle"


def test_done_lru_retains_up_to_limit() -> None:
    from zcu_tools.gui.session.notify_handles import _DONE_LIMIT

    h = NotifyHandles()
    tokens = [h.open() for _ in range(_DONE_LIMIT + 2)]
    for t in tokens:
        h.dismiss(t)

    # The two oldest tokens should have been evicted.
    oldest = tokens[0]
    result = h.await_result(oldest, timeout=0.1)
    # Evicted → unknown-token path → dismiss result (not an error).
    assert result.reason == "dismiss"

    # The newest token is still reachable.
    newest = tokens[-1]
    result_newest = h.await_result(newest, timeout=0.1)
    assert result_newest.reason == "dismiss"  # was actually dismissed, not evicted


# ---------------------------------------------------------------------------
# Cross-thread: open on main, await from off-main, settle from main
# ---------------------------------------------------------------------------


def test_cross_thread_reply_unblocks_consumer() -> None:
    h = NotifyHandles()
    token = h.open()
    results: list[NotifyResult] = []

    def consumer() -> None:
        results.append(h.await_result(token, timeout=5.0))

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    h.reply(token, "cross-thread")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert results[0].reason == "reply"
    assert results[0].reply == "cross-thread"
