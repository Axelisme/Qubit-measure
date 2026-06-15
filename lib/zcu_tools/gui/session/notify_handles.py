"""NotifyHandles — agent-initiated user-prompt channel (Stage 4b, ADR-0025).

Mirrors OperationChannel's four invariants (producer non-blocking put, consumer
bounded queue.Queue.get, set-once latch, drain-first) with an independent event
vocabulary (Reply / Dismiss / Timeout) so notify events never mix with the
operation event set (Settled / Message / Stop).

Threading contract:
  - Producer (reply / dismiss / timeout): main thread only (dialog callbacks,
    QTimer). The set-once latch is guarded by _lock for producer-side idempotency.
  - Consumer (await_result / consume): off-main IO worker thread only.
  - queue.Queue is the sole cross-thread channel — no other shared state is
    written from off-main.

NotifyHandles token space is independent of OperationHandles. Tokens are minted
exclusively on the main thread (notify.open handler), so _next_token needs no lock.
Unknown-token lookups in producer methods are no-ops; in the consumer they return
NotifyResult("dismiss") (treat an absent prompt as user-dismissed).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Retain this many settled notify channels so a very-late consumer still gets
# its answer. LRU-evicted past this limit.
_DONE_LIMIT = 16

# ---------------------------------------------------------------------------
# Event types — vocabulary independent of OperationChannel events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    """User clicked Reply and submitted text (may be an empty string)."""

    text: str


@dataclass(frozen=True)
class Dismiss:
    """User clicked Dismiss or closed the dialog window."""


@dataclass(frozen=True)
class Timeout:
    """QTimer in the dialog fired — dialog's own timeout SSOT (ADR-0025)."""


NotifyEvent = Reply | Dismiss | Timeout


# ---------------------------------------------------------------------------
# NotifyResult — the settled outcome returned to the consumer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NotifyResult:
    """The result of one gui_notify_user prompt, structured for wire folding."""

    reason: Literal["reply", "dismiss", "timeout"]
    reply: str | None = None

    def __post_init__(self) -> None:
        if self.reason == "reply" and self.reply is None:
            # An empty string is a valid reply; only a missing field is wrong.
            raise ValueError("NotifyResult with reason='reply' must include reply text")


# ---------------------------------------------------------------------------
# NotifyChannel — per-prompt ordered event FIFO
# ---------------------------------------------------------------------------


class NotifyChannel:
    """Single-prompt cross-thread channel (mirroring OperationChannel invariants).

    Producer methods (reply / dismiss / timeout) are called on the main thread;
    consume() blocks the off-main worker. set-once: the first terminal event wins;
    subsequent puts are silently ignored inside the lock-guarded _settled flag.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[NotifyEvent] = queue.Queue()
        # Set-once latch; guarded by _lock for producer idempotency.
        self._settled: NotifyResult | None = None
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Producer interface (non-blocking, main thread)
    # ------------------------------------------------------------------

    def reply(self, text: str) -> None:
        """User submitted a reply (possibly an empty string)."""
        with self._lock:
            if self._settled is not None:
                return
            self._settled = NotifyResult("reply", text)
        self._q.put(Reply(text))

    def dismiss(self) -> None:
        """User dismissed the prompt (Dismiss button or window close)."""
        with self._lock:
            if self._settled is not None:
                return
            self._settled = NotifyResult("dismiss")
        self._q.put(Dismiss())

    def timeout(self) -> None:
        """QTimer in the dialog fired (dialog is the timeout SSOT — ADR-0025)."""
        with self._lock:
            if self._settled is not None:
                return
            self._settled = NotifyResult("timeout")
        self._q.put(Timeout())

    # ------------------------------------------------------------------
    # Consumer interface (blocking with timeout, off-main thread)
    # ------------------------------------------------------------------

    def consume(self, timeout: float) -> NotifyResult:
        """Block until an event arrives or the backstop timeout elapses.

        Drain-first: processes any already-queued events without blocking before
        falling through to the blocking get. A terminally-settled channel returns
        idempotently without blocking (same as OperationChannel's fast-path).

        The backstop timeout is longer than the dialog's QTimer so the dialog
        fires first, putting a Timeout event into the queue, and this consume()
        reads it rather than timing out independently. On backstop expiry the
        method still returns NotifyResult("timeout") for safety.
        """
        deadline = time.monotonic() + timeout

        while True:
            # Drain already-queued events first (drain-first, ADR-0025).
            try:
                event = self._q.get_nowait()
            except queue.Empty:
                event = None

            if event is not None:
                return self._event_to_result(event)

            # Fast-path: already settled but queue is empty (race between
            # producer setting _settled and the put; the lock guarantees _settled
            # is set before put, so if we read _settled here the event is either
            # already drained above or about to appear — check once more).
            if self._settled is not None:
                return self._settled

            # Block for the next event.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return NotifyResult("timeout")
            try:
                event = self._q.get(timeout=remaining)
            except queue.Empty:
                return NotifyResult("timeout")
            return self._event_to_result(event)

    @staticmethod
    def _event_to_result(event: NotifyEvent) -> NotifyResult:
        if isinstance(event, Reply):
            return NotifyResult("reply", event.text)
        if isinstance(event, Dismiss):
            return NotifyResult("dismiss")
        # Timeout
        return NotifyResult("timeout")


# ---------------------------------------------------------------------------
# NotifyHandles — channel registry
# ---------------------------------------------------------------------------


class NotifyHandles:
    """Registry of active notify prompts; token space is independent of
    OperationHandles (tokens are minted in sequence but the two registries
    never share channels or query each other).

    All mint / producer calls must happen on the main thread (notify.open and
    dialog callbacks). The consumer (await_result) is off-main.
    """

    def __init__(self) -> None:
        # Minting is main-thread-only, so no lock needed for the counter.
        self._next_token: int = 1
        self._live: dict[int, NotifyChannel] = {}
        # Retain recently-settled channels for very-late consumers.
        self._done: OrderedDict[int, NotifyChannel] = OrderedDict()

    # ------------------------------------------------------------------
    # Mint (main thread)
    # ------------------------------------------------------------------

    def open(self) -> int:
        """Mint a new notify token and open its channel. Main thread only."""
        token = self._next_token
        self._next_token += 1
        self._live[token] = NotifyChannel()
        logger.debug("notify open: token=%d", token)
        return token

    # ------------------------------------------------------------------
    # Producer interface (main thread)
    # ------------------------------------------------------------------

    def reply(self, token: int, text: str) -> None:
        """Deliver a reply string to the prompt. No-op for unknown token."""
        ch = self._live.get(token)
        if ch is None:
            return
        ch.reply(text)
        self._settle(token, ch)

    def dismiss(self, token: int) -> None:
        """Mark the prompt as dismissed. No-op for unknown token."""
        ch = self._live.get(token)
        if ch is None:
            return
        ch.dismiss()
        self._settle(token, ch)

    def timeout(self, token: int) -> None:
        """Mark the prompt as timed-out (QTimer callback). No-op for unknown token."""
        ch = self._live.get(token)
        if ch is None:
            return
        ch.timeout()
        self._settle(token, ch)

    def _settle(self, token: int, ch: NotifyChannel) -> None:
        """Promote a token from _live to _done (LRU). Main-thread only."""
        self._done[token] = ch
        self._live.pop(token, None)
        # LRU eviction: the just-settled token is last, so eviction never drops it.
        while len(self._done) > _DONE_LIMIT:
            self._done.popitem(last=False)
        logger.debug("notify settle: token=%d", token)

    # ------------------------------------------------------------------
    # Consumer interface (off-main thread)
    # ------------------------------------------------------------------

    def await_result(self, token: int, timeout: float) -> NotifyResult:
        """Block until the prompt settles or the backstop timeout elapses.

        Looks up _live first, then _done (for a prompt that settled before the
        consumer started waiting). An unknown token returns dismiss (treat as
        already-dismissed / closed prompt).
        """
        ch = self._live.get(token)
        if ch is None:
            ch = self._done.get(token)
        if ch is None:
            logger.warning("notify await_result: unknown token=%d → dismiss", token)
            return NotifyResult("dismiss")
        return ch.consume(timeout)
