"""OperationHandles — the async-operation Handle/Cancel facet (ADR-0019).

Owns the operation *lifecycle*, independent of how the work executes
(BackgroundService) and of hardware *exclusion* (OperationGate). It mints the
operation token (= ``operation_id``) and exposes the three async verbs over it:
``await_outcome`` (off-main blocking wait), ``poll`` (non-blocking), ``cancel``
(async stop request — sets the worker's ``stop_event``). Settled tokens are
retained briefly (LRU) so a late waiter still returns.

Composition (ADR-0019): a hardware op (run / device / connect) takes a handle
here AND registers an ``OperationGate`` exclusion under the *same* token; an
analyze / interactive op takes only a handle (no exclusion) — it no longer fakes
a never-conflicting lease just to get an async handle. The terminal path settles
the handle here, then frees the exclusion (if any).

Session-core (``gui/session``): the handle lifecycle carries zero operation-kind
knowledge, so it is shared verbatim by every session-driving app; each app keeps
its own ``OperationGate`` exclusion policy.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Upper bound on how many settled operations are retained so that
# ``await_outcome`` can return immediately for an operation that finished before
# the caller began waiting. LRU-evicted past this — eviction only degrades a
# late waiter to "treat as already-done" (still correct, just non-blocking and
# losing the recorded outcome, which then reads as a default 'finished').
_DONE_EVENT_LIMIT = 32

OperationStatus = Literal["pending", "finished", "failed", "cancelled"]


@dataclass(frozen=True)
class OperationOutcome:
    """The terminal result of an async operation, as a neutral value.

    Carries only success/failure/cancellation + an error message — never a
    result payload (run results / soc handles are read via their own snapshot /
    query paths, not through the operation handle). ``OperationHandles`` never
    interprets ``status``; it is carried verbatim to the awaiter.
    """

    status: OperationStatus
    error: str | None = None


class OperationHandles:
    """Async-operation handles keyed by token: create / settle / await / poll /
    cancel (ADR-0019). The completion ``Event`` lets a blocking off-main handler
    (``operation.await``) wait thread-safely without touching main-thread-owned
    state. The optional ``stop_event`` is the worker's own cancellation flag,
    passed at ``create`` time: ``cancel`` sets it (a pure data handle, never a
    callback), and the worker self-translates "stopped" into a ``cancelled``
    outcome."""

    def __init__(self) -> None:
        self._next_token = 1
        # Live (pending) operations: token -> not-yet-set completion Event.
        self._events: dict[int, threading.Event] = {}
        # Live operations' worker stop_event (None when the operation has no
        # cancellation point, e.g. a blocking connect — cancel is a no-op there).
        self._stop_events: dict[int, threading.Event | None] = {}
        # Settled operations, retained briefly (LRU) so a caller awaiting after
        # settle still returns the outcome immediately. The Event stays set.
        self._done: OrderedDict[int, tuple[threading.Event, OperationOutcome]] = (
            OrderedDict()
        )

    def create(self, stop_event: threading.Event | None = None) -> int:
        """Mint an operation token and open its handle (pending). ``stop_event``
        is the worker's own cancellation flag (None when the op has no
        cancellation point); ``cancel`` sets it. Returns the token (operation_id).
        """
        token = self._next_token
        self._next_token += 1
        self._events[token] = threading.Event()
        self._stop_events[token] = stop_event
        # DEBUG: high-frequency bookkeeping — every async op (run/device/connect/
        # analyze) mints a token here, so this is the canonical "op opened" marker.
        logger.debug("operation create: token=%d", token)
        return token

    def settle(self, token: int, outcome: OperationOutcome) -> None:
        """Mark the operation terminal: store outcome, set Event, retain (LRU)."""
        evt = self._events.pop(token, None)
        self._stop_events.pop(token, None)
        if evt is None:
            return  # never created, or already settled
        # INFO: terminal lifecycle marker. A non-finished outcome carries the
        # error, so log it at WARNING with the message to make failures visible
        # without trawling DEBUG.
        if outcome.status == "finished":
            logger.info("operation settle: token=%d status=%s", token, outcome.status)
        else:
            logger.warning(
                "operation settle: token=%d status=%s error=%s",
                token,
                outcome.status,
                outcome.error,
            )
        evt.set()
        self._done[token] = (evt, outcome)
        while len(self._done) > _DONE_EVENT_LIMIT:
            self._done.popitem(last=False)

    def cancel(self, token: int) -> None:
        """Request the operation stop (set its stop_event); returns immediately.

        Async notification, not a wait: the caller polls/awaits for the actual
        terminal outcome. A no-op for an unknown/settled token or one with no
        stop_event (a connect has no cancellation point — it runs to completion
        and shutdown falls back to a timeout force-close).
        """
        stop_event = self._stop_events.get(token)
        if stop_event is not None:
            logger.info("operation cancel: token=%d", token)
            stop_event.set()

    def cancel_all(self) -> list[int]:
        """Cancel every live operation; return their tokens (for poll/await)."""
        tokens = list(self._events.keys())
        logger.info("operation cancel_all: %d live ops", len(tokens))
        for token in tokens:
            self.cancel(token)
        return tokens

    def await_outcome(self, token: int, timeout: float) -> OperationOutcome | None:
        """Block until the token settles; return its outcome.

        Thread-safe; for off-main blocking handlers. A token with no live or
        retained Event is treated as already-done (returns a default 'finished'
        outcome) so callers never hang on an operation that finished before they
        began waiting. Returns None only on timeout while still pending.
        """
        live = self._events.get(token)
        if live is not None:
            if not live.wait(timeout=timeout):
                return None
            # Woken by settle(), which moved the token into _done with its outcome.
            retained = self._done.get(token)
            return retained[1] if retained is not None else OperationOutcome("finished")
        retained = self._done.get(token)
        if retained is not None:
            return retained[1]
        # Unknown / evicted: treat as already finished.
        return OperationOutcome("finished")

    def poll(self, token: int) -> OperationOutcome | None:
        """Non-blocking: outcome if settled (or unknown), None if still pending."""
        if token in self._events:
            return None
        retained = self._done.get(token)
        if retained is not None:
            return retained[1]
        return OperationOutcome("finished")

    def live_count(self) -> int:
        """How many operations are live (pending) right now, of any facet —
        run / device / connect AND analyze / interactive. The shutdown confirm
        reads this to decide whether closing will interrupt work (Handles owns
        the lifecycle, so it is the authority on "is anything in progress",
        unlike the gate which only knows hardware exclusions)."""
        return len(self._events)
