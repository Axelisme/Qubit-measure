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

ADR-0023 extension: ``await_outcome`` supports a second wakeup source via
``FeedbackInbox`` — a thread-safe queue of user-feedback strings. A pending wait
returns early with reason='user_feedback' when a feedback arrives; the operation
handle is left unsettled (the operation keeps running). This is a non-terminal
return; the same handle can be awaited again.
"""

from __future__ import annotations

import logging
import queue
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

# Poll-tick duration for the feedback-aware wait. Short enough for responsive
# feedback delivery; long enough not to busy-spin. The outer timeout is honoured
# with resolution at this granularity.
_AWAIT_TICK_SECONDS: float = 2.0

OperationStatus = Literal["pending", "finished", "failed", "cancelled"]

# Reason tag for AwaitResult: what caused await_outcome to return.
AwaitReason = Literal["completed", "user_feedback", "timeout"]


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


@dataclass(frozen=True)
class AwaitResult:
    """The result of one ``await_outcome`` call (ADR-0023).

    ``reason`` distinguishes the three return paths:
    - ``'completed'``: the operation settled (terminal). ``outcome`` is set; a
      cancelled outcome may also carry ``feedback`` (a user "Send & Stop" message
      folded into the cancellation, so the agent gets one {cancelled, feedback}).
    - ``'user_feedback'``: a feedback string arrived before the op settled (non-
      terminal). ``feedback`` is set; the operation is still running and the handle
      can be awaited again.
    - ``'timeout'``: the bounded wait elapsed without the op settling or feedback
      arriving (non-terminal). The operation is still running.
    """

    reason: AwaitReason
    outcome: OperationOutcome | None = None
    feedback: str | None = None

    def __post_init__(self) -> None:
        if self.reason == "completed" and self.outcome is None:
            raise ValueError("AwaitResult with reason='completed' must have outcome")
        if self.reason == "user_feedback" and not self.feedback:
            raise ValueError(
                "AwaitResult with reason='user_feedback' must have feedback"
            )


class FeedbackInbox:
    """Thread-safe inbox for user-feedback strings (ADR-0023).

    Written on the Qt main thread (GUI widget); read/drained on an IO worker
    thread (``await_outcome``). The ``notify_event`` fires on every ``post``
    so that a blocked ``await_outcome`` poll-tick is woken early.

    Session-scoped: inbox is never persisted; cleared on a new session context.
    Not stored in State (State main-thread write invariant).
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue[str] = queue.SimpleQueue()
        # Set by post(), cleared by drain(); allows await to detect new feedback
        # without draining prematurely (notify_event is level-triggered: stays set
        # until drained).
        self.notify_event: threading.Event = threading.Event()

    def post(self, text: str) -> None:
        """Enqueue a feedback string and wake any pending await. Main-thread only."""
        if not text.strip():
            return  # ignore whitespace-only feedback
        self._q.put(text)
        self.notify_event.set()

    def drain(self) -> list[str]:
        """Drain all pending feedback strings; clear the notify_event. Thread-safe."""
        msgs: list[str] = []
        while True:
            try:
                msgs.append(self._q.get_nowait())
            except queue.Empty:
                break
        # Clear only after draining; a concurrent post between the loop and here
        # would re-set the event, which is correct (the next poll tick picks it up).
        self.notify_event.clear()
        return msgs

    def has_pending(self) -> bool:
        """True if at least one feedback string is waiting. Thread-safe best-effort."""
        return not self._q.empty()


class OperationHandles:
    """Async-operation handles keyed by token: create / settle / await / poll /
    cancel (ADR-0019). The completion ``Event`` lets a blocking off-main handler
    (``operation.await``) wait thread-safely without touching main-thread-owned
    state. The optional ``stop_event`` is the worker's own cancellation flag,
    passed at ``create`` time: ``cancel`` sets it (a pure data handle, never a
    callback), and the worker self-translates "stopped" into a ``cancelled``
    outcome."""

    def __init__(self, feedback_inbox: FeedbackInbox | None = None) -> None:
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
        # Optional shared feedback inbox (ADR-0023). When set, await_outcome
        # returns early with reason='user_feedback' on the next poll tick where
        # feedback is present. None disables the second wakeup source (for tests
        # or apps that do not use feedback).
        self._feedback_inbox: FeedbackInbox | None = feedback_inbox

    def set_feedback_inbox(self, inbox: FeedbackInbox) -> None:
        """Wire the shared feedback inbox (called by app wiring after construction)."""
        self._feedback_inbox = inbox

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

    def _completed_with_feedback(self, outcome: OperationOutcome) -> AwaitResult:
        """Build a 'completed' AwaitResult, folding any pending user feedback into
        a CANCELLED outcome so a "Send & Stop" surfaces as a single
        {status:cancelled, feedback} rather than two events. Feedback is attached
        ONLY for a cancelled outcome — a finished op leaves nudges in the inbox for
        the next await to deliver as a standalone user_feedback."""
        feedback: str | None = None
        inbox = self._feedback_inbox
        if outcome.status == "cancelled" and inbox is not None and inbox.has_pending():
            msgs = inbox.drain()
            if msgs:
                feedback = "\n".join(msgs)
        return AwaitResult(reason="completed", outcome=outcome, feedback=feedback)

    def await_outcome(self, token: int, timeout: float) -> AwaitResult | None:
        """Block until the token settles or a wakeup condition fires.

        Thread-safe; for off-main blocking handlers. Returns:
        - ``AwaitResult(reason='completed', outcome=<outcome>)`` when the
          operation settles (terminal). Never None on this path.
        - ``AwaitResult(reason='user_feedback', feedback=<text>)`` when a user
          feedback string arrives before the op settles (non-terminal). The
          operation is still running; the caller may re-await the same token.
        - ``AwaitResult(reason='timeout', ...)`` when the bounded ``timeout``
          elapses without completion or feedback (non-terminal).
        - ``None`` is never returned (kept as a contract break to distinguish from
          the old API during the migration; all callers must handle AwaitResult).

        ADR-0023: the feedback wakeup path is intentionally non-terminal — it
        does NOT settle the handle. The operation continues running and the caller
        (agent) decides whether to cancel or continue awaiting.

        A token with no live or retained Event is treated as already-done
        (returns a completed 'finished' outcome) so callers never hang on an
        operation that finished before they began waiting.
        """
        # Fast-path: already settled before this call.
        live = self._events.get(token)
        if live is None:
            retained = self._done.get(token)
            outcome = (
                retained[1] if retained is not None else OperationOutcome("finished")
            )
            # Folds pending feedback into a cancelled outcome (Send & Stop).
            return self._completed_with_feedback(outcome)

        inbox = self._feedback_inbox
        stop_event = self._stop_events.get(token)

        def _cancel_in_progress() -> bool:
            # A cancel was requested (stop_event set): hold any feedback so it
            # folds into the imminent cancelled settle (one {cancelled, feedback})
            # instead of returning early as a standalone user_feedback nudge.
            return stop_event is not None and stop_event.is_set()

        # --- pre-enter check: drain feedback accumulated between waits --------
        # Feedback sent between two awaits is delivered on the very next entry
        # rather than waiting a full tick — UNLESS a cancel is in progress, in
        # which case it belongs to the cancellation (folded in on settle below).
        if inbox is not None and inbox.has_pending() and not _cancel_in_progress():
            msgs = inbox.drain()
            if msgs:
                feedback_text = "\n".join(msgs)
                logger.debug(
                    "await_outcome: pre-enter feedback for token=%d: %r",
                    token,
                    feedback_text,
                )
                return AwaitResult(reason="user_feedback", feedback=feedback_text)

        # --- poll-loop: check operation + feedback each tick ------------------
        # Short-tick poll (not two parallel threads) avoids joining a companion
        # thread that may outlive the caller; outer timeout honoured to one tick.
        import time

        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return AwaitResult(reason="timeout")

            tick = min(_AWAIT_TICK_SECONDS, remaining)

            # Wake early when the operation settles OR when the tick elapses.
            settled = live.wait(timeout=tick)

            if settled:
                retained = self._done.get(token)
                outcome = (
                    retained[1]
                    if retained is not None
                    else OperationOutcome("finished")
                )
                # Folds pending feedback into a cancelled outcome (Send & Stop).
                return self._completed_with_feedback(outcome)

            # Feedback nudge — only when NOT cancelling (a cancel-in-progress
            # holds the feedback for the cancelled settle handled above).
            if inbox is not None and inbox.has_pending() and not _cancel_in_progress():
                msgs = inbox.drain()
                if msgs:
                    feedback_text = "\n".join(msgs)
                    logger.debug(
                        "await_outcome: feedback wakeup for token=%d: %r",
                        token,
                        feedback_text,
                    )
                    return AwaitResult(reason="user_feedback", feedback=feedback_text)

            # Neither settled nor feedback (or a cancel is pending) — continue.

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
