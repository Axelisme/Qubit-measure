"""OperationHandles — the async-operation Handle/Cancel facet (ADR-0019).

Owns the operation *lifecycle*, independent of how the work executes
(BackgroundRunner) and of hardware *exclusion* (OperationGate). It mints the
operation token (= ``operation_id``) and exposes the three async verbs over it:
``await_outcome`` (off-main blocking wait), ``poll`` (non-blocking), ``cancel``
(async stop request). Settled tokens are retained briefly (LRU) so a late
waiter still returns.

Cross-thread interaction uses a per-operation ``OperationChannel`` (ADR-0025):
a single ordered FIFO carrying typed events (Settled / Message / Stop).
This replaces the ADR-0023 FeedbackInbox + poll-loop await-combine: signal
ordering is guaranteed by the single queue (race-free by construction), and
``Queue.get(timeout)`` wakes immediately on enqueue (no 2s poll delay).

Composition (ADR-0019): a hardware op (run / device / connect) takes a handle
here AND registers an ``OperationGate`` exclusion under the *same* token; an
analyze / interactive op takes only a handle (no exclusion). The terminal path
settles the handle here, then frees the exclusion (if any).

Session-core (``gui/session``): the handle lifecycle carries zero
operation-kind knowledge, so it is shared verbatim by every session-driving
app; each app keeps its own ``OperationGate`` exclusion policy.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Sentinel for "the background work produced no result value" (distinct from a
# real ``None`` result). Lives here (session-core, Qt-free) so OperationRunner /
# run-policy terminal interpretation can compare against it without importing the
# Qt-coupled shared executor (``gui.background`` re-exports this same object, so
# its identity is shared across the executor and the operation layer).
NO_RESULT: Any = object()

# Upper bound on how many settled operations are retained so that
# ``await_outcome`` can return immediately for an operation that finished before
# the caller began waiting. LRU-evicted past this — eviction only degrades a
# late waiter to "treat as already-done" (still correct, just non-blocking and
# losing the recorded outcome, which then reads as a default 'finished').
_DONE_EVENT_LIMIT = 32

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
    """The result of one ``await_outcome`` call (ADR-0025).

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


# ---------------------------------------------------------------------------
# OperationChannel — per-operation single ordered event queue (ADR-0025)
# ---------------------------------------------------------------------------

# Cancel hook: called by stop() after enqueueing Stop; encapsulates how
# cancellation actually interrupts the work (run/device = stop_event.set;
# interactive = cancel_interactive; uncancellable = None).
CancelHook = Callable[[], None]


@dataclass(frozen=True)
class Settled:
    """The operation reached its terminal state (set by the worker)."""

    outcome: OperationOutcome


@dataclass(frozen=True)
class Message:
    """A user nudge: op continues running; agent receives user_feedback."""

    text: str


@dataclass(frozen=True)
class Stop:
    """A cancel request with an optional reason string."""

    reason: str | None


ChannelEvent = Settled | Message | Stop


def _join_reasons(a: str | None, b: str | None) -> str | None:
    """None-safe newline join; returns None when both are None."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return f"{a}\n{b}"


class OperationChannel:
    """Per-operation ordered event FIFO (ADR-0025).

    Producer interface (non-blocking, any thread):
    - ``settle(outcome)`` — set-once terminal; idempotent.
    - ``message(text)`` — nudge; ignored when text is blank.
    - ``stop(reason)`` — enqueue Stop FIRST, then invoke cancel_hook.
      Order is critical: the Settled event from the hook lands after Stop,
      so the consumer folds reason correctly (see ADR-0025 §stop ordering).

    Consumer interface (single consumer, blocking with timeout):
    - ``consume(timeout)`` — fold events into an AwaitResult.
    - ``settled_outcome()`` — non-blocking poll (returns _settled or None).
    """

    def __init__(self, cancel_hook: CancelHook | None = None) -> None:
        self._q: queue.Queue[ChannelEvent] = queue.Queue()
        self._cancel_hook = cancel_hook
        # Set-once terminal outcome; guarded by _lock for producer idempotency.
        self._settled: OperationOutcome | None = None
        self._lock: threading.Lock = threading.Lock()
        # Consumer-private latch: accumulated Stop reasons from past Stop events.
        self._pending_stop_reason: str | None = None

    @property
    def can_cancel(self) -> bool:
        """True when a cancel hook is registered (this op is cancellable).

        Pure read — never triggers the hook; used by
        OperationHandles.has_cancel_hook() to gate the 'Send & Stop' button
        without any op-kind knowledge in this layer (ADR-0025 §Stop-gating).
        """
        return self._cancel_hook is not None

    # ------------------------------------------------------------------
    # Producer interface (non-blocking)
    # ------------------------------------------------------------------

    def settle(self, outcome: OperationOutcome) -> None:
        """Mark the channel terminal (set-once, idempotent).

        Lock is held only for the set-once guard; the enqueue happens outside
        so the lock is never held while blocking.
        """
        with self._lock:
            if self._settled is not None:
                return  # idempotent: already settled
            self._settled = outcome
        self._q.put(Settled(outcome))

    def message(self, text: str) -> None:
        """Enqueue a nudge; blank text is ignored (same rule as old FeedbackInbox)."""
        if not text.strip():
            return
        self._q.put(Message(text))

    def stop(self, reason: str | None) -> None:
        """Request cancellation: enqueue Stop BEFORE invoking cancel_hook.

        The ordering guarantee ensures that if the cancel_hook causes the
        worker to settle immediately (interactive direct-settle), the Settled
        event arrives *after* Stop in the queue, so consume() sees Stop first
        and latches the reason before folding the Settled.
        """
        self._q.put(Stop(reason))
        if self._cancel_hook is not None:
            try:
                self._cancel_hook()
            except Exception:
                logger.exception("operation cancel hook failed")

    # ------------------------------------------------------------------
    # Consumer interface (single consumer assumed)
    # ------------------------------------------------------------------

    def settled_outcome(self) -> OperationOutcome | None:
        """Non-blocking poll: return the settled outcome if known, else None."""
        return self._settled

    def consume(self, timeout: float) -> AwaitResult:
        """Block until a returnable event arrives or ``timeout`` elapses.

        Events are consumed in strict arrival order — the queue's total order IS
        the resolution of every race (ADR-0025). One call returns at the first
        returnable event:
        - ``Settled`` → completed (feedback only when status=='cancelled' and a
          Stop reason was latched).
        - ``Stop(reason)`` → latch reason into _pending_stop_reason, keep going.
        - ``Message(text)`` → if a cancel is in progress (_pending_stop_reason
          set): fold into the reason, keep going; else return user_feedback
          (non-terminal — a pure nudge, op continues, agent may re-await).
        - timeout → return timeout.

        Already-drained events are processed non-blockingly first, so a nudge
        queued *between* two awaits is delivered (not silently dropped), and a
        terminally-settled channel returns idempotently without blocking.
        """
        deadline = time.monotonic() + timeout

        while True:
            # 1) Drain immediately-available events in arrival order. A pending
            #    nudge (enqueued between awaits) surfaces here rather than being
            #    eaten by the terminal fast-path below.
            try:
                event = self._q.get_nowait()
            except queue.Empty:
                event = None
            if event is not None:
                result = self._process_event(event)
                if result is not None:
                    return result
                continue

            # 2) Queue momentarily empty. If terminally settled, return now —
            #    idempotent re-consume of a finished op never blocks.
            if self._settled is not None:
                return self._make_completed(self._settled)

            # 3) Not settled and nothing queued: block for the next event.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return AwaitResult(reason="timeout")
            try:
                event = self._q.get(timeout=remaining)
            except queue.Empty:
                return AwaitResult(reason="timeout")
            result = self._process_event(event)
            if result is not None:
                return result
            # else: latched a Stop / folded a Message — loop.

    def _process_event(self, event: ChannelEvent) -> AwaitResult | None:
        """Fold one event. Returns an AwaitResult to return now, or None to keep
        consuming (a latched Stop or a folded-in Message)."""
        if isinstance(event, Settled):
            return self._make_completed(event.outcome)
        if isinstance(event, Stop):
            self._pending_stop_reason = _join_reasons(
                self._pending_stop_reason, event.reason
            )
            return None
        # Message: fold into the reason while a cancel is in progress, else a
        # pure non-terminal nudge.
        if self._pending_stop_reason is not None:
            self._pending_stop_reason = _join_reasons(
                self._pending_stop_reason, event.text
            )
            return None
        return AwaitResult(reason="user_feedback", feedback=event.text)

    def _make_completed(self, outcome: OperationOutcome) -> AwaitResult:
        """Build a completed AwaitResult; attach pending_stop_reason only for
        cancelled outcomes (a 'Send & Stop' surfaces as {cancelled, feedback})."""
        feedback: str | None = None
        if outcome.status == "cancelled" and self._pending_stop_reason is not None:
            feedback = self._pending_stop_reason
        return AwaitResult(reason="completed", outcome=outcome, feedback=feedback)


# ---------------------------------------------------------------------------
# OperationHandles — channel registry (ADR-0019 / ADR-0025)
# ---------------------------------------------------------------------------


class OperationHandles:
    """Async-operation handles keyed by token: create / settle / await / poll /
    cancel (ADR-0019). Each operation gets its own ``OperationChannel`` (ADR-0025)
    for cross-thread interaction; no shared FeedbackInbox or poll-loop."""

    def __init__(self) -> None:
        self._next_token = 1
        # Live (pending) operations: token -> OperationChannel.
        self._live: dict[int, OperationChannel] = {}
        # Settled operations, retained briefly (LRU) so a caller awaiting after
        # settle still returns the outcome immediately.
        self._done: OrderedDict[int, OperationChannel] = OrderedDict()

    def create(self, cancel_hook: CancelHook | None = None) -> int:
        """Mint an operation token and open its channel (pending).

        ``cancel_hook`` is the callable invoked by ``stop()`` after enqueueing
        the Stop event; it encapsulates how cancellation interrupts the work:
        - run / device setup: ``stop_event.set``
        - interactive analyze: ``cancel_interactive`` direct-settle callable
        - uncancellable (connect, FIT-analyze): ``None``

        Returns the token (operation_id).
        """
        token = self._next_token
        self._next_token += 1
        self._live[token] = OperationChannel(cancel_hook)
        logger.debug("operation create: token=%d", token)
        return token

    def settle(self, token: int, outcome: OperationOutcome) -> None:
        """Mark the operation terminal: settle its channel and retain (LRU).

        The channel is published to ``_done`` BEFORE being retracted from
        ``_live`` so the token is always reachable in at least one dict — there
        is no window where a concurrent ``await_outcome`` / ``poll`` sees it in
        neither and falls through to the default 'finished' (which would
        misreport a cancelled/failed terminal). See ADR-0025.
        """
        ch = self._live.get(token)
        if ch is None:
            # Already settled (idempotent) or never created — still delegate
            # to the retained channel if present (idempotent settle on channel).
            ch = self._done.get(token)
            if ch is not None:
                ch.settle(outcome)
            return
        if outcome.status == "finished":
            logger.info("operation settle: token=%d status=%s", token, outcome.status)
        else:
            logger.warning(
                "operation settle: token=%d status=%s error=%s",
                token,
                outcome.status,
                outcome.error,
            )
        ch.settle(outcome)
        # Publish to _done first, then retract from _live (never "neither").
        self._done[token] = ch
        self._live.pop(token, None)
        # The just-settled token is most-recent, so LRU eviction never drops it.
        while len(self._done) > _DONE_EVENT_LIMIT:
            self._done.popitem(last=False)

    def cancel(self, token: int) -> None:
        """Request the operation stop (via its cancel hook); returns immediately.

        Async notification, not a wait: the caller polls/awaits for the actual
        terminal outcome. A no-op for an unknown/settled token or one with no
        cancel_hook (e.g. connect has no cancellation point).
        """
        ch = self._live.get(token)
        if ch is not None:
            logger.info("operation cancel: token=%d", token)
            ch.stop(None)

    def message(self, token: int, text: str) -> None:
        """Deliver a nudge message to the operation's awaiter (non-terminal).

        A no-op for an unknown/settled token — the message has nowhere to go.
        """
        ch = self._live.get(token)
        if ch is not None:
            ch.message(text)

    def stop(self, token: int, reason: str | None = None) -> None:
        """Request cancel with an optional reason string.

        The reason string surfaces in the AwaitResult.feedback of the
        subsequent completed(cancelled) result (Send & Stop semantic).
        A no-op for an unknown/settled token.
        """
        ch = self._live.get(token)
        if ch is not None:
            logger.info("operation stop: token=%d reason=%r", token, reason)
            ch.stop(reason)

    def cancel_all(self) -> list[int]:
        """Cancel every live operation; return their tokens (for poll/await)."""
        tokens = list(self._live.keys())
        logger.info("operation cancel_all: %d live ops", len(tokens))
        for token in tokens:
            self.cancel(token)
        return tokens

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
        - ``None`` is never returned (kept as a contract break note; all callers
          must handle AwaitResult).

        ADR-0025: the channel's consume() method handles all folding logic.
        A token with no live or retained channel is treated as already-done.
        """
        # Check live first, then retained.
        ch = self._live.get(token)
        if ch is None:
            ch = self._done.get(token)
        if ch is None:
            # Unknown token: treat as already-done (finished).
            return AwaitResult(reason="completed", outcome=OperationOutcome("finished"))
        return ch.consume(timeout)

    def poll(self, token: int) -> OperationOutcome | None:
        """Non-blocking: outcome if settled, None if still pending, default
        'finished' if unknown. Reads the channel's set-once ``_settled`` (the
        source of truth) rather than dict membership, so the brief settle window
        where the token is in both ``_live`` and ``_done`` still reports the
        true outcome instead of a stale 'pending'."""
        ch = self._live.get(token)
        if ch is None:
            ch = self._done.get(token)
        if ch is None:
            return OperationOutcome("finished")
        return ch.settled_outcome()

    def has_cancel_hook(self, token: int) -> bool:
        """Return True when the token's channel has a cancel hook registered.

        Checks _live first, then _done (a token settling between the caller
        reading the active token and this call is still reachable). Returns
        False for unknown tokens — they have no hook by definition.

        Used by Controller.can_cancel_active_operation() to gate the
        'Send & Stop' button without any op-kind knowledge in this layer
        (ADR-0025 §Stop-gating). Pure read — never triggers the hook.
        """
        ch = self._live.get(token)
        if ch is None:
            ch = self._done.get(token)
        if ch is None:
            return False
        return ch.can_cancel

    def live_count(self) -> int:
        """How many operations are live (pending) right now, of any facet —
        run / device / connect AND analyze / interactive. The shutdown confirm
        reads this to decide whether closing will interrupt work (Handles owns
        the lifecycle, so it is the authority on "is anything in progress",
        unlike the gate which only knows hardware exclusions)."""
        return len(self._live)
