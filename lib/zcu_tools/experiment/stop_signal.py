"""Cooperative stop requests shared by experiment run callers."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CancelResult:
    """Acknowledged scope request, not a promise that work has drained.

    cancelled means the owner accepted the request, not that work drained.
    Token owners atomically match/set the target. Workflow bare script Stop
    instead records and latches intent before signaling a still-live host;
    acceptance survives host exit before owner finalization. reason is scope
    cancellation requested, scope not active, sequence mismatch, or the recipe
    admission refusal no run in progress. ambiguous refuses multiple matches.
    active_path is the unique live path at the check, stopping at the first fork.
    ScriptHost root acceptance instead keeps the root path in its cached receipt;
    acceptance does not wait for the loop notification or authored finally.
    candidates lists full matching paths only for an ambiguous refusal.
    """

    cancelled: bool
    reason: str
    active_path: str
    candidates: list[str] = field(default_factory=list)


class StopSignal:
    """Thread-safe downward stop requests with opaque reason text.

    event, when supplied, is borrowed as this token's stop bit. Descendants
    own independent events and inherit an existing request at birth. Scope
    owns names and selectors. Stopping a child never stops its parent.
    Release detaches the token without reporting data completeness or errors.
    """

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event if event is not None else threading.Event()
        self._reason: str | None = None
        self._lock = threading.RLock()
        self._parent: StopSignal | None = None
        self._children: dict[StopSignal, None] = {}

    def is_set(self) -> bool:
        """Return whether this token has a pending stop request."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Return the last requested stop's opaque text, or None if unspecified.

        Clearing the stop bit or releasing this token does not clear the reason.
        The framework stores and forwards this text without interpreting it.
        """
        with self._lock:
            return self._reason

    def set(self, reason: str | None = None) -> None:
        """Stop this subtree, storing reason verbatim; never stop its parent.

        reason is optional free text. Each request replaces the stored reason
        on this token and attached descendants, including when already stopped.
        """
        with self._lock:
            self._reason = reason
            self._event.set()
            for child in self._children:
                child.set(reason=reason)

    def child(self) -> StopSignal:
        """Attach a token inheriting any current stop request and reason.

        Future requests travel down immediately. Call release after work drains.
        """
        with self._lock:
            child = StopSignal()
            child._lock = self._lock
            child._parent = self
            if self.is_set():
                child.set(reason=self._reason)
            self._children[child] = None
            return child

    def release(self) -> None:
        """Detach from the parent; repeated release is a no-op.

        Preserve this token's stop bit and reason. Future parent requests no
        longer reach it. Release never changes the parent's stop state.
        """
        with self._lock:
            parent = self._parent
            if parent is not None:
                parent._children.pop(self)
                self._parent = None

    def clear_stop(self) -> None:
        """Clear only this token's event, retaining its reason and child state."""
        self._event.clear()

    @property
    def event(self) -> threading.Event:
        """Borrow this token's event for cooperative waits."""
        return self._event
