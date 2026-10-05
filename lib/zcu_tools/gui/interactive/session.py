"""Committed interactive state shared by GUI application owners."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Generic, TypeVar

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.ports import OwnerScheduler

S = TypeVar("S")
logger = logging.getLogger(__name__)


class Session(Generic[S]):
    """Owner-loop state, detached snapshots, atomic commits and subscriptions.

    State is small; bulk read-only analysis inputs belong to the plugin, outside
    this object. Actions calculate a full replacement before publication.
    """

    def __init__(self, initial: S, owner: OwnerScheduler) -> None:
        """Capture a deepcopy-able seed on owner, with no undo history.

        The scheduler defines the owner thread for all access. Copy failures
        propagate; construction off the owner thread raises RuntimeError.
        """
        self._owner = owner
        self._require_owner()
        self._state = copy.deepcopy(initial)
        self._previous: list[S] = []
        self._listeners: dict[int, Callable[[], None]] = {}
        self._next_listener = 0
        self._closed = False
        self._disposed = False
        self._notifying = False

    def snapshot(self) -> S:
        """Return a detached committed state; valid until the session is disposed."""
        self._require_owner()
        self._require_live()
        return copy.deepcopy(self._state)

    def commit(self, update: Callable[[S], S], *, record_undo: bool = True) -> S:
        """Calculate a complete replacement and publish it before notifying.

        ``update`` receives a detached copy of the latest committed state. If it
        raises, no state or notification changes. The result and caller's input
        never alias the stored state. record_undo=True replaces the single history
        entry; False preserves it without creating one. All gates, copying and
        notification rules apply in either mode. Subscriber failures are logged and isolated:
        a successful commit must not appear to the caller as a failed mutation.
        """
        self.ensure_input_open()
        if self._notifying:
            raise RuntimeError("interactive commit during notification")
        candidate = update(copy.deepcopy(self._state))
        self.ensure_input_open()
        replacement = copy.deepcopy(candidate)
        result = copy.deepcopy(replacement)
        if record_undo:
            self._previous = [self._state]
        self._state = replacement
        self._notify()
        return result

    def can_undo(self) -> bool:
        """Return whether one commit can be undone on the owner loop.

        Closed input returns False; a disposed session or off-owner access fails
        as for snapshot(). This query does not consume history.
        """
        self._require_owner()
        self._require_live()
        return not self._closed and bool(self._previous)

    def undo(self) -> S:
        """Restore the state before the last successful commit and notify once.

        Return a detached state and consume the single history entry, without
        creating redo. Raise FailedPreconditionError when no history exists or
        input is closed/disposed. Reject off-owner and notification-time writes.
        Subscriber failures are logged and isolated after publication.
        """
        self.ensure_input_open()
        if self._notifying:
            raise RuntimeError("interactive undo during notification")
        if not self._previous:
            raise FailedPreconditionError("interactive undo history is empty")
        replacement = self._previous[0]
        result = copy.deepcopy(replacement)
        self._state = replacement
        self._previous.clear()
        self._notify()
        return result

    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register a notification; returned cleanup handle is idempotent."""
        self.ensure_input_open()
        if not callable(callback):
            raise TypeError("interactive subscriber must be callable")
        key = self._next_listener
        self._next_listener += 1
        self._listeners[key] = callback

        def unsubscribe() -> None:
            self._require_owner()
            self._listeners.pop(key, None)

        return unsubscribe

    def ensure_input_open(self) -> None:
        """Reject all new action dispatch after terminal, including async actions."""
        self._require_owner()
        self._require_live()
        if self._closed:
            raise FailedPreconditionError("interactive input is closed")

    def close_input(self) -> None:
        """Terminal gate: keep the last committed snapshot for result construction."""
        self._require_owner()
        self._closed = True
        self._previous.clear()

    def dispose(self) -> None:
        """Drop subscriptions and reject future reads/writes after terminal cleanup."""
        self._require_owner()
        self._closed = True
        self._disposed = True
        self._listeners.clear()
        self._previous.clear()

    def _notify(self) -> None:
        self._notifying = True
        try:
            for key in tuple(self._listeners):
                if self._disposed:
                    break
                callback = self._listeners.get(key)
                if callback is not None:
                    try:
                        callback()
                    except Exception:
                        logger.exception("interactive state subscriber failed")
        finally:
            self._notifying = False

    def _require_live(self) -> None:
        if self._disposed:
            raise FailedPreconditionError("interactive session is disposed")

    def _require_owner(self) -> None:
        if not self._owner.is_owner_thread():
            raise RuntimeError("interactive session access must run on the owner loop")
