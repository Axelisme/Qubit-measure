"""Committed interactive state owned by the measure app framework."""

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
        self._owner = owner
        self._require_owner()
        self._state = copy.deepcopy(initial)
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

    def commit(self, update: Callable[[S], S]) -> S:
        """Calculate a complete replacement and publish it before notifying.

        ``update`` receives a detached copy of the latest committed state. If it
        raises, no state or notification changes. The result and caller's input
        never alias the stored state. Subscriber failures are logged and isolated:
        a successful commit must not appear to the caller as a failed mutation.
        """
        self.ensure_input_open()
        if self._notifying:
            raise RuntimeError("interactive commit during notification")
        candidate = update(copy.deepcopy(self._state))
        self.ensure_input_open()
        replacement = copy.deepcopy(candidate)
        result = copy.deepcopy(replacement)
        self._state = replacement
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

    def dispose(self) -> None:
        """Drop subscriptions and reject future reads/writes after terminal cleanup."""
        self._require_owner()
        self._closed = True
        self._disposed = True
        self._listeners.clear()

    def _require_live(self) -> None:
        if self._disposed:
            raise FailedPreconditionError("interactive session is disposed")

    def _require_owner(self) -> None:
        if not self._owner.is_owner_thread():
            raise RuntimeError("interactive session access must run on the owner loop")
