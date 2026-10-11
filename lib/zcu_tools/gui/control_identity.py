"""Application-owned write authority shared by Measure and Fluxdep.

All methods run on the constructing application owner thread. Socket adapters
supply opaque connection IDs after authentication; labels are trusted LAN
identities, not credentials. Losing authority never cancels accepted operations.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

logger = logging.getLogger(__name__)
ControlMode = Literal["owner", "readonly"]


@dataclass(frozen=True)
class ControlStatus:
    """Current authority: holder is gui or rpc:<label>; Local is always online.

    locked prohibits acquisition in Local. online means at least one connected
    socket has the remote holder's label, irrespective of that socket's mode.
    """

    holder: str
    locked: bool
    online: bool


@dataclass(frozen=True)
class ConnectionIdentity:
    """Fixed socket identity; only mode=owner may exercise label's authority."""

    label: str
    mode: ControlMode


class ControlIdentityError(FailedPreconditionError):
    """Control refusal with a stable reason_code and current status projection."""

    def __init__(self, reason: str, status: ControlStatus) -> None:
        super().__init__(reason.replace("_", " "), reason_code=reason)
        self.status = status


class ControlIdentity:
    """Single holder and fixed per-socket identities, owned by an application.

    Uninitialized sockets are observers. connect initializes a socket once;
    repeated identical initialization is inert. Explicit acquire can replace an
    offline remote holder, while connect only auto-acquires unlocked Local.
    """

    def __init__(self) -> None:
        self._owner_thread = threading.get_ident()
        self._holder = "gui"
        self._locked = False
        self._connections: dict[str, ConnectionIdentity] = {}
        self._watchers: list[Callable[[ControlStatus], None]] = []

    def status(self) -> ControlStatus:
        """Return the immutable current frontend projection on the owner thread."""
        self._assert_owner()
        online = self._holder == "gui" or any(
            self._holder == f"rpc:{identity.label}"
            for identity in self._connections.values()
        )
        return ControlStatus(self._holder, self._locked, online)

    def watch(self, callback: Callable[[ControlStatus], None]) -> Callable[[], None]:
        """Observe current status immediately and later changes; return disposer.

        Call and dispose on the owner thread. Listener failures are logged and
        isolated from committed authority transitions.
        """
        self._assert_owner()
        self._watchers.append(callback)
        self._notify(callback)

        def dispose() -> None:
            self._assert_owner()
            if callback in self._watchers:
                self._watchers.remove(callback)

        return dispose

    def connect(self, client_id: str, label: str, mode: ControlMode) -> None:
        """Initialize a socket; conflicting repeat raises InvalidInputError.

        Empty labels/unknown modes are invalid. Successful owner initialization
        auto-acquires only unlocked Local; a held/locked GUI keeps the socket
        connected without authority. Repeating connect never reacquires.
        """
        self._assert_owner()
        if not label.strip():
            raise InvalidInputError("label must be a non-empty string")
        if mode not in ("owner", "readonly"):
            raise InvalidInputError("mode must be owner or readonly")
        identity = ConnectionIdentity(label, mode)
        previous = self._connections.get(client_id)
        if previous is not None:
            if previous != identity:
                raise InvalidInputError(
                    "connection identity is fixed; reconnect to change it"
                )
            return
        before = self.status()
        self._connections[client_id] = identity
        if mode == "owner" and self._holder == "gui" and not self._locked:
            self._holder = f"rpc:{label}"
        self._publish(before)

    def disconnect(self, client_id: str) -> None:
        """Remove socket presence without releasing holder or cancelling work."""
        self._assert_owner()
        before = self.status()
        self._connections.pop(client_id, None)
        self._publish(before)

    def is_holder(self, client_id: str) -> bool:
        """True only for an initialized owner-mode socket with holder's label."""
        self._assert_owner()
        identity = self._connections.get(client_id)
        return (
            identity is not None
            and identity.mode == "owner"
            and self._holder == f"rpc:{identity.label}"
        )

    def require_holder(self, client_id: str) -> None:
        """Reject business commands with not_holder before any domain effects."""
        if not self.is_holder(client_id):
            raise ControlIdentityError("not_holder", self.status())

    def acquire(self, client_id: str) -> None:
        """Acquire Local or offline Remote; reject readonly, locked or held.

        Same-label owner sockets already hold authority and succeed unchanged.
        Refusals carry readonly_connection, control_locked or control_held.
        """
        identity = self._owner_identity(client_id)
        if self.is_holder(client_id):
            return
        before = self.status()
        if before.locked:
            raise ControlIdentityError("control_locked", before)
        if before.holder != "gui" and before.online:
            raise ControlIdentityError("control_held", before)
        self._holder = f"rpc:{identity.label}"
        self._publish(before)

    def release(self, client_id: str) -> None:
        """Return authority to unlocked Local; only owner-mode holder may release."""
        self._owner_identity(client_id)
        self.require_holder(client_id)
        before = self.status()
        self._holder = "gui"
        self._locked = False
        self._publish(before)

    def reclaim_local(self) -> None:
        """GUI intervention: reclaim to locked Local, without cancelling work."""
        before = self.status()
        self._holder = "gui"
        self._locked = True
        self._publish(before)

    def toggle_local(self) -> None:
        """Indicator click: Remote -> locked Local; Local toggles its lock."""
        before = self.status()
        self._locked = not before.locked if before.holder == "gui" else True
        self._holder = "gui"
        self._publish(before)

    def _owner_identity(self, client_id: str) -> ConnectionIdentity:
        self._assert_owner()
        identity = self._connections.get(client_id)
        if identity is None or identity.mode != "owner":
            raise ControlIdentityError("readonly_connection", self.status())
        return identity

    def _publish(self, before: ControlStatus) -> None:
        if self.status() != before:
            for callback in tuple(self._watchers):
                self._notify(callback)

    def _notify(self, callback: Callable[[ControlStatus], None]) -> None:
        try:
            callback(self.status())
        except Exception:
            logger.exception("control status listener failed")

    def _assert_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("control identity requires the application owner thread")
