"""SoC connection commands and preferences for shared session UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import ConnectionFinishedPayload

# ADR-0067: shared session ports do not depend on app or Qt owners.
if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.services.connection import (
        ConnectRequest,
        SoCConnectionService,
    )
    from zcu_tools.gui.session.services.project_settings import (
        ConnectionPreferences,
        DialogPrefill,
        ProjectSettingsService,
    )
    from zcu_tools.gui.session.types import SocCfgHandle


class SocControlPort(Protocol):
    """Owner-thread SoC commands, cached preferences and connection facts.

    This surface is independent of entry/point selection. The session owns the
    bus and hardware; callers own their bus subscriptions.
    """

    def get_bus(self) -> BaseEventBus:
        """Return the session fact bus, including SocChangedPayload events."""
        ...

    def get_dialog_prefill(self) -> DialogPrefill:
        """Read cached input preferences; this does not imply a live connection."""
        ...

    def start_connect(self, req: ConnectRequest) -> int:
        """Start an asynchronous connection and return its operation token.

        Admission conflicts raise before starting; completion is published on
        the session bus and delivered to the bound outcome callbacks.
        """
        ...

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Replace this facet's observer with owner-thread success/error callbacks.

        Success has no argument; failure supplies the human-readable reason.
        Observing an outcome does not consume it or change operation state.
        """
        ...

    def remember_connection(self, prefs: ConnectionPreferences) -> None:
        """Remember the next dialog's address inputs without connecting hardware."""
        ...

    def get_soccfg(self) -> SocCfgHandle | None:
        """Return the cached live configuration, or None when disconnected."""
        ...


class SocControlFacet(SocControlPort):
    """Compose connection commands/preferences and one replaceable outcome observer.

    Construct one facet per session on its owner thread. Methods follow the
    SocControlPort contract; entry/point and device services are not required.
    """

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        settings: ProjectSettingsService,
        connection: SoCConnectionService,
    ) -> None:
        """Bind the session fact bus, input-preference owner and connection owner."""
        self._bus = bus
        self._settings = settings
        self._connection = connection
        self._connection_outcome_subscriptions = EventSubscriptions()

    def get_bus(self) -> BaseEventBus:
        """Return the injected session fact bus."""
        return self._bus

    def get_dialog_prefill(self) -> DialogPrefill:
        """Read the settings owner's cached input preferences without hardware I/O."""
        return self._settings.get_dialog_prefill()

    def start_connect(self, req: ConnectRequest) -> int:
        """Start the requested asynchronous connection; admission conflicts raise."""
        return self._connection.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Replace the outcome observer; deliver success or an error string on owner."""
        self._connection_outcome_subscriptions.unsubscribe_all()
        self._connection_outcome_subscriptions = EventSubscriptions()

        def dispatch(
            payload: ConnectionFinishedPayload,
        ) -> None:
            if payload.success:
                on_finished()
            else:
                on_failed(payload.error_message or "SoC connection failed")

        self._connection_outcome_subscriptions.subscribe(
            self._bus, ConnectionFinishedPayload, dispatch
        )

    def remember_connection(self, prefs: ConnectionPreferences) -> None:
        """Cache address inputs for the next show without starting a connection."""
        self._settings.remember_connection(prefs)

    def get_soccfg(self) -> SocCfgHandle | None:
        """Read the live cached configuration, or None when disconnected."""
        return self._connection.get_soccfg()
