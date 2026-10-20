"""Independent entry/point and SoC control facets for shared session UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import (
    ConnectionFinishedPayload,
)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.context_types import (
        EntryInfo,
        EntryRequest,
        PointCreateRequest,
        PointUseRequest,
        SavedPoint,
        WorkingPointInfo,
    )
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.services.connection import (
        ConnectRequest,
        SoCConnectionService,
    )
    from zcu_tools.gui.session.services.project_settings import (
        ConnectionPreferences,
        ProjectSettingsService,
        SetupPreferences,
    )
    from zcu_tools.gui.session.types import SocCfgHandle


class SetupControlPort(Protocol):
    """Entry/point surface for the shared setup dialog.

    devices exposes the already-injected DeviceControlPort. Setup uses its
    State-only list/value/unit queries to suggest an editable working-point
    label; it does not poll hardware or bind a point to a device.
    """

    devices: DeviceControlPort

    def get_bus(self) -> BaseEventBus: ...
    def get_setup_preferences(self) -> SetupPreferences: ...
    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        """List cached entry names; True explicitly refreshes via resources."""
        ...

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        """Open through resources and leave point unselected; errors preserve the binding."""
        ...

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        """Create an entry without selecting a point; installation failure retains it for Open."""
        ...

    def get_entry_info(self) -> EntryInfo | None:
        """Read cached opened entry identity/paths, or None."""
        ...

    def current_point_label(self) -> str | None:
        """Read the selected saved point label, or None when none is selected; no I/O."""
        ...

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        """Read cached point descriptions; True explicitly refreshes through owner."""
        ...

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        """Prepare/select a point in the opened entry; failure preserves binding."""
        ...

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        """Create from setup/clone then select; PointBindingError retains destination."""
        ...


class SetupControlFacet(SetupControlPort):
    """Composition facade over the services used by SetupDialog."""

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        settings: ProjectSettingsService,
        context: ContextControlPort,
        device: DeviceControlPort,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._context = context
        self.devices = device

    def get_bus(self) -> BaseEventBus:
        return self._bus

    def get_setup_preferences(self) -> SetupPreferences:
        return self._settings.get_setup_preferences()

    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        return self._settings.list_entries(refresh=refresh)

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        return self._settings.open_entry(request)

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        return self._settings.new_entry(request)

    def get_entry_info(self) -> EntryInfo | None:
        return self._context.get_entry_info()

    def current_point_label(self) -> str | None:
        return self._context.current_point_label()

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        return self._context.list_points(refresh=refresh)

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        return self._context.use_point(request)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        return self._context.new_point(request)


class SocControlPort(Protocol):
    """Owner-thread SoC commands, cached preferences and connection facts.

    This surface is independent of entry/point selection. The session owns the
    bus and hardware; callers own their bus subscriptions.
    """

    def get_bus(self) -> BaseEventBus:
        """Return the session fact bus, including SocChangedPayload events."""
        ...

    def get_setup_preferences(self) -> SetupPreferences:
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

    def get_setup_preferences(self) -> SetupPreferences:
        """Read the settings owner's cached input preferences without hardware I/O."""
        return self._settings.get_setup_preferences()

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
