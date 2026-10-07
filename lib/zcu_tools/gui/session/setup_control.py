"""Setup-dialog control facet for shared session UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.event_bus import EventSubscriptions
from zcu_tools.gui.session.events import (
    ConnectionFinishedPayload,
    SimulatedEnvironmentFinishedPayload,
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
    from zcu_tools.gui.session.services.device import DeviceEntry
    from zcu_tools.gui.session.services.project_settings import (
        ConnectionPreferences,
        ProjectSettingsService,
        SetupPreferences,
    )
    from zcu_tools.gui.session.services.simulated_environment import (
        SimulatedEnvironmentCoordinator,
    )
    from zcu_tools.gui.session.types import SocCfgHandle
    from zcu_tools.program.v2.sim import SimParams


class SetupControlPort(Protocol):
    """Project/context/connection surface for the shared setup dialog."""

    def get_bus(self) -> BaseEventBus: ...
    def get_setup_preferences(self) -> SetupPreferences: ...
    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        """List cached entry names; True explicitly refreshes via resources."""
        ...

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        """Open through resources and publish DRAFT; errors preserve the binding."""
        ...

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        """Create/publish DRAFT; post-creation failure retains the entry for Open."""
        ...

    def get_entry_info(self) -> EntryInfo | None:
        """Read cached opened entry identity/paths, or None."""
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

    def start_connect(self, req: ConnectRequest) -> int: ...
    def start_simulated_environment(
        self, *, sim_params: SimParams | None = None
    ) -> int: ...
    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None: ...
    def remember_connection(self, prefs: ConnectionPreferences) -> None: ...
    def get_soccfg(self) -> SocCfgHandle | None: ...

    def list_devices(self) -> list[DeviceEntry]: ...
    def get_device_unit(self, name: str) -> str: ...


class SetupControlFacet(SetupControlPort):
    """Composition facade over the services used by SetupDialog."""

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        settings: ProjectSettingsService,
        context: ContextControlPort,
        connection: SoCConnectionService,
        simulated_environment: SimulatedEnvironmentCoordinator,
        device: DeviceControlPort,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._context = context
        self._connection = connection
        self._simulated_environment = simulated_environment
        self._device = device
        self._connection_outcome_subscriptions = EventSubscriptions()

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

    def list_points(self, *, refresh: bool = False) -> tuple[WorkingPointInfo, ...]:
        return self._context.list_points(refresh=refresh)

    def use_point(self, request: PointUseRequest) -> SavedPoint:
        return self._context.use_point(request)

    def new_point(self, request: PointCreateRequest) -> SavedPoint:
        return self._context.new_point(request)

    def start_connect(self, req: ConnectRequest) -> int:
        return self._connection.start_connect(req)

    def start_simulated_environment(
        self, *, sim_params: SimParams | None = None
    ) -> int:
        return self._simulated_environment.start(sim_params=sim_params)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        self._connection_outcome_subscriptions.unsubscribe_all()
        self._connection_outcome_subscriptions = EventSubscriptions()

        def dispatch(
            payload: ConnectionFinishedPayload | SimulatedEnvironmentFinishedPayload,
        ) -> None:
            if payload.success:
                on_finished()
            else:
                on_failed(payload.error_message or "SoC connection failed")

        self._connection_outcome_subscriptions.subscribe(
            self._bus, ConnectionFinishedPayload, dispatch
        )
        self._connection_outcome_subscriptions.subscribe(
            self._bus, SimulatedEnvironmentFinishedPayload, dispatch
        )

    def remember_connection(self, prefs: ConnectionPreferences) -> None:
        self._settings.remember_connection(prefs)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._connection.get_soccfg()

    def list_devices(self) -> list[DeviceEntry]:
        return self._device.list_devices()

    def get_device_unit(self, name: str) -> str:
        return self._device.get_device_unit(name)
