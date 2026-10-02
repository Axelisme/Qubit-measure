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
    from zcu_tools.gui.result_scope import ResultScope
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.services.connection import (
        ConnectRequest,
        SoCConnectionService,
    )
    from zcu_tools.gui.session.services.device import DeviceEntry
    from zcu_tools.gui.session.services.project_settings import (
        ConnectionPreferences,
        ProjectRequest,
        ProjectSettingsService,
        ResolvedProject,
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
    def list_result_scopes(
        self, *, refresh: bool = False
    ) -> tuple[ResultScope, ...]: ...
    def apply_project(self, req: ProjectRequest) -> bool: ...

    def use_context(self, label: str) -> None: ...
    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None: ...
    def get_context_labels(self) -> list[str]: ...
    def get_active_context_label(self) -> str | None: ...

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


class SetupControlFacet:
    """Composition facade over the services used by SetupDialog."""

    def __init__(  # noqa: PLR0913 — composition of the setup owners
        self,
        *,
        bus: BaseEventBus,
        settings: ProjectSettingsService,
        context: ContextControlPort,
        connection: SoCConnectionService,
        simulated_environment: SimulatedEnvironmentCoordinator,
        device: DeviceControlPort,
        on_project_applied: Callable[[ResolvedProject], None] | None = None,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._context = context
        self._connection = connection
        self._simulated_environment = simulated_environment
        self._device = device
        self._on_project_applied = on_project_applied
        self._connection_outcome_subscriptions = EventSubscriptions()

    def get_bus(self) -> BaseEventBus:
        return self._bus

    def get_setup_preferences(self) -> SetupPreferences:
        return self._settings.get_setup_preferences()

    def list_result_scopes(self, *, refresh: bool = False) -> tuple[ResultScope, ...]:
        return self._settings.list_result_scopes(refresh=refresh)

    def apply_project(self, req: ProjectRequest) -> bool:
        resolved = self._settings.apply_project(req)
        if self._on_project_applied is not None:
            self._on_project_applied(resolved)
        return True

    def use_context(self, label: str) -> None:
        self._context.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        self._context.new_context(bind_device=bind_device, clone_from=clone_from)

    def get_context_labels(self) -> list[str]:
        return self._context.get_context_labels()

    def get_active_context_label(self) -> str | None:
        return self._context.get_active_context_label()

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
