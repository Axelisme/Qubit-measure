"""Setup-dialog control facet for shared session UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

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
    from zcu_tools.gui.session.services.startup import (
        PersistedStartup,
        ResolvedStartupProject,
        StartupConnectionRequest,
        StartupProjectRequest,
        StartupService,
    )
    from zcu_tools.gui.session.types import SocCfgHandle


class SetupControlPort(Protocol):
    """Project/context/connection surface for the shared setup dialog."""

    def get_bus(self) -> BaseEventBus: ...
    def get_persisted_startup(self) -> PersistedStartup: ...
    def list_result_scopes(self) -> tuple[ResultScope, ...]: ...
    def apply_startup_project(self, req: StartupProjectRequest) -> bool: ...

    def use_context(self, label: str) -> None: ...
    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None: ...
    def get_context_labels(self) -> list[str]: ...
    def get_active_context_label(self) -> str | None: ...

    def start_connect(self, req: ConnectRequest) -> int: ...
    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None: ...
    def remember_startup_connection(self, req: StartupConnectionRequest) -> None: ...
    def get_soccfg(self) -> SocCfgHandle | None: ...

    def list_devices(self) -> list[DeviceEntry]: ...
    def get_device_unit(self, name: str) -> str: ...


class SetupControlFacet:
    """Composition facade over the services used by SetupDialog."""

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        startup: StartupService,
        context: ContextControlPort,
        connection: SoCConnectionService,
        device: DeviceControlPort,
        on_project_applied: Callable[[ResolvedStartupProject], None] | None = None,
    ) -> None:
        self._bus = bus
        self._startup = startup
        self._context = context
        self._connection = connection
        self._device = device
        self._on_project_applied = on_project_applied

    def get_bus(self) -> BaseEventBus:
        return self._bus

    def get_persisted_startup(self) -> PersistedStartup:
        return self._startup.get_persisted()

    def list_result_scopes(self) -> tuple[ResultScope, ...]:
        return self._startup.list_result_scopes()

    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        resolved = self._startup.apply_project(req)
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

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        for signal in (
            self._connection.connection_finished,
            self._connection.connection_failed,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass
        self._connection.connection_finished.connect(on_finished)
        self._connection.connection_failed.connect(on_failed)

    def remember_startup_connection(self, req: StartupConnectionRequest) -> None:
        self._startup.remember_connection(req)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._connection.get_soccfg()

    def list_devices(self) -> list[DeviceEntry]:
        return self._device.list_devices()

    def get_device_unit(self, name: str) -> str:
        return self._device.get_device_unit(name)
