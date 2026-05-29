from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from zcu_tools.gui.event_bus import DeviceChangedPayload, EventBus, GuiEvent

from .context import ContextService
from .device import DeviceMemoryInfo, DeviceService
from .startup_persistence import (
    DEFAULT_LEFT_PANEL_WIDTH,
    PersistedDeviceEntry,
    PersistedStartup,
    StartupPersistenceError,
)

if TYPE_CHECKING:
    from zcu_tools.gui.state import State

    from .ports import StartupStorePort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartupProjectRequest:
    chip_name: str
    qub_name: str
    res_name: str
    result_dir: str
    database_path: str

    def __post_init__(self) -> None:
        if not self.chip_name or not self.qub_name or not self.res_name:
            raise ValueError("Startup project names must be non-empty")


@dataclass(frozen=True)
class StartupConnectionRequest:
    ip: str
    port: int

    def __post_init__(self) -> None:
        if not self.ip:
            raise ValueError("Startup connection IP must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Startup connection port must be in range 1..65535")


class StartupService:
    """Own startup context construction and preference transactions."""

    def __init__(
        self,
        context: ContextService,
        devices: DeviceService,
        persistence: "StartupStorePort",
        state: "State",
        bus: EventBus,
    ) -> None:
        self._context = context
        self._devices = devices
        self._persistence = persistence
        self._state = state
        # Persistence is a projection of device State: whenever device state
        # changes, re-derive and overwrite the remembered-device set on disk.
        bus.subscribe(GuiEvent.DEVICE_CHANGED, self._on_device_changed)

    def restore_devices(self) -> None:
        data = self._persistence.load()
        if data is None:
            return
        self._devices.register_remembered_devices(
            [
                DeviceMemoryInfo(
                    type_name=entry.type_name,
                    name=entry.name,
                    address=entry.address,
                )
                for entry in data.devices
            ]
        )

    def get_persisted(self) -> PersistedStartup | None:
        return self._persistence.load()

    def apply_project(self, req: StartupProjectRequest) -> None:
        self._context.set_startup_context(
            MetaDict(),
            ModuleLibrary(),
            req.chip_name,
            req.qub_name,
            req.res_name,
            req.result_dir,
            req.database_path,
        )
        if req.result_dir:
            self._context.setup_project(req.result_dir)
        self._persistence.update_project(
            chip_name=req.chip_name,
            qub_name=req.qub_name,
            res_name=req.res_name,
            result_dir=req.result_dir,
            database_path=req.database_path,
        )

    def remember_connection(self, req: StartupConnectionRequest) -> None:
        self._persistence.update_connection(ip=req.ip, port=req.port)

    def _on_device_changed(self, _payload: DeviceChangedPayload) -> None:
        """Project the current remembered-device set from State onto disk.

        Declarative: the remembered set is whatever State currently holds with
        ``remember=True``; this overwrites the persisted list wholesale. A
        diff-guard skips the (frequent, transient) DEVICE_CHANGED emissions that
        do not alter the remembered set. Disk-write failures are logged and
        swallowed — this runs off the original caller's path and cannot surface
        an error to it.
        """
        entries = [
            PersistedDeviceEntry(
                type_name=dev.type_name, name=dev.name, address=dev.address
            )
            for dev in self._state.list_devices()
            if dev.remember
        ]
        if entries == self._persistence.get_current().devices:
            return
        try:
            self._persistence.replace_devices(entries)
        except StartupPersistenceError:
            logger.warning("Failed to persist remembered devices", exc_info=True)

    def get_left_panel_width(self) -> int:
        data = self._persistence.load()
        return DEFAULT_LEFT_PANEL_WIDTH if data is None else data.left_panel_width

    def save_left_panel_width(self, width: int) -> None:
        self._persistence.update_left_panel_width(width)
