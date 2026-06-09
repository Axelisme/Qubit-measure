"""SessionControllerPort — the Controller surface the session-core dialogs need.

The shared dialogs in ``gui/session/ui`` (setup / device) never reach into a
concrete ``gui.app.*`` Controller; they depend on this narrow Protocol and each
app's Controller implements it structurally (measure: ``Controller``;
autofluxdep: its own controller). This keeps the dialogs app-agnostic and
prevents a back-edge from the session core to any app package.

The port is the union of exactly the methods the two dialogs call:

- **setup dialog**: project/startup bootstrap (``apply_startup_project`` /
  ``get_persisted_startup`` / ``get_project_root``), context switching
  (``use_context`` / ``new_context`` / ``get_context_labels`` /
  ``get_active_context_label``), connection (``start_connect`` /
  ``bind_connection_outcome`` / ``remember_startup_connection`` /
  ``get_soccfg``), device unit lookup.
- **device dialog**: device lifecycle (``start_connect_device`` /
  ``start_disconnect_device`` / ``start_reconnect_device`` /
  ``start_setup_device`` / ``forget_device`` / ``cancel_device_operation``),
  device queries (``list_devices`` / ``get_device_snapshot`` /
  ``get_device_info`` / ``is_memory_device`` / ``get_active_device_setup``),
  and progress attach/read (``attach_progress`` / ``progress_bars``).

Both share ``get_bus`` (the event bus they subscribe to for live updates).

Return types are declared against the **shared base** (``BaseEventBus``) so an
app's richer concrete type (measure's ``EventBus``) satisfies the port
covariantly. No isinstance checks run against this Protocol — it exists purely
to type the dialogs; pyright verifies conformance at each app's dialog call
site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Protocol

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.services.connection import ConnectRequest
    from zcu_tools.gui.session.services.device import (
        ConnectDeviceRequest,
        DeviceEntry,
        DeviceSetupSnapshot,
        DeviceSnapshot,
        DisconnectDeviceRequest,
        SetupDeviceRequest,
    )
    from zcu_tools.gui.session.services.startup import (
        PersistedStartup,
        StartupConnectionRequest,
        StartupProjectRequest,
    )
    from zcu_tools.gui.session.types import SocCfgHandle


class SessionControllerPort(Protocol):
    """Narrow Controller surface the shared setup / device dialogs depend on."""

    # --- shared ------------------------------------------------------------
    def get_bus(self) -> "BaseEventBus": ...

    # --- setup dialog: project / startup -----------------------------------
    def apply_startup_project(self, req: "StartupProjectRequest") -> bool: ...
    def get_persisted_startup(self) -> "PersistedStartup": ...
    def get_project_root(self) -> str: ...

    # --- setup dialog: context switching -----------------------------------
    def use_context(self, label: str) -> None: ...
    def new_context(
        self,
        bind_device: Optional[str] = None,
        clone_from: Optional[str] = None,
    ) -> None: ...
    def get_context_labels(self) -> "list[str]": ...
    def get_active_context_label(self) -> Optional[str]: ...

    # --- setup dialog: connection ------------------------------------------
    def start_connect(self, req: "ConnectRequest") -> int: ...
    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None: ...
    def remember_startup_connection(self, req: "StartupConnectionRequest") -> None: ...
    def get_soccfg(self) -> "Optional[SocCfgHandle]": ...
    def get_device_unit(self, name: str) -> str: ...

    # --- device dialog: lifecycle ------------------------------------------
    def start_connect_device(self, req: "ConnectDeviceRequest") -> int: ...
    def start_disconnect_device(self, req: "DisconnectDeviceRequest") -> int: ...
    def start_reconnect_device(self, name: str) -> None: ...
    def start_setup_device(self, req: "SetupDeviceRequest") -> int: ...
    def forget_device(self, name: str) -> None: ...
    def cancel_device_operation(self, name: str) -> None: ...

    # --- device dialog: queries --------------------------------------------
    def list_devices(self) -> "list[DeviceEntry]": ...
    def get_device_snapshot(self, name: str) -> "Optional[DeviceSnapshot]": ...
    def get_device_info(self, name: str) -> "Optional[BaseDeviceInfo]": ...
    def is_memory_device(self, name: str) -> bool: ...
    def get_active_device_setup(self) -> "Optional[DeviceSetupSnapshot]": ...

    # --- device dialog: progress -------------------------------------------
    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]: ...
    def progress_bars(
        self, owner_id: str
    ) -> "tuple[tuple[int, ProgressBarModel], ...]": ...
