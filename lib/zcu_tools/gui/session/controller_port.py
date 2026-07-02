"""SessionControllerPort — the Controller surface the setup dialog needs.

The shared setup dialog in ``gui/session/ui`` never reaches into a concrete
``gui.app.*`` Controller; it depends on this narrow Protocol and each app's
Controller implements it structurally (measure: ``Controller``; autofluxdep: its
own controller). This keeps the dialog app-agnostic and prevents a back-edge
from the session core to any app package.

The port is the union of exactly the setup methods the shared dialog calls:

- **setup dialog**: project/startup bootstrap (``apply_startup_project`` /
  ``get_persisted_startup`` / ``get_project_root``), context switching
  (``use_context`` / ``new_context`` / ``get_context_labels`` /
  ``get_active_context_label``), connection (``start_connect`` /
  ``bind_connection_outcome`` / ``remember_startup_connection`` /
  ``get_soccfg``), device list/unit lookup for flux binding.

The setup dialog uses ``get_bus`` for live updates. The device, predictor,
progress, and context/inspect surfaces use their own narrow facets.

Return types are declared against the **shared base** (``BaseEventBus``) so an
app's richer concrete type (measure's ``EventBus``) satisfies the port
covariantly. No isinstance checks run against this Protocol — it exists purely
to type the dialogs; pyright verifies conformance at each app's dialog call
site.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.result_scope import ProjectPaths, ResultScope
    from zcu_tools.gui.session.services.connection import ConnectRequest
    from zcu_tools.gui.session.services.device import (
        DeviceEntry,
    )
    from zcu_tools.gui.session.services.startup import (
        PersistedStartup,
        StartupConnectionRequest,
        StartupProjectRequest,
    )
    from zcu_tools.gui.session.types import SocCfgHandle


class SessionControllerPort(Protocol):
    """Narrow Controller surface the shared setup dialog depends on."""

    # --- shared ------------------------------------------------------------
    def get_bus(self) -> BaseEventBus: ...

    # --- setup dialog: project / startup -----------------------------------
    def apply_startup_project(
        self, req: StartupProjectRequest
    ) -> bool | dict[str, str]: ...
    def get_persisted_startup(self) -> PersistedStartup: ...
    def get_project_root(self) -> str: ...
    def list_result_scopes(self) -> tuple[ResultScope, ...]: ...
    def derive_project_paths(self, chip_name: str, qub_name: str) -> ProjectPaths: ...

    # --- setup dialog: context switching -----------------------------------
    def use_context(self, label: str) -> None: ...
    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None: ...
    def get_context_labels(self) -> list[str]: ...
    def get_active_context_label(self) -> str | None: ...
    # --- setup dialog: connection ------------------------------------------
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
