"""StartupService — session startup context + remembered-prefs capture/restore.

Owns the session-core startup concern: bootstrap the active context from a
project (chip/qub/res + paths), remember the prefill prefs + the remembered-device
set, and project them to / from the persistence memento (``PersistedStartup``).
It is the session half of the persistence split (P-c): the *session* memento
slice lives here; an app combines it with its own experiment slice (measure:
``AppPersistedState`` wraps this + ``PersistedSession``).

App-agnostic: depends on the session ports (``StartupContextPort`` /
``RememberedDevicePort``) + ``SessionState``; no disk I/O (the app's Caretaker
owns that), no event bus.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from zcu_tools.gui.session.ports import DeviceMemoryInfo
from zcu_tools.gui.session.state import DEFAULT_LEFT_PANEL_WIDTH, StartupPrefs
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

if TYPE_CHECKING:
    from zcu_tools.gui.session.ports import RememberedDevicePort, StartupContextPort
    from zcu_tools.gui.session.state import SessionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session persistence memento slice (P-c): the startup prefs + remembered devices.
# Pydantic v2 frozen models — the app's Caretaker writes/reads them as part of its
# combined on-disk snapshot; pure data, no I/O.
# ---------------------------------------------------------------------------


class PersistedDeviceEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    type_name: str
    name: str
    address: str


class PersistedStartup(BaseModel):
    model_config = ConfigDict(frozen=True)

    chip_name: str = ""
    qub_name: str = ""
    res_name: str = ""
    result_dir: str = ""
    database_path: str = ""
    ip: str = "192.168.10.1"
    port: int = 8887
    devices: tuple[PersistedDeviceEntry, ...] = ()
    left_panel_width: int = DEFAULT_LEFT_PANEL_WIDTH


def derive_project_paths(chip_name: str, qub_name: str, root: str) -> tuple[str, str]:
    """Single source of truth for the default per-qubit result / database paths.

    Returns ``(result_dir, database_path)`` mirroring single_qubit.md, where
    ``database_path`` is the *dated* data folder
    ``Database/chip/qub/YYYY/MM/Data_MMDD`` — the notebook's ``create_datafolder``
    return value, with the date in the path itself. Save-path builders therefore
    join filenames directly under ``ctx.database_path`` and must NOT re-append the
    date (see BaseAdapter.make_default_save_paths). The date is *today's* at the
    moment this is called, so restore paths (which re-derive) always land in the
    current day's folder rather than a stale persisted one.

    Both the setup dialog and the mock/RPC startup helpers derive through here so
    the chip/qub (and date) segments are joined in exactly ONE place —
    ``apply_project`` must not re-scope. ``root`` is the base dir (e.g. cwd)."""
    from zcu_tools.utils.datasaver import get_datafolder_path

    result_dir = os.path.join(root, "result", chip_name, qub_name)
    database_path = get_datafolder_path(
        os.path.join(root, "Database"), os.path.join(chip_name, qub_name)
    )
    return result_dir, database_path


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
    """Own startup context construction + startup-prefs capture/restore.

    Stateless app service: the remembered prefs live in ``State.startup_prefs``,
    not here. Apply/connect update those prefs at write-time; capture projects
    them (+ device set from State) into a memento; restore writes them back and
    registers remembered devices. No disk I/O (the PersistenceCaretaker owns it),
    no DEVICE_CHANGED subscription (capture re-projects devices at flush time).
    """

    def __init__(
        self,
        context: "StartupContextPort",
        devices: "RememberedDevicePort",
        state: "SessionState",
    ) -> None:
        self._context = context
        self._devices = devices
        self._state = state

    def get_persisted(self) -> PersistedStartup:
        """The current remembered prefs (for the setup dialog's prefill)."""
        return self._project_prefs_to_startup()

    def apply_project(self, req: StartupProjectRequest) -> None:
        # result_dir / database_path arrive already scoped under chip/qub by the
        # caller (UI derives them via derive_project_paths; mock uses the same
        # helper). apply_project does NOT re-scope — that would double the
        # chip/qub segment.
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
        # Remember the just-applied project as the prefill values (write-time).
        prefs = self._state.startup_prefs
        prefs.chip_name = req.chip_name
        prefs.qub_name = req.qub_name
        prefs.res_name = req.res_name
        prefs.result_dir = req.result_dir
        prefs.database_path = req.database_path

    def remember_connection(self, req: StartupConnectionRequest) -> None:
        prefs = self._state.startup_prefs
        prefs.ip = req.ip
        prefs.port = req.port

    def capture_startup(self, *, left_panel_width: int) -> PersistedStartup:
        """Project the remembered prefs (+ current remember-device set from
        State) into a memento. The device set is re-derived here (deferred to
        flush time — replaces the old DEVICE_CHANGED eager projection)."""
        devices = tuple(
            PersistedDeviceEntry(
                type_name=dev.type_name, name=dev.name, address=dev.address
            )
            for dev in self._state.list_devices()
            if dev.remember
        )
        prefs = self._state.startup_prefs
        return PersistedStartup(
            chip_name=prefs.chip_name,
            qub_name=prefs.qub_name,
            res_name=prefs.res_name,
            result_dir=prefs.result_dir,
            database_path=prefs.database_path,
            ip=prefs.ip,
            port=prefs.port,
            devices=devices,
            left_panel_width=left_panel_width,
        )

    def restore_startup(self, data: PersistedStartup) -> None:
        """Seed the remembered prefs from the memento + register remembered
        devices. Project is NOT auto-applied to the active context (the user
        applies it via the setup dialog) — the instrument never auto-connects."""
        self._state.startup_prefs = StartupPrefs(
            chip_name=data.chip_name,
            qub_name=data.qub_name,
            res_name=data.res_name,
            result_dir=data.result_dir,
            database_path=data.database_path,
            ip=data.ip,
            port=data.port,
            left_panel_width=data.left_panel_width,
        )
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

    def _project_prefs_to_startup(self) -> PersistedStartup:
        prefs = self._state.startup_prefs
        return PersistedStartup(
            chip_name=prefs.chip_name,
            qub_name=prefs.qub_name,
            res_name=prefs.res_name,
            result_dir=prefs.result_dir,
            database_path=prefs.database_path,
            ip=prefs.ip,
            port=prefs.port,
            left_panel_width=prefs.left_panel_width,
        )
