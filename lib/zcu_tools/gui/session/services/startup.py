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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from zcu_tools.gui.result_scope import ProjectPaths, ResultScope, ResultScopeManager
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
    scope_id: str = ""
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
    paths = ResultScopeManager(root).derive_paths(chip_name, qub_name)
    return paths.result_dir, paths.database_path


@dataclass(frozen=True)
class StartupProjectRequest:
    chip_name: str
    qub_name: str
    res_name: str
    scope_id: str | None = None

    def __post_init__(self) -> None:
        if not self.chip_name or not self.qub_name or not self.res_name:
            raise ValueError("Startup project names must be non-empty")


@dataclass(frozen=True)
class ResolvedStartupProject:
    chip_name: str
    qub_name: str
    res_name: str
    result_dir: str
    database_path: str
    params_path: str
    scope_id: str

    def as_wire_dict(self) -> dict[str, str]:
        return {
            "chip_name": self.chip_name,
            "qub_name": self.qub_name,
            "res_name": self.res_name,
            "result_dir": self.result_dir,
            "database_path": self.database_path,
            "params_path": self.params_path,
            "scope_id": self.scope_id,
        }


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
        context: StartupContextPort,
        devices: RememberedDevicePort,
        state: SessionState,
        result_scopes: ResultScopeManager,
    ) -> None:
        self._context = context
        self._devices = devices
        self._state = state
        self._result_scopes = result_scopes

    def get_persisted(self) -> PersistedStartup:
        """The current remembered prefs (for the setup dialog's prefill)."""
        return self._project_prefs_to_startup()

    def list_result_scopes(self, *, refresh: bool = False) -> tuple[ResultScope, ...]:
        return self._result_scopes.list_scopes(refresh=refresh)

    def list_result_chip_names(self) -> tuple[str, ...]:
        return self._result_scopes.list_chip_names()

    def list_result_qub_names(self, chip_name: str) -> tuple[str, ...]:
        return self._result_scopes.list_qub_names(chip_name)

    def derive_project_paths(self, chip_name: str, qub_name: str) -> ProjectPaths:
        return self._result_scopes.derive_paths(chip_name, qub_name)

    def apply_project(self, req: StartupProjectRequest) -> ResolvedStartupProject:
        scope = self._result_scopes.ensure_scope(
            chip_name=req.chip_name,
            qub_name=req.qub_name,
            scope_id=req.scope_id,
        )
        paths = self._result_scopes.derive_paths(req.chip_name, req.qub_name)
        resolved = ResolvedStartupProject(
            chip_name=req.chip_name,
            qub_name=req.qub_name,
            res_name=req.res_name,
            result_dir=scope.result_dir,
            database_path=paths.database_path,
            params_path=scope.params_path,
            scope_id=scope.scope_id,
        )
        self._context.set_startup_context(
            MetaDict(),
            ModuleLibrary(),
            resolved.chip_name,
            resolved.qub_name,
            resolved.res_name,
            resolved.result_dir,
            resolved.database_path,
        )
        self._context.setup_project(resolved.result_dir)
        # Remember the just-applied project as the prefill values (write-time).
        prefs = self._state.startup_prefs
        prefs.chip_name = resolved.chip_name
        prefs.qub_name = resolved.qub_name
        prefs.res_name = resolved.res_name
        prefs.scope_id = resolved.scope_id
        prefs.result_dir = resolved.result_dir
        prefs.database_path = resolved.database_path
        return resolved

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
            scope_id=prefs.scope_id,
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
        self._state.set_startup_prefs(
            StartupPrefs(
                chip_name=data.chip_name,
                qub_name=data.qub_name,
                res_name=data.res_name,
                scope_id=data.scope_id,
                result_dir=data.result_dir,
                database_path=data.database_path,
                ip=data.ip,
                port=data.port,
                left_panel_width=data.left_panel_width,
            )
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
            scope_id=prefs.scope_id,
            result_dir=prefs.result_dir,
            database_path=prefs.database_path,
            ip=prefs.ip,
            port=prefs.port,
            left_panel_width=prefs.left_panel_width,
        )
