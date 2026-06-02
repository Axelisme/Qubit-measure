"""Persistence memento types — the single on-disk snapshot of GUI app state.

These are the **memento** (Memento pattern): an immutable, serializable snapshot
the ``PersistenceCaretaker`` writes/reads as one ``gui_state_v1.json`` file. They
are pydantic v2 models (frozen) so ``model_validate`` does the load-time
validation that used to be hand-written, and ``model_dump`` produces the JSON.

They are pure data — no behaviour, no disk I/O (that is the Caretaker), no cfg
codec (that is WorkspaceService's internal capture/apply). ``cfg_raw`` is an
opaque lowered-cfg dict the Caretaker never inspects.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from zcu_tools.gui.adapter import SavePaths

# Single top-level version for the whole app-state snapshot (Phase 126 merged the
# former startup_v2 + tab_session_v1 files into one). Bump on any incompatible
# shape change; a mismatch makes the Caretaker fall back to defaults.
APP_STATE_VERSION = 1

DEFAULT_LEFT_PANEL_WIDTH = 500


class PersistenceError(RuntimeError):
    """Expected failure while reading or writing the GUI state file."""


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


class PersistedTab(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter_name: str
    # Opaque lowered cfg (raw dict) — WorkspaceService owns the raw↔live codec;
    # the memento and Caretaker never look inside.
    cfg_raw: dict[str, Any]
    save_paths_override: Optional[SavePaths] = None


class PersistedSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    tabs: tuple[PersistedTab, ...] = ()
    active_tab_index: Optional[int] = None


class AppPersistedState(BaseModel):
    """The whole on-disk app-state snapshot: one file, one version."""

    model_config = ConfigDict(frozen=True)

    version: int = APP_STATE_VERSION
    startup: PersistedStartup = PersistedStartup()
    session: PersistedSession = PersistedSession()


__all__ = [
    "APP_STATE_VERSION",
    "DEFAULT_LEFT_PANEL_WIDTH",
    "PersistenceError",
    "PersistedDeviceEntry",
    "PersistedStartup",
    "PersistedTab",
    "PersistedSession",
    "AppPersistedState",
]
