"""Persistence memento types — the single on-disk snapshot of GUI app state.

These are the **memento** (Memento pattern): an immutable, serializable snapshot
the ``PersistenceCaretaker`` writes/reads as one ``gui_state_v1.json`` file. They
are pydantic v2 models (frozen) so ``model_validate`` does the load-time
validation that used to be hand-written, and ``model_dump`` produces the JSON.

They are pure data — no behaviour, no disk I/O (that is the Caretaker), no cfg
cfg conversion (that belongs to the Form resource). ``form_input`` preserves
authored JSON that the caretaker does not interpret.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue

from zcu_tools.gui.session.persistence import PersistenceError
from zcu_tools.gui.session.services.project_settings import (
    PersistedDeviceEntry as PersistedDeviceEntry,
)
from zcu_tools.gui.session.services.project_settings import (
    PersistedStartup,
)
from zcu_tools.gui.session.state import (
    DEFAULT_LEFT_PANEL_WIDTH as DEFAULT_LEFT_PANEL_WIDTH,
)

# Single top-level version for the whole app-state snapshot (Phase 126 merged the
# former startup_v2 + tab_session_v1 files into one). Bump on any incompatible
# shape change; a mismatch makes the Caretaker fall back to defaults.
APP_STATE_VERSION = 3


class PersistedTab(BaseModel):
    """One experiment name and authored Form input, without captured sources.

    adapter_name identifies the registry entry at restore time. form_input is
    detached JSON input, including expressions; compatible restore reports drops.
    """

    model_config = ConfigDict(frozen=True)

    adapter_name: str
    # Authored Form input JSON, including expression envelopes and missing fields.
    # Workspace restores compatible leaves; source snapshots are never persisted.
    form_input: dict[str, JsonValue]


class PersistedSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    tabs: tuple[PersistedTab, ...] = ()
    active_tab_index: int | None = None


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
