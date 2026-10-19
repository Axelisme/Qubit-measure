"""Explicit entry composition and remembered session settings.

Entry commands use ResultEntry with injected roots before asking the context
owner to leave point unselected. Preferences are live dialog prefill, not
persistent session state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.result_scope import ResultScopeManager
from zcu_tools.gui.session.context_types import EntryInfo, EntryRequest
from zcu_tools.resources.entry import ResultEntry

if TYPE_CHECKING:
    from zcu_tools.gui.session.ports import ProjectContextPort
    from zcu_tools.gui.session.state import SessionState

logger = logging.getLogger(__name__)


def derive_project_paths(chip_name: str, qub_name: str, root: str) -> tuple[str, str]:
    """Single source of truth for the default per-qubit result / database paths.

    Returns ``(result_dir, database_path)`` mirroring single_qubit.md, where
    ``database_path`` is the *dated* data folder
    ``Database/chip/qub/YYYY/MM/Data_MMDD`` — the notebook's ``create_datafolder``
    return value, with the date in the path itself. Save-path builders therefore
    join filenames directly under ``ctx.database_path`` and must NOT re-append the
    date. The date is *today's* at the moment this is called, so restore paths
    (which re-derive) always land in the
    current day's folder rather than a stale persisted one.

    Both the setup dialog and the mock/RPC project helpers derive through here so
    the chip/qub (and date) segments are joined in exactly ONE place —
    ``apply_project`` must not re-scope. ``root`` is the base dir (e.g. cwd)."""
    paths = ResultScopeManager(root).derive_paths(chip_name, qub_name)
    return paths.result_dir, paths.database_path


@dataclass(frozen=True)
class ProjectRequest:
    chip_name: str
    qub_name: str
    res_name: str
    scope_id: str | None = None

    def __post_init__(self) -> None:
        if not self.chip_name or not self.qub_name or not self.res_name:
            raise ValueError("Project names must be non-empty")


@dataclass(frozen=True)
class ResolvedProject:
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
class SetupPreferences:
    """What the setup dialog prefills — not the active project or connection.

    A narrow live read model; these values are not a disk session memento.
    """

    entry_name: str
    ip: str
    port: int


@dataclass(frozen=True)
class ConnectionPreferences:
    ip: str
    port: int

    def __post_init__(self) -> None:
        if not self.ip:
            raise ValueError("Connection IP must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Connection port must be in range 1..65535")


class ProjectSettingsService:
    """Own entry discovery/open/create and remembered settings on the owner thread.

    ResultEntry owns filesystem/name validation. ContextService owns binding
    publication. Entry preferences change only after installation succeeds.
    This service does not capture or restore domain state from disk.
    """

    def __init__(
        self,
        context: ProjectContextPort,
        state: SessionState,
        *,
        result_root: str | Path,
        database_root: str | Path,
    ) -> None:
        """Use explicit resource roots for owner-thread entry commands.

        context installs prepared entry handles; state owns live dialog
        preferences. ResultEntry owns name/path/file validation. roots
        never default to a singleton or depend on GUI selection.
        """
        self._owner = OwnerThreadGuard()
        self._result_root = Path(result_root)
        self._database_root = Path(database_root)
        self._entry_names: tuple[str, ...] | None = None
        self._context = context
        self._state = state

    def list_entries(self, *, refresh: bool = False) -> tuple[str, ...]:
        """List entry names through resources, retaining the last successful cache.

        First use or refresh=True reads the explicit result_root; False reuses
        the published cache. Filesystem errors propagate, not as an empty list.
        """
        self._owner.assert_owner()
        if self._entry_names is None or refresh:
            names = ResultEntry.list_names(result_root=self._result_root)
            self._entry_names = names
        return self._entry_names

    def open_entry(self, request: EntryRequest) -> EntryInfo:
        """Open and install a resource-validated entry without selecting a point or touching hardware.

        Only successful installation remembers the entry name and updates cache.
        Resource/installation errors propagate, preserving previous preferences.
        """
        self._owner.assert_owner()
        entry = ResultEntry.open(
            request.name,
            result_root=self._result_root,
            database_root=self._database_root,
        )
        return self._install_entry(entry)

    def new_entry(self, request: EntryRequest) -> EntryInfo:
        """Create and install an entry without selecting a point.

        Resource errors propagate. Installation failure retains the created
        entry and raises RuntimeError from its cause, naming Open as recovery.
        Never retry New or delete resources to imitate rollback.
        """
        self._owner.assert_owner()
        entry = ResultEntry.create(
            request.name,
            result_root=self._result_root,
            database_root=self._database_root,
        )
        try:
            return self._install_entry(entry)
        except Exception as exc:
            raise RuntimeError(
                f"Entry {request.name!r} was created but could not be installed. "
                f"Open the same entry {request.name!r} to recover; do not repeat New."
            ) from exc

    def _install_entry(self, entry: ResultEntry) -> EntryInfo:
        info = self._context.install_entry(entry)
        self._state.preferences.entry_name = info.name
        if self._entry_names is not None:
            self._entry_names = tuple(sorted(set((*self._entry_names, info.name))))
        return info

    def get_setup_preferences(self) -> SetupPreferences:
        """The remembered values the setup dialog prefills (not the active project)."""
        prefs = self._state.preferences
        return SetupPreferences(
            entry_name=prefs.entry_name,
            ip=prefs.ip,
            port=prefs.port,
        )

    def get_left_panel_width(self) -> int:
        """The remembered left-panel width (seeded by restore, read by new tabs)."""
        return self._state.preferences.left_panel_width

    def remember_connection(self, req: ConnectionPreferences) -> None:
        prefs = self._state.preferences
        prefs.ip = req.ip
        prefs.port = req.port

    def restore_left_panel_width(self, width: int) -> None:
        """Seed pure Qt panel width in logical pixels on the owner thread.

        New tabs use this width. Reject nonpositive values with ValueError.
        This command changes only in-memory presentation preferences, without I/O.
        """
        self._owner.assert_owner()
        if width <= 0:
            raise ValueError("Panel width must be positive")
        self._state.preferences.left_panel_width = width
