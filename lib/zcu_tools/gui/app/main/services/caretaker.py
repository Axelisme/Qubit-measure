"""PersistenceCaretaker — the Memento Caretaker for GUI app state.

A **Driven Adapter** (ADR-0005): it does only disk I/O on a single
``gui_state_v1.json`` file and owns the load/flush *timing*. It depends solely
on a narrow ``PersistOriginatorPort`` (the Controller) — never on State, the
services, the EventBus, or cfg. It asks the originator for one immutable
``AppPersistedState`` to write, and hands one back to restore.

``flush()`` is trigger-agnostic: it re-captures and writes every time it is
called. Today the only triggers are lifecycle (close → flush, startup →
restore_all), but a timer / button / event / RPC could call it later without
any change here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Optional

from platformdirs import user_cache_dir

from .persistence_types import APP_STATE_VERSION, AppPersistedState, PersistenceError

if TYPE_CHECKING:
    from .ports import PersistOriginatorPort, RestoreReport

logger = logging.getLogger(__name__)

_STATE_FILENAME = "gui_state_v1.json"


def _safe_cache_root() -> Path:
    try:
        return Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
    except Exception:
        return Path(gettempdir()) / ".zcu_tools_gui"


@dataclass(frozen=True)
class RestoreOutcome:
    """Result of ``restore_all``: the per-tab report plus any file-level load
    error (corrupt / wrong-version → defaults were used, error set so the
    Controller can surface it)."""

    report: RestoreReport
    load_error: PersistenceError | None


class PersistenceCaretaker:
    def __init__(
        self,
        originator: PersistOriginatorPort,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self._originator = originator
        base_dir = cache_dir if cache_dir is not None else _safe_cache_root()
        self._path = base_dir / _STATE_FILENAME

    @property
    def state_path(self) -> Path:
        return self._path

    def restore_all(self, *, load: bool = True) -> RestoreOutcome:
        """Load the snapshot and hand it to the originator. A missing file or an
        invalid / wrong-version payload falls back to a default snapshot (Fast-
        Fail on shape, but tolerant of a fresh / incompatible install).

        ``load=False`` (a "clean" start) skips reading the file and restores a
        default snapshot instead — the originator still gets a snapshot so every
        service initialises to its default state. The on-disk file is untouched
        here; a later ``flush`` at close still overwrites it as usual."""
        if load:
            state, load_error = self._load()
        else:
            state, load_error = AppPersistedState(), None
        report = self._originator.restore_persisted_state(state)
        return RestoreOutcome(report=report, load_error=load_error)

    def flush(self) -> None:
        """Capture the current app state from the originator and write it.
        Trigger-agnostic — every call re-captures from scratch."""
        state = self._originator.capture_persisted_state()
        self._write(state)

    # ------------------------------------------------------------------

    def _load(self) -> tuple[AppPersistedState, PersistenceError | None]:
        if not self._path.exists():
            return AppPersistedState(), None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            state = AppPersistedState.model_validate(raw)
        except Exception as exc:  # JSONDecodeError / OSError / ValidationError
            return AppPersistedState(), PersistenceError(
                f"Failed to read GUI state ({exc}); using defaults"
            )
        if state.version != APP_STATE_VERSION:
            return AppPersistedState(), PersistenceError(
                f"Unsupported GUI state version {state.version!r} "
                f"(expected {APP_STATE_VERSION}); using defaults"
            )
        return state, None

    def _write(self, state: AppPersistedState) -> None:
        payload = state.model_dump(mode="json")
        temp_path: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as file:
                file.write(json.dumps(payload, ensure_ascii=True, indent=2))
                temp_path = Path(file.name)
            temp_path.replace(self._path)
        except (OSError, TypeError, ValueError) as exc:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise PersistenceError(f"Failed to save GUI state: {exc}") from exc


__all__ = ["PersistenceCaretaker", "RestoreOutcome"]
