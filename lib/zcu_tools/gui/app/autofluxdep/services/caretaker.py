"""PersistenceCaretaker for autofluxdep-gui workflow state."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Protocol

from platformdirs import user_cache_dir

from .persistence_types import APP_STATE_VERSION, AppPersistedState, PersistenceError

logger = logging.getLogger(__name__)

_STATE_FILENAME = "autofluxdep_state_v1.json"


class PersistOriginatorPort(Protocol):
    """Narrow originator surface the caretaker depends on."""

    def capture_persisted_state(self) -> AppPersistedState: ...
    def restore_persisted_state(self, state: AppPersistedState) -> object: ...


def _safe_cache_root() -> Path:
    try:
        return Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
    except Exception:
        return Path(gettempdir()) / ".zcu_tools_gui"


@dataclass(frozen=True)
class RestoreOutcome:
    report: object
    load_error: PersistenceError | None


class PersistenceCaretaker:
    """Single-file Memento caretaker for autofluxdep workflow state."""

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
        logger.info(
            "autofluxdep persistence restore: load=%s path=%s", load, self._path
        )
        if load:
            state, load_error = self._load()
        else:
            state, load_error = AppPersistedState(), None
        report = self._originator.restore_persisted_state(state)
        return RestoreOutcome(report=report, load_error=load_error)

    def flush(self) -> None:
        logger.info("autofluxdep persistence flush: %s", self._path)
        try:
            state = self._originator.capture_persisted_state()
            self._write(state)
        except PersistenceError:
            raise
        except Exception as exc:
            logger.exception(
                "autofluxdep persistence capture/write failed: %s", self._path
            )
            raise PersistenceError(
                f"Failed to save autofluxdep GUI state: {exc}"
            ) from exc

    def _load(self) -> tuple[AppPersistedState, PersistenceError | None]:
        if not self._path.exists():
            return AppPersistedState(), None
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            state = AppPersistedState.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "autofluxdep persistence load failed (%s); using defaults",
                self._path,
                exc_info=exc,
            )
            return AppPersistedState(), PersistenceError(
                f"Failed to read autofluxdep GUI state ({exc}); using defaults"
            )
        if state.version != APP_STATE_VERSION:
            logger.warning(
                "autofluxdep persistence version mismatch: got %r expected %r",
                state.version,
                APP_STATE_VERSION,
            )
            return AppPersistedState(), PersistenceError(
                f"Unsupported autofluxdep GUI state version {state.version!r} "
                f"(expected {APP_STATE_VERSION}); using defaults"
            )
        return state, None

    def _write(self, state: AppPersistedState) -> None:
        temp_path: Path | None = None
        try:
            payload = state.model_dump(mode="json")
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
            logger.exception("autofluxdep persistence write failed: %s", self._path)
            raise PersistenceError(
                f"Failed to save autofluxdep GUI state: {exc}"
            ) from exc


__all__ = ["PersistenceCaretaker", "RestoreOutcome"]
