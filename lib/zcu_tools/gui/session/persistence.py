"""Generic single-file persistence mechanism for GUI applications."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Generic, Protocol, TypeVar

from platformdirs import user_cache_dir

StateT = TypeVar("StateT")
ReportT = TypeVar("ReportT")
ReportT_co = TypeVar("ReportT_co", covariant=True)

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """Expected failure while loading or saving a GUI snapshot."""


class SnapshotCodec(Protocol[StateT]):
    """App-owned snapshot schema boundary."""

    def default(self) -> StateT: ...
    def decode(self, raw: object) -> StateT: ...
    def encode(self, state: StateT) -> object: ...


class SnapshotOriginator(Protocol[StateT, ReportT_co]):
    """Typed Memento originator consumed by the caretaker."""

    def capture_persisted_state(self) -> StateT: ...
    def restore_persisted_state(self, state: StateT) -> ReportT_co: ...


@dataclass(frozen=True)
class RestoreOutcome(Generic[ReportT]):
    report: ReportT
    load_error: PersistenceError | None


def _safe_cache_root() -> Path:
    try:
        return Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
    except Exception:
        return Path(gettempdir()) / ".zcu_tools_gui"


class SingleFileCaretaker(Generic[StateT, ReportT]):
    """Load and atomically replace one JSON snapshot file."""

    def __init__(
        self,
        originator: SnapshotOriginator[StateT, ReportT],
        *,
        codec: SnapshotCodec[StateT],
        filename: str,
        cache_dir: Path | None = None,
    ) -> None:
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a non-empty basename")
        self._originator = originator
        self._codec = codec
        base_dir = cache_dir if cache_dir is not None else _safe_cache_root()
        self._path = base_dir / filename

    @property
    def state_path(self) -> Path:
        return self._path

    def restore_all(self, *, load: bool = True) -> RestoreOutcome[ReportT]:
        logger.info("persistence restore: load=%s path=%s", load, self._path)
        if load:
            state, load_error = self._load()
        else:
            state, load_error = self._codec.default(), None
        report = self._originator.restore_persisted_state(state)
        return RestoreOutcome(report=report, load_error=load_error)

    def flush(self) -> None:
        logger.info("persistence flush: %s", self._path)
        temp_path: Path | None = None
        try:
            state = self._originator.capture_persisted_state()
            payload = self._codec.encode(state)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=self._path.parent, delete=False
            ) as file:
                temp_path = Path(file.name)
                file.write(json.dumps(payload, ensure_ascii=True, indent=2))
            temp_path.replace(self._path)
        except Exception as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "persistence temporary-file cleanup failed: %s",
                        temp_path,
                        exc_info=True,
                    )
            logger.exception("persistence save failed: %s", self._path)
            raise PersistenceError(f"Failed to save GUI state: {exc}") from exc

    def _load(self) -> tuple[StateT, PersistenceError | None]:
        try:
            if not self._path.exists():
                return self._codec.default(), None
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return self._codec.decode(raw), None
        except Exception as exc:
            logger.warning(
                "persistence load failed (%s); using defaults",
                self._path,
                exc_info=exc,
            )
            return self._codec.default(), PersistenceError(
                f"Failed to read GUI state ({exc}); using defaults"
            )


__all__ = [
    "PersistenceError",
    "RestoreOutcome",
    "SingleFileCaretaker",
    "SnapshotCodec",
    "SnapshotOriginator",
]
