"""Generic single-file persistence mechanism for GUI applications."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generic, Protocol, TypeVar

from zcu_tools.utils.cache import user_cache_root

StateT = TypeVar("StateT")
ReportT = TypeVar("ReportT")
ReportT_co = TypeVar("ReportT_co", covariant=True)

logger = logging.getLogger(__name__)


class PersistenceError(RuntimeError):
    """Expected failure while loading or saving a GUI snapshot."""


class SnapshotCodec(Protocol[StateT]):
    """App-owned snapshot schema boundary.

    decode raises ValueError for incompatible persisted data. Unexpected provider
    or programming failures must use other exception types and propagate.
    """

    def default(self) -> StateT: ...
    def decode(self, raw: object) -> StateT: ...
    def encode(self, state: StateT) -> object: ...


class SnapshotOriginator(Protocol[StateT, ReportT_co]):
    """Typed Memento originator consumed by the caretaker."""

    def capture_persisted_state(self) -> StateT: ...
    def restore_persisted_state(self, state: StateT) -> ReportT_co: ...


@dataclass(frozen=True)
class RestoreOutcome(Generic[ReportT]):
    """Completed originator restore and any recoverable startup failure.

    report is the app-owned result of the installed state. load_error is None on
    clean/missing/successful reads, otherwise the error the host must display;
    invalid persisted data opens defaults without changing the original file.
    """

    report: ReportT
    load_error: PersistenceError | None


class SingleFileCaretaker(Generic[StateT, ReportT]):
    """Load and atomically replace one JSON snapshot file.

    Invalid JSON/UTF-8 and codec ValueError open defaults with a visible error.
    Loading never renames or rewrites the original file. Other provider/restore
    failures propagate. Hosts display RestoreOutcome.load_error and own save
    timing; recovered startup does not disable later flush.

    cache_dir selects an explicit directory, bypassing user cache resolution.
    When omitted, use <user_cache_root()>/gui; invalid ZCU_TOOLS_CACHE_DIR values
    raise ValueError during construction.
    """

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
        base_dir = cache_dir if cache_dir is not None else user_cache_root() / "gui"
        self._path = base_dir / filename

    @property
    def state_path(self) -> Path:
        return self._path

    def restore_all(self, *, load: bool = True) -> RestoreOutcome[ReportT]:
        """Restore persisted state or defaults and return the host display report.

        load=False skips reading an existing file. Invalid persisted data opens
        defaults without writing. Read I/O errors retain the existing
        default/error policy.
        Unclassified provider or originator failures propagate unchanged.
        """
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
            text = self._path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return self._codec.default(), self._load_error(exc)
        except OSError as exc:
            logger.warning("persistence read failed: %s", self._path, exc_info=exc)
            return self._codec.default(), PersistenceError(
                f"Failed to read GUI state ({exc}); using defaults"
            )
        try:
            return self._codec.decode(json.loads(text)), None
        except ValueError as exc:
            return self._codec.default(), self._load_error(exc)

    def _load_error(self, cause: Exception) -> PersistenceError:
        error = PersistenceError(f"Failed to read GUI state ({cause}); using defaults")
        logger.warning("%s", error)
        return error


__all__ = [
    "PersistenceError",
    "RestoreOutcome",
    "SingleFileCaretaker",
    "SnapshotCodec",
    "SnapshotOriginator",
]
