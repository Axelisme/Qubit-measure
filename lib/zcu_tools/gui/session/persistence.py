"""Generic single-file persistence mechanism for GUI applications."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
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


class SnapshotIdentityError(ValueError):
    """Saved identity is incompatible; originator has not changed any state.

    Originators raise this only for trusted snapshot identity checks, never for
    provider, validation or rendering failures. The caretaker backs up the file
    before restoring defaults. Other restore exceptions propagate unchanged.
    """


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
    incompatible data errors include the preserved sibling backup path.
    """

    report: ReportT
    load_error: PersistenceError | None


def _safe_cache_root() -> Path:
    try:
        return Path(user_cache_dir("zcu_tools", "zcu_tools")) / "gui"
    except Exception:
        return Path(gettempdir()) / ".zcu_tools_gui"


class SingleFileCaretaker(Generic[StateT, ReportT]):
    """Load and atomically replace one JSON snapshot file.

    Invalid JSON/UTF-8, codec ValueError and SnapshotIdentityError from restore
    move the original to a unique timestamped sibling before default startup.
    Backup failure raises PersistenceError, without installing defaults. Other
    provider/restore failures propagate. Hosts display RestoreOutcome.load_error
    and own save timing; recovered startup does not disable later flush.
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
        base_dir = cache_dir if cache_dir is not None else _safe_cache_root()
        self._path = base_dir / filename

    @property
    def state_path(self) -> Path:
        return self._path

    def restore_all(self, *, load: bool = True) -> RestoreOutcome[ReportT]:
        """Restore persisted state or defaults and return the host display report.

        load=False skips reading/backing up an existing file. Incompatibility
        backs up the original before restoring defaults; failed backup aborts
        startup. Read I/O errors retain the existing default/error policy.
        Unclassified provider or originator failures propagate unchanged.
        """
        logger.info("persistence restore: load=%s path=%s", load, self._path)
        if load:
            state, load_error = self._load()
        else:
            state, load_error = self._codec.default(), None
        try:
            report = self._originator.restore_persisted_state(state)
        except SnapshotIdentityError as exc:
            if not load or load_error is not None:
                raise
            load_error = self._backup_incompatible(exc)
            report = self._originator.restore_persisted_state(self._codec.default())
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
            return self._codec.default(), self._backup_incompatible(exc)
        except OSError as exc:
            logger.warning("persistence read failed: %s", self._path, exc_info=exc)
            return self._codec.default(), PersistenceError(
                f"Failed to read GUI state ({exc}); using defaults"
            )
        try:
            return self._codec.decode(json.loads(text)), None
        except ValueError as exc:
            return self._codec.default(), self._backup_incompatible(exc)

    def _backup_incompatible(self, cause: Exception) -> PersistenceError:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        backup: Path | None = None
        try:
            # Reserve a unique sibling before rename; never overwrite an old backup.
            with NamedTemporaryFile(
                dir=self._path.parent,
                prefix=f"{self._path.name}.incompatible-{timestamp}-",
                delete=False,
            ) as file:
                backup = Path(file.name)
            self._path.replace(backup)
        except OSError as exc:
            if backup is not None:
                backup.unlink(missing_ok=True)
            raise PersistenceError(
                f"Failed to back up incompatible GUI state {self._path}: {exc}"
            ) from exc
        error = PersistenceError(
            f"Failed to read GUI state ({cause}); backed up to {backup}; using defaults"
        )
        logger.warning("%s", error)
        return error


__all__ = [
    "PersistenceError",
    "RestoreOutcome",
    "SingleFileCaretaker",
    "SnapshotCodec",
    "SnapshotIdentityError",
    "SnapshotOriginator",
]
