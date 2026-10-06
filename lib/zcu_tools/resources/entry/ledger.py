"""Append-only per-entry ledger and locally owned JSON record attachments."""

import errno
import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from filelock import FileLock, Timeout
from pydantic import TypeAdapter

from .ledger_models import JsonObject, LedgerEvent

_JSON_OBJECT = TypeAdapter(JsonObject)


@dataclass(frozen=True)
class LedgerEntry:
    """A lookup result: typed event, optional local record and original JSON line.

    event_json retains whitespace and unknown future-minor fields, without LF.
    record is the referenced JSON object or None; historical import paths are
    never followed. Each lookup reads independent data from disk.
    """

    event: LedgerEvent
    record: JsonObject | None
    event_json: str


class RecordsLedger:
    """Serialize reads/appends with one resolved ledger.jsonl.lock sidecar.

    No cache, event producer, repair, rollback or cross-file transaction is
    provided. I/O errors propagate. Malformed envelopes or JSON raise ValueError
    with the ledger path and line number. Lock acquisition raises native Timeout
    with diagnostic notes. A failed append may leave an orphan record or partial
    tail; later operations report that tail rather than skipping it.
    """

    def __init__(
        self, records: Path, *, entry_id: str, lock_timeout: float = 10.0
    ) -> None:
        """Bind an existing records directory and UUID, without creating files.

        records is not the entry root or a filename. Missing/non-directory paths
        raise FileNotFoundError/NotADirectoryError. Invalid entry_id or non-finite,
        negative lock_timeout seconds raise ValueError. All handles resolve the
        same directory before choosing their ledger and lock paths.
        """
        try:
            UUID(entry_id)
        except ValueError as exc:
            raise ValueError("entry_id must be a UUID") from exc
        if not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise ValueError("lock_timeout must be finite nonnegative seconds")
        self._records = records.resolve()
        self._records.stat()
        if not self._records.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(self._records)
            )
        self._entry_id = entry_id
        self._timeout = lock_timeout
        self._path = self._records / "ledger.jsonl"
        self._lock = FileLock(str(self._path) + ".lock")

    @contextmanager
    def _locked(self) -> Generator[None]:
        try:
            self._lock.acquire(timeout=self._timeout)
        except Timeout as exc:
            exc.add_note(f"{self._path}: lock_timeout={self._timeout} seconds")
            raise
        try:
            yield
        finally:
            self._lock.release()

    def append(self, event: LedgerEvent, *, record: JsonObject | None = None) -> None:
        """Append one caller-identified 1.0 event; optionally attach JSON content.

        event.record must be None. An attachment is published under <id>.json and
        its canonical reference added to a copy, never to the input event. Under
        the lock, validate all historical lines before appending and flushing a
        complete UTF-8 line. Duplicate IDs, wrong entry_id, invalid content and
        existing attachments are rejected without overwriting history. After an
        I/O failure only unpublished temporary files are removed, not evidence.
        """
        if event.record is not None:
            raise ValueError(f"{self._path}: caller event.record must be None")
        validated = LedgerEvent.model_validate(event.model_dump())
        if validated.format_version != "1.0":
            raise ValueError(f"{self._path}: append requires format_version 1.0")
        if validated.entry_id != self._entry_id:
            raise ValueError(f"{self._path}: event.entry_id belongs to another entry")
        attachment = None
        if record is not None:
            try:
                attachment = _JSON_OBJECT.validate_python(record)
            except ValueError as exc:
                raise ValueError(f"{self._path}: invalid record: {exc}") from exc
            validated = validated.model_copy(
                update={"record": f"records/{validated.id}.json"}
            )
        line = validated.model_dump_json() + "\n"
        with self._locked():
            entries = self._read()
            if any(item.event.id == validated.id for item in entries):
                raise ValueError(f"{self._path}: duplicate event ID {validated.id!r}")
            if attachment is not None:
                self._publish_record(validated.id, attachment)
            with self._path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(line)
                stream.flush()

    def events(self) -> tuple[LedgerEvent, ...]:
        """Return validated events in append order without reading attachments.

        An existing records directory with no ledger returns (). Corrupt rows,
        duplicate IDs, foreign entry identities and incomplete tails raise;
        successful reads never rewrite ledger bytes.
        """
        with self._locked():
            return tuple(item.event for item in self._read())

    def get(self, event_id: str) -> LedgerEntry:
        """Read an event and its local JSON attachment under the ledger lock.

        Unknown IDs raise KeyError, including when the ledger has not been
        created. Missing attachments raise FileNotFoundError with their physical
        path; malformed attachment JSON raises ValueError with that path. Imports
        query only the outer record, not the historical source_record.
        """
        with self._locked():
            entries = self._read()
            for item in entries:
                if item.event.id == event_id:
                    record = None
                    if item.event.record is not None:
                        path = self._records / f"{event_id}.json"
                        try:
                            content = path.read_text(encoding="utf-8")
                            record = _JSON_OBJECT.validate_json(content)
                        except ValueError as exc:
                            raise ValueError(f"{path}: invalid record: {exc}") from exc
                    return LedgerEntry(item.event, record, item.event_json)
        raise KeyError(event_id)

    def _read(self) -> tuple[LedgerEntry, ...]:
        try:
            stream = self._path.open("rb")
        except FileNotFoundError:
            # Only an absent ledger in an existing directory means no events.
            self._records.stat()
            if self._path.is_symlink():
                raise
            return ()
        entries: list[LedgerEntry] = []
        ids: set[str] = set()
        with stream:
            for number, encoded_line in enumerate(stream, start=1):
                try:
                    line = encoded_line.decode("utf-8")
                    if not line.endswith("\n"):
                        raise ValueError("incomplete tail: expected LF")
                    raw = line[:-1]
                    data = _JSON_OBJECT.validate_json(raw)
                    if "format" not in data or "format_version" not in data:
                        raise ValueError("missing format or format_version header")
                    version = data.get("format_version")
                    # The model validates version syntax before unknown fields
                    # are projected; the raw envelope remains in event_json.
                    future = isinstance(version, str) and version != "1.0"
                    event = LedgerEvent.model_validate(
                        data, extra="ignore" if future else "forbid"
                    )
                    if event.entry_id != self._entry_id:
                        raise ValueError("entry_id belongs to another entry")
                    if event.id in ids:
                        raise ValueError(f"duplicate event ID {event.id!r}")
                except ValueError as exc:
                    raise ValueError(f"{self._path}:{number}: {exc}") from exc
                ids.add(event.id)
                entries.append(LedgerEntry(event, None, raw))
        return tuple(entries)

    def _publish_record(self, event_id: str, content: JsonObject) -> None:
        path = self._records / f"{event_id}.json"
        if path.exists() or path.is_symlink():
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(path))
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self._records, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(_JSON_OBJECT.dump_json(content).decode("utf-8") + "\n")
                stream.flush()
                stream.close()
                temporary.replace(path)
            finally:
                # replace consumed it on success; never remove a published record.
                temporary.unlink(missing_ok=True)
