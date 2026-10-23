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

from zcu_tools.utils import atomic_replace

from .ledger_models import JsonObject, LedgerEvent

_JSON_OBJECT = TypeAdapter(JsonObject)


@dataclass(frozen=True)
class LedgerEntry:
    """A lookup result: typed event, optional local record and original JSON line.

    event_json retains whitespace and unknown future-minor fields, without LF.
    record is the referenced JSON object or None; historical import paths are
    never followed. Each lookup returns isolated data and rereads its attachment.
    """

    event: LedgerEvent
    record: JsonObject | None
    event_json: str


@dataclass(frozen=True)
class _LedgerState:
    """Validated prefix and ID index, published only after the whole tail passes."""

    identity: tuple[int, int]
    modified: tuple[int, int]
    offset: int
    head: bytes
    last_line: bytes
    entries: dict[str, LedgerEntry]


class RecordsLedger:
    """Serialize reads/appends with one resolved ledger.jsonl.lock sidecar.

    Each handle indexes validated lines and parses only appended bytes. A shorter
    file, changed first 4096 bytes, changed previous last line or changed file
    identity triggers a full reread, as does a modified file of unchanged size.
    The ledger is append-only: an in-place middle rewrite combined with growth,
    preserving the file identity, head and previous last line, is not detected.
    Call reload() to validate the complete history after external rewrites.
    Attachments are never cached. No event producer, repair, rollback or cross-file
    transaction is provided. I/O errors propagate. Malformed envelopes or JSON
    raise ValueError with the ledger path and line number. Lock acquisition raises native Timeout
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
        self._state: _LedgerState | None = None

    @property
    def source_path(self) -> Path:
        """Return the bound absolute ledger.jsonl path without file I/O.

        The path identifies this handle's storage for diagnostics. Reading it
        does not create the ledger, refresh the index or access attachments.
        """
        return self._path

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
            if validated.id in entries:
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
            return tuple(
                item.event.model_copy(deep=True) for item in self._read().values()
            )

    def reload(self) -> tuple[LedgerEvent, ...]:
        """Fully validate history from byte zero and replace this handle's index.

        Return isolated events in append order without reading attachments, as
        events() does. Use this after external non-append history edits that may
        preserve the incremental read boundaries. Errors propagate without
        publishing a partially validated index or rewriting the ledger.
        """
        with self._locked():
            return tuple(
                item.event.model_copy(deep=True)
                for item in self._read(force_reload=True).values()
            )

    def dependencies_exist(self, event_ids: tuple[str, ...]) -> bool:
        """Check event IDs and local attachment existence without parsing records.

        Empty input returns True without I/O. Otherwise hold the ledger lock,
        update the incremental index and check each unique ID and its referenced
        local attachment with stat. A missing ID or attachment returns False.
        Ledger schema, lock and other I/O errors propagate. Attachment content is
        not read or validated; imports never follow historical source_record.
        Callers that require content validation must use get/get_many/find_many.
        """
        if not event_ids:
            return True
        with self._locked():
            entries = self._read()
            for event_id in dict.fromkeys(event_ids):
                item = entries.get(event_id)
                if item is None:
                    return False
                if item.event.record is not None:
                    try:
                        (self._records / f"{event_id}.json").stat()
                    except FileNotFoundError:
                        return False
            return True

    def get(self, event_id: str) -> LedgerEntry:
        """Read an event and its local JSON attachment under the ledger lock.

        Unknown IDs raise KeyError, including when the ledger has not been
        created. Missing attachments raise FileNotFoundError with their physical
        path; malformed attachment JSON raises ValueError with that path. Imports
        query only the outer record, not the historical source_record.
        """
        return self.get_many((event_id,))[0]

    def get_many(self, event_ids: tuple[str, ...]) -> tuple[LedgerEntry, ...]:
        """Read requested events and attachments with one locked ledger parse.

        Return entries in input order, including repeated IDs. Each unique ID
        and attachment is checked once per call. Validated ledger lines are indexed;
        attachments are read from disk on every call.
        The first missing ID raises KeyError; earlier attachment failures
        propagate just as in get(). Unrequested attachments are not read.
        Empty input returns () without accessing the ledger.
        """
        if not event_ids:
            return ()
        with self._locked():
            entries = self._read()
            found: dict[str, LedgerEntry] = {}
            for event_id in dict.fromkeys(event_ids):
                item = entries[event_id]
                found[event_id] = self._with_record(item)
            return tuple(found[event_id] for event_id in event_ids)

    def find_many(self, event_ids: tuple[str, ...]) -> dict[str, LedgerEntry | None]:
        """Look up all IDs with one locked parse, mapping absent IDs to None.

        Keys follow first occurrence in event_ids. Each unique requested event
        and its local attachment is checked once; unrequested attachments are
        not read. Empty input returns {} without I/O. Ledger schema, attachment
        and I/O errors propagate, including after an absent ID. Results are
        isolated copies with freshly read attachments. Use get_many for strict
        lookup that raises at the first absent ID.
        """
        if not event_ids:
            return {}
        with self._locked():
            entries = self._read()
            found: dict[str, LedgerEntry | None] = {}
            for event_id in dict.fromkeys(event_ids):
                item = entries.get(event_id)
                found[event_id] = self._with_record(item) if item is not None else None
            return found

    def _with_record(self, item: LedgerEntry) -> LedgerEntry:
        record = None
        if item.event.record is not None:
            path = self._records / f"{item.event.id}.json"
            try:
                content = path.read_text(encoding="utf-8")
                record = _JSON_OBJECT.validate_json(content)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid record: {exc}") from exc
        return LedgerEntry(item.event.model_copy(deep=True), record, item.event_json)

    def _read(self, *, force_reload: bool = False) -> dict[str, LedgerEntry]:
        try:
            stream = self._path.open("rb")
        except FileNotFoundError:
            # Only an absent ledger in an existing directory means no events.
            self._records.stat()
            if self._path.is_symlink():
                raise
            self._state = None
            return {}
        with stream:
            status = os.fstat(stream.fileno())
            identity = (status.st_dev, status.st_ino)
            modified = (status.st_mtime_ns, status.st_ctime_ns)
            previous = None if force_reload else self._state
            if previous is not None:
                head = stream.read(len(previous.head))
                stream.seek(previous.offset - len(previous.last_line))
                last_line = stream.read(len(previous.last_line))
                if (
                    identity != previous.identity
                    or status.st_size < previous.offset
                    or head != previous.head
                    or last_line != previous.last_line
                    or (
                        status.st_size == previous.offset
                        and modified != previous.modified
                    )
                ):
                    previous = None
            entries = previous.entries if previous is not None else {}
            stream.seek(previous.offset if previous is not None else 0)
            added: dict[str, LedgerEntry] = {}
            last_line = previous.last_line if previous is not None else b""
            for number, encoded_line in enumerate(stream, start=len(entries) + 1):
                try:
                    item = self._parse_entry(encoded_line)
                    if item.event.id in entries or item.event.id in added:
                        raise ValueError(f"duplicate event ID {item.event.id!r}")
                except ValueError as exc:
                    raise ValueError(f"{self._path}:{number}: {exc}") from exc
                added[item.event.id] = item
                last_line = encoded_line
            offset = stream.tell()
            # A bounded prefix detects replacement without rereading growing history.
            stream.seek(0)
            head = stream.read(min(offset, 4096))
        # Do not publish any prefix of a malformed tail or a failed I/O operation.
        entries.update(added)
        self._state = _LedgerState(identity, modified, offset, head, last_line, entries)
        return entries

    def _parse_entry(self, encoded_line: bytes) -> LedgerEntry:
        """Decode one complete line and validate its envelope and entry identity."""
        line = encoded_line.decode("utf-8")
        if not line.endswith("\n"):
            raise ValueError("incomplete tail: expected LF")
        raw = line[:-1]
        data = _JSON_OBJECT.validate_json(raw)
        if "format" not in data or "format_version" not in data:
            raise ValueError("missing format or format_version header")
        # Reading projects known fields at every supported minor;
        # event_json retains the complete original envelope.
        event = LedgerEvent.model_validate(
            data, extra="ignore", context={"ledger_read": True}
        )
        if event.entry_id != self._entry_id:
            raise ValueError("entry_id belongs to another entry")
        return LedgerEntry(event, None, raw)

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
                atomic_replace(temporary, path)
            finally:
                # replace consumed it on success; never remove a published record.
                temporary.unlink(missing_ok=True)
