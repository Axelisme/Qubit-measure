"""Append-only per-entry ledger and locally owned JSON record attachments."""

import errno
import math
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from filelock import FileLock, Timeout
from pydantic import TypeAdapter

from zcu_tools.resources.file_lock import lock_path_for
from zcu_tools.utils import atomic_replace

from .ledger_models import JsonObject, LedgerEvent, LedgerEventContent

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
class _IndexedEntry:
    content: LedgerEventContent
    event_json: str


@dataclass(frozen=True)
class _LedgerState:
    """Validated prefix and ID index, published only after the whole tail passes."""

    identity: tuple[int, int]
    modified: tuple[int, int]
    offset: int
    head: bytes
    last_line: bytes
    entries: dict[str, _IndexedEntry]


class RecordsLedger:
    """Serialize reads/appends with one per-user runtime lock for ledger.jsonl.

    Each handle indexes header/identity-checked JSON and parses only appended bytes. A shorter
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

    def __init__(self, records: Path, *, lock_timeout: float = 10.0) -> None:
        """Bind a records directory without creating files.

        records is not the entry root or a filename. A missing directory is empty
        until append creates it; its parent must exist for append. Existing
        non-directory paths raise NotADirectoryError. Non-finite or negative
        lock_timeout seconds raise ValueError. All handles resolve the
        same directory before choosing their ledger and runtime lock paths.
        Binding creates the runtime lock directory on demand, not a lock file.
        """
        if not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise ValueError("lock_timeout must be finite nonnegative seconds")
        self._records = records.resolve()
        if self._records.exists() and not self._records.is_dir():
            raise NotADirectoryError(
                errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(self._records)
            )
        self._timeout = lock_timeout
        self._path = self._records / "ledger.jsonl"
        self._lock = FileLock(str(lock_path_for(self._path)))
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
        the lock, check historical JSON/header/event and run references, duplicate
        IDs and complete tails before flushing a UTF-8 line. Unused historical
        payload fields are not projected. This new event is fully validated.
        Historical entry_id is ignored, not membership.
        Duplicate IDs, invalid content and
        existing attachments are rejected without overwriting history. After an
        I/O failure only unpublished temporary files are removed, not evidence.
        """
        if event.record is not None:
            raise ValueError(f"{self._path}: caller event.record must be None")
        validated = LedgerEvent.model_validate(event.model_dump())
        if validated.format_version != "1.0":
            raise ValueError(f"{self._path}: append requires format_version 1.0")
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
            self._records.mkdir(exist_ok=True)
            if attachment is not None:
                self._publish_record(validated.id, attachment)
            with self._path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(line)
                stream.flush()

    def events(self) -> tuple[LedgerEvent, ...]:
        """Return validated events in append order without reading attachments.

        A missing records directory or ledger returns () without creating it. Corrupt rows,
        duplicate IDs and incomplete tails raise; historical entry_id is ignored;
        successful reads never rewrite ledger bytes.
        """
        with self._locked():
            return tuple(item.content.project() for item in self._read().values())

    def read_events_snapshot(self) -> tuple[LedgerEvent, ...]:
        """Read isolated validated events without acquiring or creating a lock.

        Use for read-only browsing, not read-modify-write transactions. Read the
        complete current ledger; missing ledger returns (). Validation matches
        events(), including duplicates and complete LF tails.
        A concurrent append may expose an incomplete tail and raise ValueError;
        callers may refresh later. Attachments are not opened and no files are
        created. I/O errors propagate. This does not update the incremental index.
        """
        return tuple(
            item.content.project()
            for item in self._read(force_reload=True, publish_state=False).values()
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
                item.content.project()
                for item in self._read(force_reload=True).values()
            )

    def content(self, event_id: str) -> LedgerEventContent:
        """Read one immutable partial event without projecting payload or attachment.

        JSON/header/event and run references, duplicate IDs and tail are checked.
        Unknown event IDs raise KeyError. Leaf reads report requested field errors.
        """
        with self._locked():
            return self._read()[event_id].content

    def contents(self) -> tuple[LedgerEventContent, ...]:
        """List immutable partial events in append order, without attachments.

        Index validation matches content(). Other fields remain opaque until used.
        No ledger bytes are written; a missing ledger returns an empty tuple.
        """
        with self._locked():
            return tuple(item.content for item in self._read().values())

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

        Return entries in input order, including repeated IDs. Each requested
        complete event and attachment is checked once per call. Partial lines are indexed;
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

    def _with_record(self, item: _IndexedEntry) -> LedgerEntry:
        event = item.content.project()
        record = None
        if event.record is not None:
            path = self._records / f"{event.id}.json"
            if not path.resolve().is_relative_to(self._records):
                raise ValueError(f"{path}: record escapes its root")
            try:
                content = path.read_text(encoding="utf-8")
                record = _JSON_OBJECT.validate_json(content)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid record: {exc}") from exc
        return LedgerEntry(event, record, item.event_json)

    def _read(
        self, *, force_reload: bool = False, publish_state: bool = True
    ) -> dict[str, _IndexedEntry]:
        try:
            stream = self._path.open("rb")
        except FileNotFoundError:
            if self._path.is_symlink():
                raise
            if publish_state:
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
            added: dict[str, _IndexedEntry] = {}
            last_line = previous.last_line if previous is not None else b""
            for number, encoded_line in enumerate(stream, start=len(entries) + 1):
                try:
                    item = self._parse_entry(encoded_line, number=number)
                    event_id = item.content.id
                    if event_id in entries or event_id in added:
                        raise ValueError(f"duplicate event ID {event_id!r}")
                except ValueError as exc:
                    raise ValueError(f"{self._path}:{number}: {exc}") from exc
                added[event_id] = item
                last_line = encoded_line
            offset = stream.tell()
            # A bounded prefix detects replacement without rereading growing history.
            stream.seek(0)
            head = stream.read(min(offset, 4096))
        # Do not publish any prefix of a malformed tail or a failed I/O operation.
        entries.update(added)
        if publish_state:
            self._state = _LedgerState(
                identity, modified, offset, head, last_line, entries
            )
        return entries

    def _parse_entry(self, encoded_line: bytes, *, number: int) -> _IndexedEntry:
        """Decode one complete line and validate only required index information."""
        line = encoded_line.decode("utf-8")
        if not line.endswith("\n"):
            raise ValueError("incomplete tail: expected LF")
        raw = line[:-1]
        data = _JSON_OBJECT.validate_json(raw)
        if "format" not in data or "format_version" not in data:
            raise ValueError("missing format or format_version header")
        content = LedgerEventContent(data, source=f"{self._path}:{number}")
        content.validate_index()
        return _IndexedEntry(content, raw)

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
