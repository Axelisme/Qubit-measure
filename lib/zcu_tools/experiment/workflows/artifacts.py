"""Journal, manifest, allocated commit directories and atomic JSON publication."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, JsonValue, TypeAdapter

from zcu_tools.utils import atomic_replace

from .journal import JournalPayload, RecordCommitted, RunMetadata, RunStarted
from .models import EncodedRecord, Lifecycle, OperationFact, RunIdentity, RunPaths
from .ports import Clock

_logger = logging.getLogger(__name__)
_json_adapter = TypeAdapter(JsonValue)


@dataclass(frozen=True)
class RunManifest:
    """Run entry metadata, not a checkpoint or original-data inventory.

    identity/workflow/plan/initial_tunables preserve startup values. roots are
    absolute and journal names JSONL. started_at/updated_at are UTC; ended_at is
    None before terminal. lifecycle/reason summarize control. record_schemas
    maps each committed model's qualified type name to its JSON schema once.
    Schemas are informational; readers do not validate against them.
    format/version identify this manifest document.
    """

    identity: RunIdentity
    workflow: str
    plan: JsonValue
    initial_tunables: JsonValue
    roots: RunPaths
    journal: Path
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None
    lifecycle: Lifecycle
    reason: str | None
    record_schemas: dict[str, JsonValue] = field(default_factory=dict)
    format: Literal["zcu-workflow-run"] = field(default="zcu-workflow-run", init=False)
    format_version: Literal[1] = field(default=1, init=False)


@dataclass(frozen=True)
class RecordStarted:
    """Commit-scope evidence: call_seq, prior commit_seq and read-order revisions."""

    call_seq: int
    commit_seq: int
    revisions: tuple[int, ...]


@dataclass(frozen=True)
class RecordEnvelope:
    """One persisted commit, independent of its directory allocation number.

    run_id/workflow bind the manifest. commit_seq orders committed records.
    directory is data-root-relative commits/<allocation number>. committed_at
    is UTC. started retains scope identity and actual tunables reads.
    record_type keys the informational manifest schema. record is encoded JSON.
    format/version identify the record file schema.
    """

    run_id: str
    workflow: str
    commit_seq: int
    directory: str
    committed_at: datetime
    started: RecordStarted
    record_type: str
    record: EncodedRecord
    operations: tuple[OperationFact, ...] = ()
    format: Literal["zcu-workflow-record"] = field(
        default="zcu-workflow-record", init=False
    )
    format_version: Literal[2] = field(default=2, init=False)


def _json_object[T](value: T, adapter: TypeAdapter[T]) -> dict[str, JsonValue]:
    projected = _json_adapter.validate_python(
        adapter.dump_python(value, mode="json", warnings="error")
    )
    if not isinstance(projected, dict):
        raise TypeError("Artifact payload must encode as a JSON object")
    return projected


def _json_line(value: dict[str, JsonValue]) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


def _check_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("Artifact timestamps must be timezone-aware UTC")


def _write_atomic(destination: Path, payload: dict[str, JsonValue]) -> None:
    temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}.tmp.json")
    with temporary.open("x", encoding="utf-8", newline="\n") as writer:
        line = _json_line(payload)
        if writer.write(line) != len(line):
            raise OSError("Artifact write was incomplete")
        writer.flush()
    atomic_replace(temporary, destination)


class RunArtifacts:
    """Own framework JSON only; never discover or inspect authored raw files.

    metadata supplies detached startup evidence. clock returns aware UTC. lock
    is the Engine control lock, shared by allocation, journal and publication.
    Construction rejects existing/overlapping roots and persists startup. All
    I/O failures propagate except best-effort committed record.json output.
    No fsync/crash resume, file rollback or hardware ownership is provided.
    """

    def __init__(
        self, metadata: RunMetadata, clock: Clock, control_lock: RLock
    ) -> None:
        roots = RunPaths(
            metadata.roots.metadata_root.resolve(), metadata.roots.data_root.resolve()
        )
        self._check_roots(roots)
        _check_utc(metadata.started_at)
        self._clock = clock
        self._lock = control_lock
        self._paths = roots
        self._run_id = metadata.identity.run_id
        self._seq = 0
        self._allocation_seq = 0
        self._allocated: set[Path] = set()
        self._committed: set[Path] = set()
        self._journal_broken = False
        self._manifest = RunManifest(
            deepcopy(metadata.identity),
            metadata.workflow,
            deepcopy(metadata.plan),
            deepcopy(metadata.tunables),
            roots,
            self.journal_path,
            metadata.started_at,
            metadata.started_at,
            None,
            "running",
            None,
        )
        payload = _json_object(self._manifest, TypeAdapter(RunManifest))
        _json_line(payload)
        roots.metadata_root.mkdir(parents=True, exist_ok=False)
        roots.data_root.mkdir(parents=True, exist_ok=False)
        _write_atomic(roots.metadata_root / "manifest.json", payload)
        self.journal_path.touch(exist_ok=False)
        self.append(
            RunStarted(
                deepcopy(metadata.identity),
                metadata.workflow,
                deepcopy(metadata.plan),
                deepcopy(metadata.tunables),
                roots,
                metadata.started_at,
            )
        )

    @staticmethod
    def _check_roots(roots: RunPaths) -> None:
        first, second = roots.metadata_root, roots.data_root
        if first == second or first in second.parents or second in first.parents:
            raise ValueError("Run roots must be distinct and non-overlapping")
        for root in (first, second):
            if root.exists():
                raise FileExistsError(f"Run root already exists: {root}")

    @property
    def paths(self) -> RunPaths:
        """Return immutable normalized absolute run paths."""
        return self._paths

    @property
    def journal_path(self) -> Path:
        """Return metadata_root/journal.jsonl, never an authored file path."""
        return self._paths.metadata_root / "journal.jsonl"

    def append(self, event: JournalPayload) -> None:
        """Append/flush one typed payload with contiguous sequence and UTC time.

        Encoding errors consume no sequence. A failed/short write poisons later
        appends because a partial line may remain; no recovery is attempted.
        """
        with self._lock:
            if self._journal_broken:
                raise RuntimeError("Journal is unusable after an I/O failure")
            payload = _json_object(event, TypeAdapter(JournalPayload))
            now = self._clock.now()
            _check_utc(now)
            payload.update(
                format_version=1,
                seq=self._seq + 1,
                run_id=self._run_id,
                time=now.isoformat(),
            )
            line = _json_line(payload)
            try:
                with self.journal_path.open(
                    "a", encoding="utf-8", newline="\n"
                ) as writer:
                    if writer.write(line) != len(line):
                        raise OSError("Journal write was incomplete")
                    writer.flush()
            except OSError:
                self._journal_broken = True
                raise
            self._seq += 1

    def set_lifecycle(self, lifecycle: Lifecycle, reason: str | None = None) -> None:
        """Atomically replace lifecycle metadata; same-state changes raise ValueError."""
        with self._lock:
            if lifecycle == self._manifest.lifecycle:
                raise ValueError("Manifest updates require a lifecycle change")
            now = self._clock.now()
            _check_utc(now)
            manifest = replace(
                self._manifest,
                lifecycle=lifecycle,
                reason=reason,
                updated_at=now,
                ended_at=now if lifecycle in ("done", "cancelled", "failed") else None,
            )
            self._publish_manifest(manifest)

    def _publish_manifest(self, manifest: RunManifest) -> None:
        _write_atomic(
            self._paths.metadata_root / "manifest.json",
            _json_object(manifest, TypeAdapter(RunManifest)),
        )
        self._manifest = manifest

    def new_commit_dir(self) -> Path:
        """Allocate/create the next six-digit commits directory, independent of commits.

        Each call allocates a distinct directory. Failed creation does not
        consume a number. Allocation does not write a record or switch scopes.
        """
        with self._lock:
            number = self._allocation_seq + 1
            directory = self._paths.data_root / "commits" / f"{number:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            self._allocated.add(directory)
            self._allocation_seq = number
            return directory

    def prepare_commit(self, directory: Path | None) -> Path:
        """Allocate when None; otherwise reject foreign/already-committed directories."""
        with self._lock:
            if directory is None:
                return self.new_commit_dir()
            directory = directory.resolve()
            if directory not in self._allocated:
                raise ValueError(
                    f"Commit directory was not allocated by this run: {directory}"
                )
            if directory in self._committed:
                raise ValueError(f"Commit directory is already committed: {directory}")
            return directory

    def register_record(self, record: BaseModel) -> str:
        """Publish each qualified model's JSON schema once and return its name.

        A reused name with a different schema raises ValueError. Schema and
        manifest I/O errors propagate before any record is committed.
        """
        with self._lock:
            model = type(record)
            name = f"{model.__module__}.{model.__qualname__}"
            schema = _json_adapter.validate_python(model.model_json_schema())
            schemas = self._manifest.record_schemas
            if name in schemas:
                if schemas[name] != schema:
                    raise ValueError(f"Record schema changed for {name}")
                return name
            self._publish_manifest(
                replace(self._manifest, record_schemas={**schemas, name: schema})
            )
            return name

    def publish_record(
        self, directory: Path, event: RecordCommitted, started: RecordStarted
    ) -> None:
        """Journal/flush a commit, then best-effort write its matching record.json.

        The caller holds the same lock through in-memory record publication.
        Record output errors are logged; the flushed journal remains authority.
        Authored files are not inspected or rolled back.
        """
        with self._lock:
            self.append(event)
            self._committed.add(directory)
            envelope = RecordEnvelope(
                self._run_id,
                self._manifest.workflow,
                event.commit_seq,
                event.directory,
                event.committed_at,
                started,
                event.record_type,
                event.encoded_record,
                event.operations,
            )
            try:
                _write_atomic(
                    directory / "record.json",
                    _json_object(envelope, TypeAdapter(RecordEnvelope)),
                )
            except Exception:
                _logger.exception(
                    "Cannot save committed workflow record in %s", directory
                )

    def save_summary(self, summary: EncodedRecord) -> None:
        """Atomically write a non-None done return; output errors propagate."""
        _write_atomic(
            self._paths.metadata_root / "summary.json",
            _json_object(summary, TypeAdapter(EncodedRecord)),
        )
