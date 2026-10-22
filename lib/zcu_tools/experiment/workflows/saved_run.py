"""Read framework commit datasets without interpreting authored native files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath

from pydantic import BaseModel, TypeAdapter

from .artifacts import RecordEnvelope, RunManifest
from .models import EncodedRecord, OperationFacts


@dataclass(frozen=True)
class SavedRecord[P]:
    """One committed row and its exact authored-data directory.

    commit_seq is publication order, not the allocated directory number.
    committed_at is UTC. revisions contains actual reads in deduplicated order.
    record_type is the qualified authored model name. directory is the absolute
    allocated commit directory; record_path is its framework record.json.
    call_seq identifies the commit scope. record is caller-validated P or the
    unvalidated EncodedRecord returned by read_records_raw. operations retains
    the committed scope's actual attempts, including discarded/failed attempts.
    """

    commit_seq: int
    committed_at: datetime
    revisions: tuple[int, ...]
    record_type: str
    directory: Path
    record_path: Path
    call_seq: int
    record: P
    operations: tuple[OperationFacts, ...] = ()


def read_summary(run: Path) -> EncodedRecord | None:
    """Read summary.json from the exact metadata run directory.

    Missing file yields None. Missing/nondirectory run raises
    FileNotFoundError/NotADirectoryError. Malformed JSON/model raises ValueError
    naming summary.json. Other I/O errors propagate. No journal/native file scan.
    """
    directory = run.resolve(strict=True)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    source = directory / "summary.json"
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return None
    try:
        return TypeAdapter(EncodedRecord).validate_json(payload)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error


def read_manifest(run: Path, workflow: str | None = None) -> RunManifest:
    """Read manifest.json, verifying roots, UTC times and optional workflow name.

    run is the exact metadata directory. Missing/I/O errors propagate; malformed
    metadata raises ValueError naming its source. record_schemas is information
    only, not a validator or dynamic-type registry.
    """
    source = run.resolve() / "manifest.json"
    try:
        manifest = TypeAdapter(RunManifest).validate_json(source.read_bytes())
        if workflow is not None and manifest.workflow != workflow:
            raise ValueError(f"Expected workflow {workflow}, got {manifest.workflow}")
        roots = manifest.roots
        if (
            not roots.metadata_root.is_absolute()
            or roots.metadata_root.resolve() != source.parent
        ):
            raise ValueError("metadata_root must identify the supplied run directory")
        if (
            not roots.data_root.is_absolute()
            or roots.data_root == roots.metadata_root
            or roots.data_root in roots.metadata_root.parents
            or roots.metadata_root in roots.data_root.parents
        ):
            raise ValueError("data_root must be absolute, distinct and non-overlapping")
        for value in (manifest.started_at, manifest.updated_at, manifest.ended_at):
            if value is not None and value.utcoffset() != timedelta(0):
                raise ValueError("Manifest times must be timezone-aware UTC")
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    return manifest


def _read_record(manifest: RunManifest, source: Path) -> RecordEnvelope:
    try:
        envelope = TypeAdapter(RecordEnvelope).validate_json(source.read_bytes())
        if (
            envelope.run_id != manifest.identity.run_id
            or envelope.workflow != manifest.workflow
        ):
            raise ValueError("Record identity differs from manifest")
        if (
            envelope.directory
            != source.parent.relative_to(manifest.roots.data_root).as_posix()
        ):
            raise ValueError("Record directory reference differs from its location")
        if (
            envelope.commit_seq <= 0
            or envelope.started.commit_seq != envelope.commit_seq - 1
        ):
            raise ValueError("Invalid commit sequence")
        if envelope.started.call_seq < 0:
            raise ValueError("Invalid commit scope")
        revisions = envelope.started.revisions
        if any(revision < 0 for revision in revisions) or len(set(revisions)) != len(
            revisions
        ):
            raise ValueError("Invalid read-order tunables revisions")
        if envelope.committed_at.utcoffset() != timedelta(0):
            raise ValueError("Commit time must be timezone-aware UTC")
        if envelope.record_type not in manifest.record_schemas:
            raise ValueError("Record type has no manifest schema information")
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    return envelope


def read_records_raw(run: Path) -> tuple[SavedRecord[EncodedRecord], ...]:
    """Read all available committed envelopes, sorted by commit_seq, without model validation.

    Uncommitted allocated directories lack record.json and contribute no row.
    Authored native files are never listed/parsed to create records. Missing run
    or manifest raises; an empty run yields (). Malformed envelopes, duplicate
    commit sequence or wrong identity/reference raise source-named ValueError.
    Journal is not read; lost record files cannot be reconstructed here.
    """
    manifest = read_manifest(run)
    root = manifest.roots.data_root
    if not root.exists():
        return ()
    records: list[SavedRecord[EncodedRecord]] = []
    sequences: set[int] = set()
    for source in sorted(root.rglob("record.json")):
        directory = source.parent
        envelope = _read_record(manifest, source)
        if envelope.commit_seq in sequences:
            raise ValueError(
                f"{source}: duplicate commit sequence {envelope.commit_seq}"
            )
        sequences.add(envelope.commit_seq)
        records.append(
            SavedRecord(
                envelope.commit_seq,
                envelope.committed_at,
                envelope.started.revisions,
                envelope.record_type,
                directory,
                source,
                envelope.started.call_seq,
                envelope.record,
                envelope.operations,
            )
        )
    return tuple(sorted(records, key=lambda record: record.commit_seq))


def read_records[P: BaseModel](
    run: Path, record: type[P] | TypeAdapter[P]
) -> tuple[SavedRecord[P], ...]:
    """Read a homogeneous run as caller-supplied model instances, ordered by commit_seq.

    record is a BaseModel class or its TypeAdapter. Qualified type names must
    match; degraded encoding or field validation raises ValueError naming the
    record.json source. Use read_records_raw for heterogeneous runs. Manifest
    schemas remain informational, and no jsonschema dependency is required.
    """
    adapter = record if isinstance(record, TypeAdapter) else TypeAdapter(record)
    expected = _record_type(adapter)
    rows: list[SavedRecord[P]] = []
    for source in read_records_raw(run):
        try:
            if source.record_type != expected:
                raise ValueError(
                    f"Expected record type {expected}, got {source.record_type}"
                )
            encoded = source.record
            if encoded.mode != "json" or encoded.error is not None:
                raise ValueError("Record does not contain structured JSON")
            value = adapter.validate_python(encoded.value)
        except ValueError as error:
            raise ValueError(f"{source.record_path}: {error}") from error
        rows.append(
            SavedRecord(
                source.commit_seq,
                source.committed_at,
                source.revisions,
                source.record_type,
                source.directory,
                source.record_path,
                source.call_seq,
                value,
                source.operations,
            )
        )
    return tuple(rows)


def _record_type[P: BaseModel](adapter: TypeAdapter[P]) -> str:
    # Pydantic's public model core schema retains the actual class, unlike JSON title overrides.
    schema = adapter.core_schema
    while schema["type"] in ("function-after", "function-before", "function-wrap"):
        inner = schema.get("schema")
        if inner is None:
            break
        schema = inner
    model = schema.get("cls")
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError("read_records requires a single BaseModel type or TypeAdapter")
    return f"{model.__module__}.{model.__qualname__}"


def native_path(directory: Path, filename: str) -> Path:
    """Resolve an authored relative reference beneath its record directory.

    Absolute paths, parent traversal and Windows drives raise ValueError. The
    target need not exist; the caller owns native format parsing and errors.
    """
    relative = Path(filename)
    if (
        not filename
        or relative.is_absolute()
        or PureWindowsPath(filename).drive
        or ".." in relative.parts
        or "\\" in filename
    ):
        raise ValueError(f"Invalid workflow-relative filename: {filename!r}")
    resolved = (directory / relative).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError(
            f"Workflow filename escapes its commit directory: {filename!r}"
        )
    return resolved
