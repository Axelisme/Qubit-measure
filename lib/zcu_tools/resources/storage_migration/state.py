"""Manifest/report encoding and owned, resumable file publication."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from zcu_tools.datafile import JsonObject

from .errors import MigrationInputError
from .models import (
    FailureItem,
    FileMigrationItem,
    MigrationFileState,
    MigrationManifest,
    MigrationReport,
    PendingItem,
)

_JSON = TypeAdapter(JsonObject)
_MANIFEST = TypeAdapter(MigrationManifest)
_REPORT = TypeAdapter(MigrationReport)


def _merge_raw(raw: JsonValue, known: JsonValue) -> JsonValue:
    if isinstance(raw, dict) and isinstance(known, dict):
        merged: JsonObject = dict(raw)
        for key, value in known.items():
            merged[key] = _merge_raw(raw.get(key), value)
        return merged
    if isinstance(raw, list) and isinstance(known, list):
        merged_list: list[JsonValue] = []
        for index, value in enumerate(known):
            old: JsonValue = raw[index] if index < len(raw) else None
            if isinstance(value, dict):
                # List edits must not transfer future fields to a different source.
                identity = tuple(
                    key
                    for key in (
                        "source",
                        "destination",
                        "operation",
                        "location",
                        "old_file",
                        "old_key",
                    )
                    if key in value
                )
                if identity:
                    old = next(
                        (
                            item
                            for item in raw
                            if isinstance(item, dict)
                            and all(item.get(key) == value[key] for key in identity)
                        ),
                        None,
                    )
            merged_list.append(_merge_raw(old, value))
        return merged_list
    return known


def report_json(report: MigrationReport) -> JsonObject:
    """Encode known report fields over its raw future fields without writing files."""
    known = _JSON.validate_json(_REPORT.dump_json(report, exclude={"raw"}))
    return _JSON.validate_python(_merge_raw(report.raw, known), strict=True)


def _manifest_json(manifest: MigrationManifest) -> JsonObject:
    known = _JSON.validate_json(
        _MANIFEST.dump_json(manifest, exclude={"raw", "report"})
    )
    known["report"] = report_json(manifest.report)
    return _JSON.validate_python(_merge_raw(manifest.raw, known), strict=True)


def _check_header(raw: JsonObject, expected: str, path: Path) -> None:
    version = raw.get("format_version")
    if raw.get("format") != expected or not isinstance(version, str):
        raise MigrationInputError(f"{path}: invalid {expected} header")
    parts = version.split(".")
    if (
        len(parts) != 2
        or any(not part.isascii() or not part.isdecimal() for part in parts)
        or int(parts[0]) != 1
    ):
        raise MigrationInputError(f"{path}: unsupported {expected} version {version!r}")


def read_manifest(path: Path) -> MigrationManifest:
    """Read exact UTF-8 1.x manifest/report models and retain both raw JSON trees.

    Invalid headers/types raise located MigrationInputError; I/O errors propagate.
    Identity, path ownership and hash checks belong to the converter before writes.
    """
    try:
        text = path.read_text(encoding="utf-8")
        raw = _JSON.validate_json(text, strict=True)
        _check_header(raw, "zcu.storage-migration", path)
        report_raw = raw.get("report")
        if not isinstance(report_raw, dict):
            raise MigrationInputError(f"{path}: missing report")
        _check_header(report_raw, "zcu.migration-report", path)
        manifest = _MANIFEST.validate_json(text, strict=True)
    except (UnicodeError, ValidationError) as exc:
        raise MigrationInputError(f"{path}: {exc}") from exc
    return replace(manifest, raw=raw, report=replace(manifest.report, raw=report_raw))


def write_json(
    path: Path, document: JsonObject, *, replace_existing: bool = True
) -> None:
    """Replace an already authorized exact JSON path using same-directory temp.

    Caller owns this file and provides an existing parent. replace_existing=False
    publishes exclusively, raising FileExistsError on a collision. Encode UTF-8 finite
    JSON; encoding/I/O failures propagate and clean this call\'s temporary file.
    No cross-file transaction or power-loss guarantee is provided.
    """
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        if replace_existing:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    """Read exact file bytes and return lowercase SHA256; propagate I/O failures."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class MigrationSession:
    """Keep manifest checkpoints and publication recovery in one offline owner.

    The converter uses this session for copies, native publications and moves.
    dry_run suppresses every write; it still checks source/destination hashes.
    No lock or power-loss/cross-file atomicity is promised.
    """

    def __init__(
        self, path: Path, manifest: MigrationManifest, *, dry_run: bool
    ) -> None:
        """Bind the validated manifest path/state; dry_run keeps changes in memory.

        path is the owned exact migration-state.json path. manifest holds the
        immutable identity, accumulated report and publication recovery state.
        No files are read or written during construction.
        """
        self.path = path
        self.manifest = manifest
        self.dry_run = dry_run

    def checkpoint(self) -> None:
        """Persist current manifest atomically, or do nothing in dry_run; I/O propagates."""
        if not self.dry_run:
            write_json(self.path, _manifest_json(self.manifest))

    def report(self, report: MigrationReport) -> None:
        """Replace the accumulated report and checkpoint; no separate report write."""
        self.manifest = replace(self.manifest, report=report)
        self.checkpoint()

    def baseline(self, source: Path) -> str:
        """Hash an absolute source and checkpoint its first baseline.

        A later mismatch raises MigrationInputError without changing baseline.
        Missing/unreadable sources propagate filesystem errors.
        """
        current = file_hash(source)
        recorded = self.manifest.source_hashes.get(str(source))
        if recorded is not None and recorded != current:
            raise MigrationInputError(f"{source}: source hash changed")
        if recorded is None:
            hashes = dict(self.manifest.source_hashes)
            hashes[str(source)] = current
            self.manifest = replace(self.manifest, source_hashes=hashes)
            self.checkpoint()
        return current

    def state(self, source: Path, operation: str) -> MigrationFileState | None:
        """Find absolute source plus copy/move_labber/native operation, or return None."""
        return next(
            (
                item
                for item in self.manifest.files
                if item.source == source and item.operation == operation
            ),
            None,
        )

    def update_file(self, state: MigrationFileState) -> None:
        """Upsert the source/operation recovery state and checkpoint; I/O propagates."""
        files = tuple(
            item
            for item in self.manifest.files
            if (item.source, item.operation) != (state.source, state.operation)
        )
        self.manifest = replace(self.manifest, files=(*files, state))
        self.checkpoint()

    def verify_destination(self, state: MigrationFileState) -> None:
        """Require prepared hash to match actual destination bytes or raise conflict.

        Missing or unreadable destination propagates filesystem errors.
        """
        if (
            state.destination_hash is None
            or file_hash(state.destination) != state.destination_hash
        ):
            raise MigrationInputError(f"{state.destination}: destination hash changed")

    def failure(self, source: Path, operation: str, error: Exception | None) -> None:
        """Replace this absolute source/action diagnostic, or remove it with None.

        Checkpoint the cumulative report without clearing other action failures.
        """
        report = self.manifest.report
        failures = tuple(
            item
            for item in report.failures
            if (item.source, item.operation) != (source, operation)
        )
        if error is not None:
            failures = (
                *failures,
                FailureItem(source=source, operation=operation, error=str(error)),
            )
        self.report(replace(report, failures=failures))

    def _plan(
        self, source: Path, destination: Path, operation: str
    ) -> MigrationFileState:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        state = MigrationFileState(
            operation="native"
            if operation == "native"
            else "move_labber"
            if operation == "move_labber"
            else "copy",
            source=source,
            destination=destination,
            temp_path=destination.with_name(
                f".{destination.name}.migration-{uuid4().hex}.tmp"
            ),
            source_hash=self.baseline(source),
            destination_hash=None,
            phase="planned",
            native_validated=False if operation == "native" else None,
        )
        self.update_file(state)
        return state

    def _prepare(
        self, state: MigrationFileState, build: Callable[[Path], None] | None
    ) -> MigrationFileState:
        if state.destination.exists() or state.destination.is_symlink():
            raise FileExistsError(state.destination)
        if self.baseline(state.source) != state.source_hash:
            raise MigrationInputError(f"{state.source}: source hash changed")
        state.destination.parent.mkdir(parents=True, exist_ok=True)
        state.temp_path.unlink(missing_ok=True)
        if build is None:
            shutil.copyfile(state.source, state.temp_path)
        else:
            build(state.temp_path)
        destination_hash = file_hash(state.temp_path)
        if self.baseline(state.source) != state.source_hash or (
            state.operation != "native" and destination_hash != state.source_hash
        ):
            raise MigrationInputError(
                f"{state.source}: source/copy hash changed during publication"
            )
        state = replace(state, phase="prepared", destination_hash=destination_hash)
        self.update_file(state)
        return state

    def _publish_prepared(self, state: MigrationFileState) -> MigrationFileState:
        if state.destination.exists() or state.destination.is_symlink():
            self.verify_destination(state)
        else:
            if file_hash(state.temp_path) != state.destination_hash:
                raise MigrationInputError(f"{state.temp_path}: prepared hash changed")
            os.link(state.temp_path, state.destination)
        state.temp_path.unlink(missing_ok=True)
        state = replace(state, phase="published")
        self.update_file(state)
        return state

    def publish(
        self,
        source: Path,
        destination: Path,
        *,
        operation: str,
        build: Callable[[Path], None] | None = None,
    ) -> MigrationFileState:
        """Recover or publish one owned exact destination and return durable state.

        operation is copy/move_labber/native. build writes native temp bytes;
        None copies source bytes. Verify hashes at every publication/removal
        window. A move only removes source after published destination verification.
        Conflicts raise MigrationInputError/FileExistsError; execution failures
        persist diagnostic/recovery state and propagate. dry_run only plans.
        """
        if operation not in ("copy", "move_labber", "native"):
            raise ValueError(f"Invalid publication operation {operation!r}")
        state = self.state(source, operation)
        if state is None:
            state = self._plan(source, destination, operation)
        elif state.destination != destination:
            raise MigrationInputError(f"{source}: publication destination changed")
        if self.dry_run:
            return state
        try:
            if state.phase == "planned":
                state = self._prepare(state, build)
            if state.phase == "prepared":
                state = self._publish_prepared(state)
            self.verify_destination(state)
            if operation == "move_labber" and state.phase == "published":
                if source.exists():
                    if self.baseline(source) != state.source_hash:
                        raise MigrationInputError(f"{source}: source hash changed")
                    source.unlink()
                state = replace(state, phase="source_removed")
                self.update_file(state)
            self.failure(source, operation, None)
            return state
        except Exception as exc:
            # Persist diagnostic and ownership so a later explicit resume can recover.
            self.failure(source, operation, exc)
            raise

    def completed(self, state: MigrationFileState) -> None:
        """Record completed copy, validated native or removed Labber and checkpoint.

        dry_run and incomplete/unvalidated states add no success report item.
        """
        if self.dry_run or state.destination_hash is None:
            return
        if state.operation == "native" and not state.native_validated:
            return
        report = self.manifest.report
        status = (
            "converted"
            if state.operation == "native"
            else "moved"
            if state.operation == "move_labber"
            else "preserved"
        )
        item = FileMigrationItem(
            source=state.source,
            destination=state.destination,
            source_hash=state.source_hash,
            destination_hash=state.destination_hash,
            status=status,
        )
        if status == "converted":
            items = tuple(
                old for old in report.converted_files if old.source != state.source
            )
            report = replace(report, converted_files=(*items, item))
        elif status == "moved":
            if state.phase != "source_removed":
                return
            items = tuple(
                old for old in report.moved_labber_files if old.source != state.source
            )
            report = replace(report, moved_labber_files=(*items, item))
        else:
            items = tuple(
                old for old in report.preserved_files if old.source != state.source
            )
            report = replace(report, preserved_files=(*items, item))
        self.report(report)


def record_pending(
    session: MigrationSession, source: Path, location: str, reason: str
) -> None:
    """Upsert the absolute source/location pending reason and checkpoint the cumulative report."""
    report = session.manifest.report
    items = tuple(
        item
        for item in report.pending
        if (item.source, item.location) != (source, location)
    )
    session.report(
        replace(
            report,
            pending=(
                *items,
                PendingItem(
                    source=source,
                    location=location,
                    reason=reason,
                    suggested_action="Review the source and supply explicit mapping or acquisition evidence.",
                ),
            ),
        )
    )


def clear_pending(session: MigrationSession, source: Path, location: str) -> None:
    """Remove only the given source/location pending item and checkpoint; I/O propagates."""
    report = session.manifest.report
    session.report(
        replace(
            report,
            pending=tuple(
                item
                for item in report.pending
                if (item.source, item.location) != (source, location)
            ),
        )
    )
