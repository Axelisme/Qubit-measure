"""Owned entry staging before the first migration manifest exists."""

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Never

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from zcu_tools.datafile import JsonObject, json_values_equal
from zcu_tools.resources.context.library import (
    MODULE_LIBRARY_FORMAT,
    MODULE_LIBRARY_VERSION,
    ModuleLibraryDocument,
)
from zcu_tools.resources.document_store import DocumentStore
from zcu_tools.resources.entry import EntryIdentity, ResultEntry, source_path
from zcu_tools.resources.entry.schema import (
    PARAMETER_FORMAT,
    PARAMETER_VERSION,
    SetupDocument,
)

from .backup import (
    BackupInventory,
    backup_records_path,
    create_backup,
    inventory_paths,
    snapshot_sources,
)
from .errors import MigrationInputError
from .models import MigrationDifference, MigrationManifest, MigrationRequest
from .preservation import preflight_preservation
from .state import overlay_known_fields, write_json

_JSON = TypeAdapter(JsonObject)


@dataclass(frozen=True, kw_only=True)
class InitializationDocument:
    """One owner-created YAML document captured before publishing the entry.

    kind selects the public setup/library schema. value contains the normalized
    typed document plus unknown raw fields, never a hand-built owner layout.
    Its path is the result/Database-relative key in ownership.documents.
    """

    kind: Literal["setup", "library"]
    value: JsonObject


@dataclass(frozen=True, kw_only=True)
class InitializationOwnership:
    """Fixed bootstrap intent and publication baseline, not a transaction journal.

    manifest holds the fixed request identity, declarations, report date/path and
    initial empty conversion state. created_at is the fixed UTC setup time.
    temporary_name names only this tool's sibling staging directories, derived
    from manifest.identity.entry_id. source_inventory fixes complete live hashes
    and directories before the first backup/entry write. prepared becomes true
    only after owner creation has completed and hashes/directories/documents have
    captured that initial entry. Those paths use result/Database relative labels.
    documents holds typed YAML comparisons; other files use hashes (SHA256).
    raw retains unknown wire fields across checkpoints. format/format_version
    identify this tool's zcu.migration-initialization 1.x record.
    """

    manifest: MigrationManifest
    created_at: datetime
    temporary_name: str
    source_inventory: BackupInventory
    prepared: bool = False
    hashes: Mapping[str, str] = field(default_factory=dict)
    directories: tuple[str, ...] = ()
    documents: Mapping[str, InitializationDocument] = field(default_factory=dict)
    format: str = "zcu.migration-initialization"
    format_version: str = "1.0"
    raw: JsonObject = field(default_factory=dict)


_OWNERSHIP = TypeAdapter(InitializationOwnership)


def initialization_path(manifest: MigrationManifest) -> Path:
    """Locate this source's ownership record in its external backup records.

    manifest.report supplies validated source and backup paths. No directory is
    created; a missing backup_path raises MigrationInputError.
    """
    backup = manifest.report.backup_path
    if backup is None:
        raise MigrationInputError("initialization requires a backup path")
    return (
        backup_records_path(backup, manifest.report.source)
        / "migration-initialization.json"
    )


def _checkpoint(
    path: Path, ownership: InitializationOwnership, *, new: bool = False
) -> None:
    known = _JSON.validate_json(
        _OWNERSHIP.dump_json(
            ownership,
            exclude={"raw": True, "manifest": {"raw": True, "report": {"raw": True}}},
        )
    )
    write_json(
        path, overlay_known_fields(ownership.raw, known), replace_existing=not new
    )


def _conflict(
    source: Path, destination: Path, comparison: str, expected: str, actual: str
) -> Never:
    raise MigrationInputError(
        "initialization conflict",
        differences=(
            MigrationDifference(
                source=source,
                destination=destination,
                comparison=comparison,
                expected=expected,
                actual=actual,
            ),
        ),
    )


def _read(path: Path) -> InitializationOwnership:
    if path.is_symlink():
        raise MigrationInputError(f"{path}: ownership symlink conflict")
    try:
        text = path.read_bytes()
        raw = _JSON.validate_json(text, strict=True)
        ownership = _OWNERSHIP.validate_json(text, strict=True)
    except (ValidationError, UnicodeError) as exc:
        raise MigrationInputError(f"{path}: invalid initialization ownership") from exc
    parts = ownership.format_version.split(".")
    if (
        ownership.format != "zcu.migration-initialization"
        or len(parts) != 2
        or parts[0] != "1"
        or not parts[1].isascii()
        or not parts[1].isdecimal()
    ):
        raise MigrationInputError(f"{path}: unsupported initialization header")
    manifest_raw = raw.get("manifest")
    if not isinstance(manifest_raw, dict):
        raise MigrationInputError(f"{path}: missing initialization manifest")
    report_raw = manifest_raw.get("report")
    if not isinstance(report_raw, dict):
        raise MigrationInputError(f"{path}: missing initialization report")
    manifest = replace(
        ownership.manifest,
        raw=manifest_raw,
        report=replace(ownership.manifest.report, raw=report_raw),
    )
    return replace(ownership, manifest=manifest, raw=raw)


def _validate_intent(
    path: Path,
    ownership: InitializationOwnership,
    candidate: MigrationManifest,
    request: MigrationRequest,
) -> None:
    fixed = ownership.manifest
    if (
        fixed.report.entry_id != fixed.identity.entry_id
        or fixed.files
        or fixed.runs
        or fixed.module_files
        or fixed.parameters_complete
        or fixed.data_complete
        or fixed.cleanup_phase != "pending"
        or fixed.format != "zcu.storage-migration"
        or fixed.report.format != "zcu.migration-report"
    ):
        raise MigrationInputError(f"{path}: invalid initial manifest state")
    expected = replace(candidate.identity, entry_id=fixed.identity.entry_id)
    comparisons = (
        ("identity", expected, fixed.identity),
        ("mapping-version", candidate.mapping_version, fixed.mapping_version),
        ("declarations", candidate.report.declarations, fixed.report.declarations),
        ("source", candidate.report.source, fixed.report.source),
        ("destination", candidate.report.destination, fixed.report.destination),
        ("backup", candidate.report.backup_path, fixed.report.backup_path),
    )
    for kind, wanted, actual in comparisons:
        if wanted != actual:
            _conflict(
                candidate.report.source.result_path,
                path,
                kind,
                str(wanted),
                str(actual),
            )
    if (
        request.report_path is not None
        and request.report_path.resolve() != fixed.report_path
    ):
        _conflict(
            candidate.report.source.result_path,
            path,
            "report-path",
            str(fixed.report_path),
            str(request.report_path),
        )
    if ownership.temporary_name != f".migration-init-{fixed.identity.entry_id.hex}":
        raise MigrationInputError(f"{path}: invalid owned temporary name")
    # The owner validates the fixed UUID/time before any path can be removed.
    EntryIdentity(entry_id=fixed.identity.entry_id, created_at=ownership.created_at)
    current = snapshot_sources(candidate.report.source)
    if current != ownership.source_inventory:
        _conflict(
            candidate.report.source.result_path,
            path,
            "source-inventory",
            ownership.source_inventory.summary,
            current.summary,
        )


def _document_value(path: Path, kind: Literal["setup", "library"]) -> JsonObject:
    if kind == "setup":
        document = DocumentStore(
            path,
            SetupDocument,
            format=PARAMETER_FORMAT,
            supported_version=PARAMETER_VERSION,
        ).snapshot()
        known = _JSON.validate_python(document.model_dump(mode="json"), strict=True)
    else:
        library = DocumentStore(
            path,
            ModuleLibraryDocument,
            format=MODULE_LIBRARY_FORMAT,
            supported_version=MODULE_LIBRARY_VERSION,
        ).snapshot()
        known = _JSON.validate_python(library.model_dump(mode="json"), strict=True)
    raw = _JSON.validate_python(
        YAML(typ="safe").load(path.read_text(encoding="utf-8")), strict=True
    )
    return overlay_known_fields(raw, known)


def _validate_prepared(
    ownership: InitializationOwnership, roots: Mapping[str, Path]
) -> None:
    current = inventory_paths(roots)
    if (
        set(current.hashes) != set(ownership.hashes)
        or current.directories != ownership.directories
    ):
        _conflict(
            ownership.manifest.report.source.result_path,
            roots["result"],
            "entry-inventory",
            str((tuple(ownership.hashes), ownership.directories)),
            str((tuple(current.hashes), current.directories)),
        )
    for relative, digest in ownership.hashes.items():
        label, name = relative.split("/", 1)
        path = roots[label] / name
        document = ownership.documents.get(relative)
        if document is not None:
            try:
                actual = _document_value(path, document.kind)
            except (ValueError, YAMLError) as exc:
                _conflict(
                    ownership.manifest.report.source.result_path,
                    path,
                    "typed-yaml",
                    str(document.value),
                    f"invalid document: {exc}",
                )
            if not json_values_equal(document.value, actual):
                _conflict(
                    ownership.manifest.report.source.result_path,
                    path,
                    "typed-yaml",
                    str(document.value),
                    str(actual),
                )
        elif current.hashes[relative] != digest:
            _conflict(
                ownership.manifest.report.source.result_path,
                path,
                "sha256",
                digest,
                current.hashes[relative],
            )


def _validate_staging(ownership: InitializationOwnership) -> None:
    destination = ownership.manifest.report.destination
    source = ownership.manifest.report.source
    if destination.database_path.exists() or destination.database_path.is_symlink():
        raise FileExistsError(destination.database_path)
    root = destination.result_path
    temporary = root.parent / ownership.temporary_name
    if temporary == root or any(
        temporary.is_relative_to(old) or old.is_relative_to(temporary)
        for old in (source.result_path, source.database_path)
    ):
        raise MigrationInputError(f"{temporary}: staging overlaps protected paths")
    if not ownership.prepared and root.exists():
        raise FileExistsError(root)
    for path in (root, temporary):
        if path.is_symlink():
            raise MigrationInputError(f"{path}: initialization symlink conflict")


def _publish_entry(
    path: Path, ownership: InitializationOwnership, request: MigrationRequest
) -> ResultEntry:
    destination = ownership.manifest.report.destination
    staged = {
        "result": destination.result_path.parent / ownership.temporary_name,
    }
    final = {"result": destination.result_path}
    _validate_staging(ownership)
    if not ownership.prepared:
        # Only the exact sibling names fixed before creation belong to this tool.
        for target in staged.values():
            if target.exists():
                inventory_paths({"temporary": target})
                shutil.rmtree(target)
        entry = ResultEntry.create(
            ownership.temporary_name,
            result_root=destination.result_path.parent,
            database_root=destination.database_path.parent,
            identity=EntryIdentity(
                entry_id=ownership.manifest.identity.entry_id,
                created_at=ownership.created_at,
            ),
        )
        setup = source_path(entry.setup)
        library = entry.module_library().source_path
        inventory = inventory_paths(staged)
        document_paths: tuple[tuple[Path, Literal["setup", "library"]], ...] = (
            (setup, "setup"),
            (library, "library"),
        )
        documents = {
            f"result/{file.relative_to(entry.result_path).as_posix()}": InitializationDocument(
                kind=kind, value=_document_value(file, kind)
            )
            for file, kind in document_paths
        }
        ownership = replace(
            ownership,
            prepared=True,
            hashes=inventory.hashes,
            directories=inventory.directories,
            documents=documents,
        )
        _checkpoint(path, ownership)
    _validate_available_prepared(ownership)
    for label, target in final.items():
        if not target.exists():
            staged[label].rename(target)
    return ResultEntry.open(
        request.name,
        result_root=destination.result_path.parent,
        database_root=destination.database_path.parent,
    )


def _validate_available_prepared(ownership: InitializationOwnership) -> None:
    final = ownership.manifest.report.destination.result_path
    staged = final.parent / ownership.temporary_name
    if final.exists() and staged.exists():
        raise MigrationInputError(f"{final}: both staged and final paths exist")
    _validate_prepared(ownership, {"result": final if final.exists() else staged})


def _load_or_plan(
    path: Path, candidate: MigrationManifest, request: MigrationRequest
) -> InitializationOwnership:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise MigrationInputError(f"{ancestor}: initialization symlink conflict")
    if path.exists():
        ownership = _read(path)
        _validate_intent(path, ownership, candidate, request)
        manifest_path = (
            ownership.manifest.report.destination.result_path
            / "records/migration-state.json"
        )
        for target in (manifest_path, ownership.manifest.report_path):
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
        return ownership
    ownership = InitializationOwnership(
        manifest=candidate,
        created_at=datetime.now(timezone.utc),
        temporary_name=f".migration-init-{candidate.identity.entry_id.hex}",
        source_inventory=snapshot_sources(candidate.report.source),
    )
    for target in (
        candidate.report.destination.result_path,
        candidate.report.destination.database_path,
        candidate.report_path,
    ):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    temporary = (
        candidate.report.destination.result_path.parent / ownership.temporary_name
    )
    if temporary.exists():
        raise FileExistsError(temporary)
    return ownership


def initialize_entry(
    request: MigrationRequest, candidate: MigrationManifest
) -> tuple[MigrationManifest, ResultEntry | None]:
    """Verify or resume bootstrap ownership, then publish a complete initial entry.

    request/candidate must have validated, resolved nonoverlapping source, backup,
    destination and report locations. With no ownership, both final destinations
    and the report must be absent. Persist fixed identity/time, declarations,
    report date/path and complete source baseline before the first backup write.
    Existing ownership must match that intent and live sources before any write.
    Complete/verify the backup, create via the owner under the recorded sibling
    temporary name, checkpoint its initial content, then rename the result root.
    Database remains absent until data publication creates its destinations.
    Interrupted unprepared staging is discarded only at those recorded paths.
    Prepared/final content is never rewritten: typed YAML (including unknown raw
    fields) and other file hashes/directories must match before the rename.
    Return the initial manifest and opened final entry; dry_run validates without
    writes and returns no handle. Conflicts raise MigrationInputError or
    FileExistsError; schema/I/O failures propagate. No writer coordination,
    arbitrary-entry adoption, or power-loss transaction guarantee is provided.
    """
    path = initialization_path(candidate)
    ownership = _load_or_plan(path, candidate, request)
    manifest = ownership.manifest
    _validate_staging(ownership)
    preflight_preservation(
        manifest.report.source,
        manifest.report.destination,
        manifest.report.declarations,
    )
    backup_path = manifest.report.backup_path
    if backup_path is None:
        raise MigrationInputError("initialization requires a backup path")
    # A read-only preflight prevents conflicts from leaving a new ownership record.
    create_backup(backup_path, manifest.report.source, dry_run=True)
    if ownership.prepared:
        _validate_available_prepared(ownership)
    if request.dry_run:
        return replace(
            manifest, report=replace(manifest.report, part=request.part)
        ), None
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _checkpoint(path, ownership, new=True)
    backup = create_backup(backup_path, manifest.report.source, dry_run=False)
    manifest = replace(
        manifest,
        backup=backup,
        report=replace(
            manifest.report,
            part=request.part,
            skipped=backup.skipped if backup is not None else (),
        ),
    )
    current = snapshot_sources(manifest.report.source)
    if current != ownership.source_inventory:
        _conflict(
            manifest.report.source.result_path,
            path,
            "source-inventory",
            ownership.source_inventory.summary,
            current.summary,
        )
    entry = _publish_entry(path, ownership, request)
    return manifest, entry
