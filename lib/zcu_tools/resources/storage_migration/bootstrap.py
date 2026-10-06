"""Offline migration bootstrap ownership."""

import re
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from zcu_tools.datafile import JsonObject
from zcu_tools.resources.entry import (
    ResultEntry,
)

from .errors import MigrationInputError
from .models import (
    MigrationDestination,
    MigrationIdentity,
    MigrationManifest,
    MigrationMapping,
    MigrationReport,
    MigrationRequest,
    MigrationSource,
)
from .paths import contained_path, validate_segment
from .recovery import validate_owned_state
from .state import MigrationSession, read_manifest, report_json, write_json

_JSON = TypeAdapter(JsonObject)


def validate_mapping(mapping: MigrationMapping) -> None:
    """Require a major.minor mapping revision, unique keys and explicit action targets.

    Raise MigrationInputError for invalid declarations. No registration or I/O occurs."""
    if re.fullmatch(r"[0-9]+\.[0-9]+", mapping.mapping_version) is None:
        raise MigrationInputError("mapping_version: expected major.minor")
    seen: set[str] = set()
    for rule in mapping.rules:
        if not rule.old_key or rule.old_key in seen:
            raise MigrationInputError(f"Duplicate/empty mapping key {rule.old_key!r}")
        seen.add(rule.old_key)
        if rule.action not in ("value", "stderr", "module", "remove", "pending"):
            raise MigrationInputError(f"{rule.old_key}: invalid action")
        if rule.action in ("value", "stderr", "module") and not rule.target_path:
            raise MigrationInputError(f"{rule.old_key}: target_path is required")


def _locations(
    request: MigrationRequest,
) -> tuple[MigrationSource, MigrationDestination]:
    for field in ("source_chip", "source_qubit", "name"):
        validate_segment(getattr(request, field), field)
    if request.part not in ("parameters", "data", "all"):
        raise MigrationInputError(f"Unknown part {request.part!r}")
    result_root, database_root, results_root = (
        request.result_root.resolve(),
        request.database_root.resolve(),
        request.results_root.resolve(),
    )
    source = MigrationSource(
        result_path=contained_path(
            result_root / request.source_chip / request.source_qubit, result_root
        ),
        database_path=contained_path(
            database_root / request.source_chip / request.source_qubit, database_root
        ),
    )
    destination = MigrationDestination(
        result_path=contained_path(results_root / request.name, results_root),
        database_path=contained_path(database_root / request.name, database_root),
    )
    for path in (source.result_path, source.database_path):
        if not path.is_dir():
            raise MigrationInputError(f"{path}: source directory does not exist")
    for target in (destination.result_path, destination.database_path):
        for old in (source.result_path, source.database_path):
            if target.is_relative_to(old) or old.is_relative_to(target):
                raise MigrationInputError(
                    f"{target}: source/destination overlap with {old}"
                )
    if destination.result_path.is_relative_to(
        destination.database_path
    ) or destination.database_path.is_relative_to(destination.result_path):
        raise MigrationInputError("Result and Database destinations overlap")
    return source, destination


def _identity(request: MigrationRequest, entry_id: UUID) -> MigrationIdentity:
    return MigrationIdentity(
        result_root=request.result_root.resolve(),
        database_root=request.database_root.resolve(),
        results_root=request.results_root.resolve(),
        source_chip=request.source_chip,
        source_qubit=request.source_qubit,
        name=request.name,
        entry_id=entry_id,
    )


def _resume_session(
    request: MigrationRequest,
    mapping: MigrationMapping,
    path: Path,
    source: MigrationSource,
    destination: MigrationDestination,
) -> tuple[MigrationManifest, ResultEntry]:
    if not path.is_file():
        raise MigrationInputError(f"{path}: resume requires this tool's manifest")
    manifest = read_manifest(path)
    _validate_report_location(manifest.report_path, source, destination)
    manifest = _recover_first_report(manifest)
    expected = _identity(request, manifest.identity.entry_id)
    if (
        manifest.identity != expected
        or manifest.mapping_version != mapping.mapping_version
    ):
        raise MigrationInputError(
            f"{path}: immutable request or mapping_version changed"
        )
    if (
        manifest.report.source != source
        or manifest.report.destination != destination
        or manifest.report.entry_id != expected.entry_id
    ):
        raise MigrationInputError(f"{path}: report identity disagrees")
    if (
        request.report_path is not None
        and request.report_path.resolve() != manifest.report_path
    ):
        raise MigrationInputError(f"{path}: report_path is fixed on resume")
    entry = ResultEntry.open(
        request.name,
        result_root=expected.results_root,
        database_root=expected.database_root,
    )
    if UUID(entry.entry_id) != manifest.identity.entry_id:
        raise MigrationInputError(f"{path}: entry_id changed")
    for label in entry.list_points():
        entry.use_point(label)
    union = request.part if manifest.report.part == request.part else "all"
    return replace(manifest, report=replace(manifest.report, part=union)), entry


def _validate_report_location(
    report_path: Path,
    source: MigrationSource,
    destination: MigrationDestination,
) -> None:
    default = destination.result_path / "records" / "migration-report.json"
    if not report_path.is_absolute() or report_path.resolve() != report_path:
        raise MigrationInputError(
            f"{report_path}: report path must be resolved and absolute"
        )
    if report_path != default and any(
        report_path.is_relative_to(root)
        for root in (
            source.result_path,
            source.database_path,
            destination.result_path,
            destination.database_path,
        )
    ):
        raise MigrationInputError(
            f"{report_path}: report must be independent of entry/source files"
        )


def _new_session(
    request: MigrationRequest,
    mapping: MigrationMapping,
    source: MigrationSource,
    destination: MigrationDestination,
) -> tuple[MigrationManifest, ResultEntry | None]:
    for target in (destination.result_path, destination.database_path):
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    report_path = (
        request.report_path.resolve()
        if request.report_path is not None
        else destination.result_path / "records" / "migration-report.json"
    )
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(report_path)
    if request.report_path is not None:
        if any(
            report_path.is_relative_to(root)
            for root in (
                source.result_path,
                source.database_path,
                destination.result_path,
                destination.database_path,
            )
        ):
            raise MigrationInputError(
                f"{report_path}: --report must be an independent file"
            )
        if not report_path.parent.is_dir():
            raise MigrationInputError(f"{report_path}: report parent must exist")
    entry = None
    entry_id = uuid4()
    if not request.dry_run:
        entry = ResultEntry.create(
            request.name,
            result_root=request.results_root,
            database_root=request.database_root,
        )
        entry_id = UUID(entry.entry_id)
    report = MigrationReport(
        source=source, destination=destination, entry_id=entry_id, part=request.part
    )
    manifest = MigrationManifest(
        identity=_identity(request, entry_id),
        mapping_version=mapping.mapping_version,
        report_path=report_path,
        source_hashes={},
        evidence=None,
        files=(),
        runs=(),
        parameters_complete=False,
        data_complete=False,
        report=report,
    )
    return manifest, entry


def publish_report(session: MigrationSession) -> None:
    """Publish the session's accumulated JSON report at its immutable owned path.

    First publication is exclusive; later updates replace only the recorded report.
    Checkpoint first-publication ownership. dry_run writes nothing. I/O/collisions
    propagate; the durable manifest can rebuild an interrupted report update."""
    if session.dry_run:
        return
    write_json(
        session.manifest.report_path,
        report_json(session.manifest.report),
        replace_existing=session.manifest.report_published,
    )
    if not session.manifest.report_published:
        session.manifest = replace(session.manifest, report_published=True)
        session.checkpoint()


def _recover_first_report(manifest: MigrationManifest) -> MigrationManifest:
    if not manifest.report_published and manifest.report_path.exists():
        existing = _JSON.validate_json(
            manifest.report_path.read_text(encoding="utf-8"), strict=True
        )
        if existing != report_json(manifest.report):
            raise MigrationInputError(
                f"{manifest.report_path}: unowned report collision"
            )
        return replace(manifest, report_published=True)
    return manifest


def prepare_session(
    request: MigrationRequest, mapping: MigrationMapping
) -> tuple[MigrationSession, ResultEntry | None]:
    """Resolve and validate request paths, then create or resume this tool's entry.

    mapping supplies the exact revision required on resume. Return a session and
    entry handle, or None for a new dry-run entry. Refuse collisions, overlapping
    trees, invalid current containers or conflicting manifest/hash/identity with
    MigrationInputError/FileExistsError. Checkpoint initial ownership before source
    operations. No legacy source is modified; I/O/schema failures propagate."""
    source, destination = _locations(request)
    path = destination.result_path / "records" / "migration-state.json"
    if request.resume:
        manifest, entry = _resume_session(request, mapping, path, source, destination)
    else:
        manifest, entry = _new_session(request, mapping, source, destination)
    session = MigrationSession(path, manifest, dry_run=request.dry_run)
    validate_owned_state(session)
    session.checkpoint()
    publish_report(session)
    return session, entry
