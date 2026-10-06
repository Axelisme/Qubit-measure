"""Offline migration runs ownership."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from zcu_tools.datafile import (
    ExperimentPayload,
    RunMetadata,
    RunSnapshot,
    decode_labber_comment,
    load_legacy_labber_payload,
    load_run_data,
    save_run_data,
)
from zcu_tools.resources.entry import (
    ArtifactKey,
    ResultEntry,
    SaveLayout,
    new_run_id,
)

from .errors import MigrationInputError
from .evidence import has_legacy_expression
from .models import (
    KeyMappingItem,
    LegacyRunEvidence,
    MigrationFileState,
    MigrationMapping,
    MigrationRunEvidenceDocument,
    RunAssignment,
)
from .paths import contained_path
from .preservation import preserve_result_files
from .state import MigrationSession, clear_pending, record_pending


def _check_evidence(evidence: LegacyRunEvidence, entry: ResultEntry | None) -> None:
    if has_legacy_expression(evidence.cfg.values):
        raise ValueError("cfg: legacy expression cannot be converted")
    for path, parameter in evidence.snapshot.params.items():
        source = parameter.source
        if source.source != "manual":
            if entry is None:
                raise ValueError(f"snapshot.params.{path}: ledger source is not proven")
            try:
                entry.ledger.get(source.source)
            except KeyError as exc:
                raise ValueError(
                    f"snapshot.params.{path}: unresolved ledger event {source.source!r}"
                ) from exc


def _run_metadata(
    legacy: LegacyRunEvidence, assignment: RunAssignment, entry_id: UUID
) -> RunMetadata:
    snapshot = legacy.snapshot
    return RunMetadata(
        run_id=assignment.run_id,
        experiment=legacy.experiment,
        started_at=legacy.started_at,
        finished_at=legacy.finished_at,
        completion=legacy.completion,
        labber_path=None,
        snapshot=RunSnapshot(
            entry_id=entry_id,
            entry_name=snapshot.entry_name,
            point=snapshot.point,
            description=snapshot.description,
            roles=snapshot.roles,
            params=snapshot.params,
        ),
        provenance=legacy.provenance,
    )


def _load_payload(
    source: Path,
    legacy: LegacyRunEvidence,
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
) -> ExperimentPayload:
    schema = mapping.data_schemas.get(legacy.experiment)
    if schema is None:
        raise ValueError(f"Undeclared experiment tag {legacy.experiment!r}")
    if session.baseline(source) != legacy.source_hash:
        raise ValueError(
            "source_hash: acquisition evidence disagrees with source bytes"
        )
    _check_evidence(legacy, entry)
    payload = load_legacy_labber_payload(source, schema=schema)
    tags = set(payload.metadata.tags) & mapping.data_schemas.keys()
    if tags and tags != {legacy.experiment}:
        raise ValueError(
            f"experiment: evidence {legacy.experiment!r} disagrees with file tags {sorted(tags)!r}"
        )
    comment = decode_labber_comment(payload.metadata.comment)
    if comment.cfg is not None and comment.cfg != legacy.cfg.values:
        raise ValueError("cfg: evidence disagrees with file comment.cfg")
    return payload


def _assignment(
    session: MigrationSession, source: Path, source_hash: str
) -> RunAssignment:
    assignment = next(
        (item for item in session.manifest.runs if item.source == source), None
    )
    if assignment is None:
        at = datetime.now(timezone.utc)
        assignment = RunAssignment(
            source=source,
            source_hash=source_hash,
            run_id=new_run_id(at=at),
            assigned_at=at,
        )
        session.manifest = replace(
            session.manifest, runs=(*session.manifest.runs, assignment)
        )
        session.checkpoint()
    elif assignment.source_hash != source_hash:
        raise MigrationInputError(f"{source}: run assignment hash changed")
    return assignment


def _validate_native(
    session: MigrationSession,
    native: MigrationFileState,
    experiment: str,
    validate_native: Callable[[Path, str], None],
) -> MigrationFileState:
    session.verify_destination(native)
    if native.native_validated:
        return native
    try:
        load_run_data(native.destination)
        validate_native(native.destination, experiment)
        session.verify_destination(native)
        native = replace(native, native_validated=True)
        session.update_file(native)
        session.failure(native.source, "validate_native", None)
        return native
    except Exception as exc:
        # Keep the published native and Labber source for explicit retry.
        session.failure(native.source, "validate_native", exc)
        raise


def _convert_run(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
    source: Path,
    legacy: LegacyRunEvidence | None,
    validate_native: Callable[[Path, str], None],
) -> None:
    if legacy is None:
        record_pending(
            session,
            source,
            "run evidence",
            "Missing explicit historical snapshot/acquisition/completion evidence; Labber source retained",
        )
        return
    native = session.state(source, "native")
    payload = None
    if native is None or native.phase == "planned":
        try:
            payload = _load_payload(source, legacy, session, entry, mapping)
        except MigrationInputError:
            raise
        except ValueError as exc:
            record_pending(session, source, "run evidence", str(exc))
            return
    clear_pending(session, source, "run evidence")
    assignment = _assignment(session, source, legacy.source_hash)
    destination = session.manifest.report.destination
    layout = SaveLayout(
        result_path=destination.result_path,
        database_path=destination.database_path,
        run_id=assignment.run_id,
        point=legacy.snapshot.point,
        saved_at=assignment.assigned_at,
    )
    outputs = layout.outputs(ArtifactKey(section="run", name="data", member="data"))
    native_path = next(item.path for item in outputs if item.format == "data_h5")
    report = session.manifest.report
    mappings = tuple(
        item
        for item in report.key_mappings
        if (item.old_file, item.old_key) != (source, "snapshot.entry_id")
    )
    session.report(
        replace(
            report,
            key_mappings=(
                *mappings,
                KeyMappingItem(
                    old_file=source,
                    old_key="snapshot.entry_id",
                    new_file=native_path,
                    new_path="/meta/snapshot/entry_id",
                    action="value",
                    reason="Assign the new entry UUID; the legacy system did not record this historical UUID.",
                ),
            ),
        )
    )
    metadata = _run_metadata(legacy, assignment, session.manifest.identity.entry_id)

    def build(path: Path) -> None:
        if payload is None:
            raise MigrationInputError(
                f"{source}: missing payload for native publication"
            )
        save_run_data(path, payload, metadata, cfg=legacy.cfg)

    native = session.publish(source, native_path, operation="native", build=build)
    if session.dry_run:
        return
    native = _validate_native(session, native, legacy.experiment, validate_native)
    session.completed(native)
    relative = source.relative_to(session.manifest.report.source.database_path)
    moved = session.publish(
        source,
        contained_path(
            destination.database_path / "Labber" / relative, destination.database_path
        ),
        operation="move_labber",
    )
    session.completed(moved)


def convert_data(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
    evidence_document: MigrationRunEvidenceDocument | None,
    validate_native: Callable[[Path, str], None],
) -> None:
    """Convert evidence-backed Labber sources, then copy/verify/remove only validated originals.

    session owns fixed run identities and publication recovery. entry resolves
    historical ledger references; mapping supplies explicit disk schemas.
    evidence_document is the selected complete historical evidence or None.
    validate_native synchronously validates each published native before its source
    can be removed. Missing/conflicting metadata is pending; execution, callback,
    identity/hash and I/O failures propagate and retain owned recovery state.
    dry_run neither writes nor invokes the callback."""
    preserve_result_files(session)
    source_root = session.manifest.report.source.database_path
    evidence = (
        {}
        if evidence_document is None
        else {item.source: item for item in evidence_document.entries}
    )
    paths = {
        contained_path(path, source_root)
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in (".hdf5", ".h5")
    }
    paths.update(
        state.source
        for state in session.manifest.files
        if state.operation in ("native", "move_labber")
    )
    for source in sorted(paths):
        _convert_run(
            session,
            entry,
            mapping,
            source,
            evidence.get(source.relative_to(source_root)),
            validate_native,
        )
    session.manifest = replace(session.manifest, data_complete=True)
    session.checkpoint()
