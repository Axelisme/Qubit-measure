"""Offline migration recovery ownership."""

import json
import re
from pathlib import Path

from zcu_tools.resources.entry import ArtifactKey, ResultEntry, SaveLayout

from .documents import difference, validate_document_baselines
from .errors import MigrationInputError
from .evidence import parse_run_evidence
from .models import (
    MigrationFileState,
    MigrationManifest,
    ModuleFilePublication,
)
from .paths import contained_path, source_identity, source_removed, validate_segment
from .processed_spectrums import LEGACY_PATH, SPECTRUM_PATH
from .state import MigrationSession, file_hash


def _validate_file_paths(
    state: MigrationFileState, manifest: MigrationManifest
) -> None:
    source, destination = manifest.report.source, manifest.report.destination
    if source_identity(state.source, manifest) != state.source:
        raise MigrationInputError("manifest: invalid source path")
    destination_root = (
        destination.result_path
        if state.destination.is_relative_to(destination.result_path)
        else destination.database_path
    )
    if (
        not state.destination.is_absolute()
        or contained_path(state.destination, destination_root) != state.destination
    ):
        raise MigrationInputError("manifest: invalid destination path")
    if (
        state.temp_path.parent != state.destination.parent
        or re.fullmatch(
            re.escape(f".{state.destination.name}.migration-") + r"[0-9a-f]{32}\.tmp",
            state.temp_path.name,
        )
        is None
    ):
        raise MigrationInputError("manifest: invalid owned temporary path")
    contained_path(state.temp_path, destination_root)
    if state.operation == "move_labber":
        expected = (
            destination.database_path
            / "Labber"
            / state.source.relative_to(source.database_path)
        )
        if state.destination != expected:
            raise MigrationInputError("manifest: invalid Labber move destination")
    elif state.operation == "samples":
        if (
            state.source != source.result_path / "samples.csv"
            or state.destination.parent
            != destination.result_path / "records" / "components"
            or state.destination.suffix != ".jsonl"
        ):
            raise MigrationInputError("manifest: invalid samples publication paths")
        validate_segment(state.destination.stem, "sample component")
    elif state.operation in ("spectrums", "spectrums_legacy"):
        relative = SPECTRUM_PATH if state.operation == "spectrums" else LEGACY_PATH
        if (
            state.source != source.result_path / SPECTRUM_PATH
            or state.destination != destination.result_path / relative
        ):
            raise MigrationInputError(
                "manifest: invalid processed spectrum publication paths"
            )
    elif state.operation == "native":
        _validate_native_destination(state, manifest)


def _validate_native_destination(
    state: MigrationFileState, manifest: MigrationManifest
) -> None:
    assignment = next(
        (run for run in manifest.runs if run.source == state.source), None
    )
    if assignment is None or manifest.evidence is None:
        raise MigrationInputError(
            "manifest: native destination has no matching run assignment"
        )
    evidence = parse_run_evidence(json.dumps(manifest.evidence), source=state.source)
    legacy = next(
        (
            item
            for item in evidence.entries
            if manifest.report.source.database_path / item.source == state.source
        ),
        None,
    )
    if legacy is None:
        raise MigrationInputError("manifest: native destination has no run evidence")
    destination = manifest.report.destination
    layout = SaveLayout(
        result_path=destination.result_path,
        database_path=destination.database_path,
        run_id=assignment.run_id,
        experiment=legacy.experiment,
        point=legacy.snapshot.point,
        saved_at=assignment.assigned_at,
    )
    outputs = layout.outputs(ArtifactKey(section="run", name="data", member="data"))
    expected = next(item.path for item in outputs if item.format == "data_h5")
    if state.destination != expected:
        raise MigrationInputError(
            "manifest: native destination has no matching run assignment"
        )


def _validate_file_phase(
    state: MigrationFileState, manifest: MigrationManifest
) -> None:
    if manifest.source_hashes.get(str(state.source)) != state.source_hash:
        raise MigrationInputError("manifest: file baseline disagrees")
    if re.fullmatch(r"[0-9a-f]{64}", state.source_hash) is None:
        raise MigrationInputError("manifest: invalid source hash")
    if state.phase != "planned" and (
        state.destination_hash is None
        or re.fullmatch(r"[0-9a-f]{64}", state.destination_hash) is None
    ):
        raise MigrationInputError("manifest: missing prepared hash")
    if state.phase == "planned" and state.destination_hash is not None:
        raise MigrationInputError("manifest: planned state already has prepared hash")
    if state.native_validated and state.phase != "published":
        raise MigrationInputError("manifest: unpublished native cannot be validated")
    if state.phase == "source_removed" and state.operation != "move_labber":
        raise MigrationInputError("manifest: invalid source_removed phase")
    if state.operation == "native":
        if not isinstance(state.native_validated, bool):
            raise MigrationInputError("manifest: missing native validation state")
    elif state.native_validated is not None:
        raise MigrationInputError("manifest: invalid native validation state")


def _validate_baselines(manifest: MigrationManifest) -> None:
    source = manifest.report.source
    for path, expected in manifest.source_hashes.items():
        source_path = Path(path)
        if not source_path.is_absolute() or not any(
            source_path.is_relative_to(root)
            for root in (source.result_path, source.database_path)
        ):
            raise MigrationInputError("manifest: invalid baseline source")
        if source_removed(source_path, manifest) or manifest.cleanup_phase != "pending":
            continue
        if source_path.exists():
            if file_hash(source_path) != expected:
                raise MigrationInputError(f"{source_path}: source hash changed")
        elif not any(
            state.source == source_path
            and state.operation == "move_labber"
            and state.phase in ("published", "source_removed")
            for state in manifest.files
        ):
            raise MigrationInputError(f"{source_path}: baseline source disappeared")


def _validate_assignments(manifest: MigrationManifest) -> None:
    sources: set[Path] = set()
    ids: set[str] = set()
    for run in manifest.runs:
        validate_segment(run.run_id, "run_id")
        if run.source in sources or run.run_id in ids:
            raise MigrationInputError("manifest: duplicate run assignment")
        sources.add(run.source)
        ids.add(run.run_id)
        if (
            not run.source.is_absolute()
            or not run.source.is_relative_to(manifest.report.source.database_path)
            or source_identity(run.source, manifest) != run.source
        ):
            raise MigrationInputError("manifest: invalid run source")
        if (
            manifest.source_hashes.get(str(run.source)) != run.source_hash
            or run.assigned_at.utcoffset() is None
        ):
            raise MigrationInputError(
                "manifest: invalid run baseline or assignment time"
            )


def _validate_module_publication(
    publication: ModuleFilePublication, manifest: MigrationManifest, entry: ResultEntry
) -> None:
    source_root = manifest.report.source.result_path
    source = publication.source
    if (
        not source.is_absolute()
        or contained_path(source, source_root) != source
        or not (
            source.parent == source_root
            or source.parent.parent == source_root
            or source.parent.parent == source_root / "exps"
        )
        or source.name != "module_cfg.yaml"
    ):
        raise MigrationInputError("manifest: invalid point-local module source")
    validate_segment(source.parent.name, "context label")
    destination_root = manifest.report.destination.result_path
    if (
        not publication.destination.is_absolute()
        or contained_path(publication.destination, destination_root)
        != publication.destination
    ):
        raise MigrationInputError("manifest: invalid point-local module destination")
    if (
        manifest.source_hashes.get(str(source)) != publication.source_hash
        or re.fullmatch(r"[0-9a-f]{64}", publication.source_hash) is None
        or re.fullmatch(r"[0-9a-f]{64}", publication.destination_hash) is None
    ):
        raise MigrationInputError("manifest: module baseline/hash disagrees")
    # Typed document comparison runs before opening a potentially cleaning library.
    if str(publication.destination) not in (manifest.documents or {}):
        raise difference(
            source,
            publication.destination,
            "document-baseline",
            "complete comparison evidence",
            "缺比較依據",
        )
    expected = entry.module_library(
        None if source.parent == source_root else source.parent.name
    ).source_path
    if publication.destination != expected:
        raise MigrationInputError("manifest: invalid point-local module destination")


def validate_owned_state(
    session: MigrationSession, *, entry: ResultEntry | None
) -> None:
    """Check a loaded session's owned files, assignments and immutable source baselines.

    Require contained absolute paths, valid publication phases and matching hashes.
    entry is bootstrap's matching opened entry, or None for a new dry-run with
    no destination handle. Recorded module publications require an entry and use
    its public library source_path; destination layout is not owned here.
    Raise MigrationInputError on conflict or an absent required entry;
    missing/unreadable files and library schema errors propagate. No files change."""
    manifest = session.manifest
    if entry is not None:
        validate_document_baselines(
            session, labels=entry.list_points(), report_existing=session.path.is_file()
        )
    _validate_assignments(manifest)
    seen: set[tuple[Path, str]] = set()
    for state in manifest.files:
        identity = (state.source, state.operation)
        if identity in seen:
            raise MigrationInputError(f"{session.path}: duplicate file state")
        seen.add(identity)
        _validate_file_paths(state, manifest)
        _validate_file_phase(state, manifest)
        if state.phase in ("published", "source_removed") or (
            state.phase == "prepared" and state.destination.exists()
        ):
            session.verify_destination(state)
    module_sources: set[Path] = set()
    module_destinations: set[Path] = set()
    for publication in manifest.module_files:
        if (
            publication.source in module_sources
            or publication.destination in module_destinations
        ):
            raise MigrationInputError("manifest: duplicate module publication")
        module_sources.add(publication.source)
        module_destinations.add(publication.destination)
        if entry is None:
            raise MigrationInputError("manifest: module publication requires an entry")
        _validate_module_publication(publication, manifest, entry)
    _validate_baselines(manifest)
