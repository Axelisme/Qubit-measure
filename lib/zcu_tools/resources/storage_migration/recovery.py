"""Offline migration recovery ownership."""

import re
from pathlib import Path

from .errors import MigrationInputError
from .models import (
    MigrationFileState,
    MigrationManifest,
    ModuleFilePublication,
)
from .paths import contained_path, source_identity, source_removed, validate_segment
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
    elif state.operation == "native":
        assignment = next(
            (run for run in manifest.runs if run.source == state.source), None
        )
        if (
            assignment is None
            or state.destination
            != destination.database_path / "runs" / assignment.run_id / "data.h5"
        ):
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
        if source_removed(source_path, manifest):
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
    publication: ModuleFilePublication, manifest: MigrationManifest
) -> None:
    source_root = manifest.report.source.result_path
    source = publication.source
    if (
        not source.is_absolute()
        or contained_path(source, source_root) != source
        or source.parent.parent != source_root
        or source.name != "module_cfg.yaml"
    ):
        raise MigrationInputError("manifest: invalid point-local module source")
    validate_segment(source.parent.name, "context label")
    destination_root = manifest.report.destination.result_path
    expected = destination_root / "points" / source.parent.name / "module_cfg.yaml"
    if (
        publication.destination != expected
        or contained_path(publication.destination, destination_root) != expected
    ):
        raise MigrationInputError("manifest: invalid point-local module destination")
    if (
        manifest.source_hashes.get(str(source)) != publication.source_hash
        or re.fullmatch(r"[0-9a-f]{64}", publication.source_hash) is None
        or re.fullmatch(r"[0-9a-f]{64}", publication.destination_hash) is None
    ):
        raise MigrationInputError("manifest: module baseline/hash disagrees")
    if file_hash(publication.destination) != publication.destination_hash:
        raise MigrationInputError(
            f"{publication.destination}: module destination hash changed"
        )


def validate_owned_state(session: MigrationSession) -> None:
    """Check a loaded session's owned files, assignments and immutable source baselines.

    Require contained absolute paths, valid publication phases and matching hashes.
    Raise MigrationInputError on conflict; missing/unreadable files propagate I/O.
    No state or source files change."""
    manifest = session.manifest
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
        _validate_module_publication(publication, manifest)
    _validate_baselines(manifest)
