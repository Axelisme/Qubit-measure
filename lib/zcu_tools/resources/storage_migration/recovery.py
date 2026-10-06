"""Offline migration recovery ownership."""

import re
from pathlib import Path

from .errors import MigrationInputError
from .models import (
    MigrationFileState,
    MigrationManifest,
)
from .paths import contained_path, validate_segment
from .state import MigrationSession, file_hash


def _validate_file_paths(
    state: MigrationFileState, manifest: MigrationManifest
) -> None:
    source, destination = manifest.report.source, manifest.report.destination
    if not state.source.is_absolute() or state.source.resolve() != state.source:
        raise MigrationInputError("manifest: invalid source path")
    source_root = (
        source.result_path
        if state.source.is_relative_to(source.result_path)
        else source.database_path
    )
    contained_path(state.source, source_root)
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
            or contained_path(run.source, manifest.report.source.database_path)
            != run.source
        ):
            raise MigrationInputError("manifest: invalid run source")
        if (
            manifest.source_hashes.get(str(run.source)) != run.source_hash
            or run.assigned_at.utcoffset() is None
        ):
            raise MigrationInputError(
                "manifest: invalid run baseline or assignment time"
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
    _validate_baselines(manifest)
