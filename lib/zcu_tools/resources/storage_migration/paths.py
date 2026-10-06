"""Offline migration paths ownership."""

from pathlib import Path, PureWindowsPath

from .errors import MigrationInputError
from .models import MigrationManifest


def validate_segment(value: str, field: str) -> None:
    """Require a nonempty POSIX/Windows-safe path segment or raise MigrationInputError."""
    if (
        not value
        or value in (".", "..")
        or any(char in value for char in ("/", "\\", "\0"))
        or PureWindowsPath(value).anchor
    ):
        raise MigrationInputError(f"{field}: expected a safe single path segment")


def contained_path(path: Path, root: Path) -> Path:
    """Resolve path under the absolute root or raise MigrationInputError on escape."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise MigrationInputError(f"{path}: escapes {root}")
    return resolved


def source_removed(path: Path, manifest: MigrationManifest) -> bool:
    """Return whether the durable move of this absolute source is source_removed.

    Match the stored lexical identity without touching the original filesystem
    location. Other phases still require the original source checks.
    """
    return any(
        state.source == path
        and state.operation == "move_labber"
        and state.phase == "source_removed"
        for state in manifest.files
    )


def source_identity(path: Path, manifest: MigrationManifest) -> Path:
    """Validate an absolute source under the manifest's old entry roots.

    For a durably removed source, return its lexical identity without resolving
    or reading its now-unowned location. Otherwise resolve containment normally.
    Invalid/escaping paths raise MigrationInputError; filesystem failures propagate.
    """
    roots = (manifest.report.source.result_path, manifest.report.source.database_path)
    if not path.is_absolute() or ".." in path.parts:
        raise MigrationInputError(f"{path}: invalid absolute source identity")
    root = next((root for root in roots if path.is_relative_to(root)), None)
    if root is None:
        raise MigrationInputError(f"{path}: source escapes manifest roots")
    if source_removed(path, manifest):
        return path
    return contained_path(path, root)
