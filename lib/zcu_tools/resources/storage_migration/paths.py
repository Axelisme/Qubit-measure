"""Offline migration paths ownership."""

from pathlib import Path, PureWindowsPath

from .errors import MigrationInputError


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
