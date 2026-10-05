"""Shared entry path-component validation, independent of filesystem layout."""

from pathlib import Path, PureWindowsPath


def validate_path_segment(name: str, *, field: str) -> None:
    """Reject unsafe POSIX/Windows components with ValueError naming field.

    name must be one nonempty component without dot traversal, separators, NUL
    or a Windows drive/anchor. field identifies the caller argument in errors.
    This check does not access the filesystem or interpret physical semantics.
    """
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        or Path(name).is_absolute()
        or PureWindowsPath(name).anchor
    ):
        raise ValueError(f"{field}={name!r}: expected a single path component")
