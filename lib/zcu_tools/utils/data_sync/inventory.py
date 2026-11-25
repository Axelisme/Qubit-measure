"""List and verify data-tree files through the rclone transport."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from . import rclone
from .settings import DATA_DIRS

_RFC3339 = re.compile(
    r"([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.([0-9]{1,9}))?(Z|[+-][0-9]{2}:[0-9]{2})"
)
_MD5 = re.compile(r"[0-9a-fA-F]{32}")
_FOLDED_DATA_DIRS = frozenset(name.casefold() for name in DATA_DIRS)


class FileStamp(TypedDict):
    """One endpoint's observed metadata, not a content check."""

    mtime: str  # UTC RFC3339, retaining fractional seconds up to nanoseconds.
    size_bytes: int  # Nonnegative byte count.


@dataclass(frozen=True)
class ListedFile:
    """Metadata reported for a file, even if it cannot be verified by content."""

    mtime: str  # UTC RFC3339, retaining up to nine fractional digits.
    size_bytes: int | None  # Nonnegative byte count, or unknown.
    md5: str | None  # Lowercase hex MD5 of the file bytes, or unavailable.

    def stamp(self) -> FileStamp | None:
        """Return the observed time and size, or None if size is unavailable."""
        if self.size_bytes is None:
            return None
        return {"mtime": self.mtime, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class FileInventory:
    """Data files and the directory entries that can affect their paths."""

    files: dict[str, ListedFile]  # Unique prefix-relative path to observed data file.
    duplicates: set[str]  # Paths with multiple objects; none also appear in files.
    directories: set[str]  # Observed directories, including empty ones.
    root_files: set[str]  # Files occupying the names of expected data directories.


@dataclass(frozen=True)
class DestinationPaths:
    """Occupied relative paths in and at the roots of the destination data trees."""

    files: set[str]  # Files, including excluded locks and files occupying tree roots.
    directories: set[str]  # Directories, including empty ones.


def data_filters(*, include_tree_entries: bool = False) -> list[str]:
    """Return filters for data files, optionally also files occupying tree names."""
    return [
        "--filter",
        "- *.lock",
        *_tree_filters(include_tree_entries=include_tree_entries),
    ]


def _tree_filters(*, include_tree_entries: bool) -> list[str]:
    options: list[str] = []
    for directory in DATA_DIRS:
        if include_tree_entries:
            options += ["--filter", f"+ /{directory}"]
        options += ["--filter", f"+ /{directory}/**"]
    return [*options, "--filter", "- **"]


def is_data_path(path: str) -> bool:
    """Check for a prefix-relative slash path in a data tree, excluding locks."""
    return _in_data_tree(path) and "/" in path and not path.endswith(".lock")


def _in_data_tree(path: str, *, case_insensitive: bool = False) -> bool:
    parts = path.split("/")
    tree = parts[0]
    matches_tree = (
        tree.casefold() in _FOLDED_DATA_DIRS if case_insensitive else tree in DATA_DIRS
    )
    return (
        matches_tree
        and all(part not in ("", ".", "..") for part in parts)
        and "\x00" not in path
    )


def file_stamp(mtime: object, size_bytes: object) -> FileStamp:
    """Validate external metadata and normalize the timestamp to UTC without rounding."""
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("Expected a nonnegative integer file size.")
    return {"mtime": _utc_mtime(mtime), "size_bytes": size_bytes}


def _utc_mtime(mtime: object) -> str:
    match = _RFC3339.fullmatch(mtime) if isinstance(mtime, str) else None
    if match is None:
        raise ValueError("Expected an RFC3339 modification time.")
    try:
        seconds = datetime.fromisoformat(match[1] + match[3]).astimezone(UTC)
    except OverflowError as exc:
        raise ValueError(
            f"Modification time is outside the UTC date range: {mtime}"
        ) from exc
    # datetime only carries microseconds; keep the RFC3339 fraction separately.
    fraction = (match[2] or "").rstrip("0")
    utc = seconds.isoformat().removesuffix("+00:00")
    return f"{utc}.{fraction}Z" if fraction else f"{utc}Z"


def mtime_order(utc_mtime: str) -> tuple[datetime, int]:
    """Return a chronological sort key for a normalized UTC inventory timestamp."""
    seconds, _, fraction = utc_mtime.removesuffix("Z").partition(".")
    return datetime.fromisoformat(seconds), int(fraction.ljust(9, "0"))


def list_files(
    binary: Path, config_dir: Path, root: str, *, allow_missing: bool = False
) -> FileInventory:
    """List in-scope files below a local or named-remote prefix root.

    Use the verified binary and config_dir/rclone.conf. Report unknown hashes
    and sizes without dropping files. With allow_missing, an absent directory
    is empty. Raise RuntimeError on listing failure or ValueError on invalid output.
    """
    entries = _list_entries(
        binary,
        config_dir,
        root,
        ["--hash-type", "MD5", *data_filters(include_tree_entries=True)],
        allow_missing=allow_missing,
    )
    return _parse_listing(entries)


def list_case_insensitive_destination(
    binary: Path, config_dir: Path, root: str
) -> DestinationPaths:
    """List occupied paths, including case variants and excluded locks.

    Use the verified binary and config_dir/rclone.conf on a reported
    case-insensitive destination. Read no hashes or modification times.
    Treat an absent root as empty; raise on listing failure or invalid output.
    """
    entries = _list_entries(
        binary,
        config_dir,
        root,
        ["--no-modtime", "--ignore-case", *_tree_filters(include_tree_entries=True)],
        allow_missing=True,
    )
    paths = DestinationPaths(set(), set())
    for entry in entries:
        path, is_dir = _listed_path(entry, case_insensitive=True)
        (paths.directories if is_dir else paths.files).add(path)
    return paths


def _list_entries(
    binary: Path,
    config_dir: Path,
    root: str,
    options: list[str],
    *,
    allow_missing: bool,
) -> list[dict[str, object]]:
    result = rclone.run_cli(
        binary,
        config_dir,
        ["lsjson", root, "--recursive", "--fast-list", "--no-mimetype", *options],
        capture_output=True,
    )
    if result.returncode:
        # lsjson starts the JSON array before listing. Never accept partial data.
        if (
            allow_missing
            and result.returncode == 3
            and result.stdout.strip() == "["
            and _directory_not_found(binary, config_dir, root)
        ):
            return []
        message = f"Could not list {root} (rclone exit {result.returncode})."
        if result.stderr:
            message += f"\n{result.stderr.rstrip()}"
        raise RuntimeError(message)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    entries: object = json.loads(result.stdout)
    if not isinstance(entries, list) or not all(
        isinstance(item, dict) for item in entries
    ):
        raise ValueError("rclone did not return a list of file inventory entries.")
    return entries


def _directory_not_found(binary: Path, config_dir: Path, root: str) -> bool:
    # Exit 3 on a recursive listing can refer to a missing descendant, not root.
    result = rclone.run_cli(
        binary, config_dir, ["lsjson", root, "--stat"], capture_output=True
    )
    return result.returncode == 3


def _parse_listing(entries: list[dict[str, object]]) -> FileInventory:
    files: dict[str, ListedFile] = {}
    directories: set[str] = set()
    root_files: set[str] = set()
    duplicates: set[str] = set()
    for item in entries:
        path, is_dir = _listed_path(item)
        if path in files or path in directories or path in root_files:
            # Drive can contain multiple objects at the same relative path.
            files.pop(path, None)
            duplicates.add(path)
        if is_dir:
            directories.add(path)
        elif path in DATA_DIRS:
            root_files.add(path)
        elif path not in duplicates:
            files[path] = _listed_file(item)
    return FileInventory(files, duplicates, directories, root_files)


def _listed_path(
    item: dict[str, object], *, case_insensitive: bool = False
) -> tuple[str, bool]:
    if (
        not isinstance(path := item.get("Path"), str)
        or not isinstance(is_dir := item.get("IsDir"), bool)
        or not _in_data_tree(path, case_insensitive=case_insensitive)
        or (not case_insensitive and not is_dir and path.endswith(".lock"))
    ):
        raise ValueError(f"rclone returned an invalid data path or type: {item!r}.")
    return path, is_dir


def _listed_file(item: dict[str, object]) -> ListedFile:
    hashes = item.get("Hashes")
    md5: object = hashes.get("md5") if isinstance(hashes, dict) else None
    size = item.get("Size")
    if type(size) is not int or size < -1:
        raise ValueError(f"Invalid file size for {item.get('Path')!r}: {size!r}.")
    return ListedFile(
        _utc_mtime(item.get("ModTime")),
        None if size == -1 else size,
        md5.lower() if isinstance(md5, str) and _MD5.fullmatch(md5) else None,
    )
