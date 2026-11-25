"""List and verify data-tree files through the rclone transport."""

from __future__ import annotations

import json
import re
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


class FileStamp(TypedDict):
    """One endpoint's observed metadata, not a content check."""

    mtime: str  # UTC RFC3339, retaining fractional seconds up to nanoseconds.
    size_bytes: int  # Nonnegative byte count.


@dataclass(frozen=True)
class CheckableFile:
    """Metadata and an MD5 reported by an endpoint for one unambiguous path."""

    stamp: FileStamp  # Modification time and size at this endpoint.
    md5: str  # Lowercase hex MD5 of the file bytes.


def data_filters() -> list[str]:
    """Return rclone filter arguments for the shared transfer and inventory scope."""
    options = ["--filter", "- *.lock"]
    for directory in DATA_DIRS:
        options += ["--filter", f"+ /{directory}/**"]
    return [*options, "--filter", "- **"]


def is_data_path(path: str) -> bool:
    """Check for a prefix-relative slash path in a data tree, excluding locks."""
    parts = path.split("/")
    return (
        len(parts) > 1
        and parts[0] in DATA_DIRS
        and all(part not in ("", ".", "..") for part in parts)
        and "\x00" not in path
        and not path.endswith(".lock")
    )


def file_stamp(mtime: object, size_bytes: object) -> FileStamp:
    """Validate external metadata and normalize the timestamp to UTC without rounding."""
    match = _RFC3339.fullmatch(mtime) if isinstance(mtime, str) else None
    if match is None or type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("Expected an RFC3339 mtime and a nonnegative integer size.")
    try:
        seconds = datetime.fromisoformat(match[1] + match[3]).astimezone(UTC)
    except OverflowError as exc:
        raise ValueError(
            f"Modification time is outside the UTC date range: {mtime}"
        ) from exc
    # datetime only carries microseconds; keep the RFC3339 fraction separately.
    fraction = (match[2] or "").rstrip("0")
    utc = seconds.isoformat().removesuffix("+00:00")
    return {
        "mtime": f"{utc}.{fraction}Z" if fraction else f"{utc}Z",
        "size_bytes": size_bytes,
    }


def list_checkable_files(
    binary: Path, config_dir: Path, root: str
) -> dict[str, CheckableFile]:
    """Read data files with known size and MD5 in one rclone listing of root.

    root is the local or named-remote prefix root. Use the verified binary and
    config_dir/rclone.conf. Omit duplicate paths and files without a usable MD5
    or size. Raise RuntimeError on listing failure or ValueError on invalid output.
    """
    result = rclone.run_cli(
        binary,
        config_dir,
        [
            "lsjson",
            root,
            "--files-only",
            "--recursive",
            "--fast-list",
            "--no-mimetype",
            "--hash-type",
            "MD5",
            *data_filters(),
        ],
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"Could not list {root} (rclone exit {result.returncode}).")
    return _parse_listing(result.stdout)


def _parse_listing(raw: str) -> dict[str, CheckableFile]:
    value: object = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("rclone returned a non-list file inventory.")
    files: dict[str, CheckableFile] = {}
    seen: set[str] = set()
    for item in value:
        path = _listed_path(item)
        if path in seen:
            # Drive can contain multiple objects at the same relative path.
            files.pop(path, None)
        else:
            seen.add(path)
            entry = _checkable_file(item)
            if entry is not None:
                files[path] = entry
    return files


def _listed_path(item: object) -> str:
    if (
        not isinstance(item, dict)
        or not isinstance(path := item.get("Path"), str)
        or not is_data_path(path)
        or item.get("IsDir") is not False
    ):
        raise ValueError("rclone returned an invalid data file path or type.")
    return path


def _checkable_file(item: dict[str, object]) -> CheckableFile | None:
    hashes = item.get("Hashes")
    md5: object = hashes.get("md5") if isinstance(hashes, dict) else None
    size = item.get("Size")
    if not isinstance(md5, str) or not _MD5.fullmatch(md5):
        return None
    if size == -1:  # rclone reports -1 when the size is unknown.
        return None
    return CheckableFile(file_stamp(item.get("ModTime"), size), md5.lower())
