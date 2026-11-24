"""Non-secret checkout settings for the data trees synchronized by rclone."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.utils import atomic_replace

DATA_DIRS = ("result", "Database")
_DATA_DIRS_FOLDED = frozenset(name.casefold() for name in DATA_DIRS)


def is_data_dir(segment: str) -> bool:
    """Identify a reserved data directory name, including case aliases."""
    return segment.casefold() in _DATA_DIRS_FOLDED


@dataclass(frozen=True)
class SyncSettings:
    """Validated selection saved by setup; does not contain account credentials."""

    remote_root: str  # Named rclone remote and optional relative root, e.g. lab:data.
    source: str  # This checkout's slash-separated prefix under remote_root.
    labdata: Path  # Absolute local collection root for pull.

    def remote_path(self, relative: str) -> str:
        """Append a validated slash-separated path to the selected cloud root."""
        separator = "" if self.remote_root.endswith(":") else "/"
        return f"{self.remote_root}{separator}{relative}"


def validate_remote_root(value: str) -> str:
    """Accept a named remote plus optional relative folder, never a local path."""
    name, separator, folder = value.partition(":")
    if not separator or len(name) < 2 or any(char in name for char in "/\\,\r\n\x00"):
        raise ValueError("Expected a named rclone remote, e.g. lab:measurements.")
    if folder.startswith("/") or any(char in folder for char in "\\\r\n\x00"):
        raise ValueError("A cloud root must be a named remote with a relative path.")
    if any(part in (".", "..") for part in folder.split("/")):
        raise ValueError("Cloud root cannot contain traversal components.")
    return name + ":" + folder.rstrip("/")


def validate_source(value: str) -> str:
    """Accept slash-separated safe folder names outside result and Database trees."""
    if not all(
        _is_folder_name(part) and not is_data_dir(part) for part in value.split("/")
    ):
        raise ValueError(
            f"Invalid source: {value!r}; use a relative prefix without result or Database segments."
        )
    return value


def _is_folder_name(value: str) -> bool:
    return (
        bool(value)
        and value not in (".", "..")
        and not value.startswith("-")
        and not value.endswith((".", " "))
        and not any(char in value for char in '/\\:*?"<>|\r\n\x00')
    )


def load_settings(settings_dir: Path) -> SyncSettings | None:
    """Read sync.json from settings_dir; return None only if it does not exist."""
    path = settings_dir / "sync.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return _decode_settings(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid sync settings: {path}; run setup again. {exc}"
        ) from exc


def _decode_settings(raw: str) -> SyncSettings:
    value: object = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "remote_root", "source", "labdata"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(value["remote_root"], str)
        or not isinstance(value["source"], str)
        or not isinstance(value["labdata"], str)
    ):
        raise ValueError("Expected version 1 with remote_root, source and labdata.")
    labdata = Path(value["labdata"])
    if not labdata.is_absolute():
        raise ValueError("Saved labdata must be an absolute path.")
    return SyncSettings(
        validate_remote_root(value["remote_root"]),
        validate_source(value["source"]),
        labdata,
    )


def save_settings(settings_dir: Path, selection: SyncSettings) -> None:
    """Atomically replace sync.json in an existing directory with validated settings."""
    value = {
        "version": 1,
        "remote_root": selection.remote_root,
        "source": selection.source,
        "labdata": str(selection.labdata),
    }
    with tempfile.TemporaryDirectory(dir=settings_dir) as staging:
        temporary = Path(staging) / "sync.json"
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.chmod(0o600)
        atomic_replace(temporary, settings_dir / "sync.json")
