"""Non-secret selection persisted by desktop setup and consumed by the CLI."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransferSettings:
    """Selected cloud parent shared by upload and download; contains no tokens."""

    # Named rclone remote and optional relative folder, e.g. lab:results.
    remote_parent: str


def validate_remote_parent(value: str) -> str:
    """Accept a named remote plus relative path; reject local/inline/escaping paths."""
    name, separator, folder = value.partition(":")
    if (
        not separator
        or not name
        or name.startswith(":")
        or any(char in name for char in "/\\,\r\n\x00")
    ):
        raise ValueError(
            "Expected a named rclone remote, e.g. lab:measurement-results."
        )
    if len(name) == 1:
        raise ValueError(
            "Use a remote name with at least two characters to avoid Windows drive letters."
        )
    if (
        folder.startswith("/")
        or "\\" in folder
        or any(char in folder for char in "\r\n\x00")
    ):
        raise ValueError("A cloud parent must be a named remote with a relative path.")
    if any(part in (".", "..") for part in folder.split("/")):
        raise ValueError("Cloud parent cannot contain traversal components.")
    return name + ":" + folder.rstrip("/")


def load_settings(config_dir: Path) -> TransferSettings | None:
    """Load selection or return None if absent; malformed settings raise ValueError."""
    path = config_dir / "transfer.json"
    if not path.exists():
        return None
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "remote_parent"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(value["remote_parent"], str)
    ):
        raise ValueError(f"Invalid transfer settings: {path}; run setup again.")
    return TransferSettings(validate_remote_parent(value["remote_parent"]))


def save_settings(config_dir: Path, selection: TransferSettings) -> None:
    """Atomically replace only transfer.json after explicit location confirmation."""
    parent = validate_remote_parent(selection.remote_parent)
    with tempfile.TemporaryDirectory(dir=config_dir) as staging:
        temporary = Path(staging) / "transfer.json"
        temporary.write_text(
            json.dumps(
                {"version": 1, "remote_parent": parent}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(config_dir / "transfer.json")


def validate_entry_name(value: str) -> str:
    """Accept one entry folder name, never a path or an rclone option."""
    if (
        not value
        or value in (".", "..")
        or value.startswith("-")
        or value.endswith((".", " "))
        or any(char in value for char in '/\\:*?"<>|\r\n\x00')
    ):
        raise ValueError(
            f"Invalid entry name: {value!r}; pass one folder name, not a path."
        )
    return value
