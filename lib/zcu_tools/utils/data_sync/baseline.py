"""Load, verify and atomically save one Labdata synchronization baseline."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from zcu_tools.utils import atomic_replace

from .inventory import (
    FileInventory,
    FileStamp,
    ListedFile,
    file_stamp,
    is_data_path,
    list_files,
)

_VERSION = 1


class BaselineEntry(TypedDict):
    """Each endpoint's metadata when the two file contents were confirmed equal."""

    local: FileStamp  # Actual metadata from Labdata.
    remote: FileStamp  # Actual metadata from the configured remote.


@dataclass(frozen=True)
class BaselineStore:
    """Read and publish the baseline shared by checkouts using this data pair."""

    labdata: Path  # Absolute physical Labdata root.
    remote_root: str  # Validated named rclone root, without the prefix.
    prefix: str  # Validated slash-separated path below both roots.

    @property
    def path(self) -> Path:
        """Return the per-pair JSON path, without reading or creating it."""
        identity = json.dumps([self.remote_root, self.prefix], ensure_ascii=False)
        pair_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        # Persist one record per pair in <labdata>/.data-sync/state/.
        return self.labdata / ".data-sync" / "state" / f"{pair_id}.json"

    def load(self) -> dict[str, BaselineEntry] | None:
        """Read the saved stamps, return None for no record, or raise on unreadable data."""
        path = self.path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot read sync baseline {path}: {exc}") from exc
        try:
            return self._decode(raw)
        except (RecursionError, ValueError) as exc:
            raise ValueError(f"Invalid sync baseline {path}: {exc}") from exc

    def _decode(self, raw: str) -> dict[str, BaselineEntry]:
        value: object = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        version = value.get("version")
        if type(version) is not int or version != _VERSION:
            raise ValueError("Unsupported baseline version.")
        if (
            value.get("remote_root") != self.remote_root
            or value.get("prefix") != self.prefix
        ):
            raise ValueError("Saved cloud root or prefix does not match this pair.")
        files: object = value.get("files")
        if not isinstance(files, dict):
            raise ValueError("Expected a files object.")
        return {path: _decode_entry(path, entry) for path, entry in files.items()}

    def save(self, files: dict[str, BaselineEntry]) -> None:
        """Atomically replace this pair's record with validated, verified file stamps.

        The caller preserves previous entries not known to match. Create the state
        directory as needed. Propagate I/O failures without replacing the old file.
        """
        path = self.path
        value = {
            "version": _VERSION,
            "remote_root": self.remote_root,
            "prefix": self.prefix,
            "files": files,
        }
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as staging:
            temporary = Path(staging) / path.name
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            atomic_replace(temporary, path)


def _decode_entry(path: str, entry: object) -> BaselineEntry:
    if not is_data_path(path) or not isinstance(entry, dict):
        raise ValueError(f"Invalid file entry for {path!r}.")
    local = _decode_stamp(entry.get("local"))
    remote = _decode_stamp(entry.get("remote"))
    if local["size_bytes"] != remote["size_bytes"]:
        raise ValueError(f"Local and remote sizes differ for {path!r}.")
    return {"local": local, "remote": remote}


def _decode_stamp(value: object) -> FileStamp:
    if not isinstance(value, dict):
        raise ValueError("Expected local and remote file metadata.")
    return file_stamp(value.get("mtime"), value.get("size_bytes"))


def verify_pair(
    binary: Path, config_dir: Path, local: str, remote: str, *, push: bool
) -> dict[str, BaselineEntry]:
    """Read the two prefix roots after a copy and return only verified file stamps.

    Use the verified rclone binary and config_dir/rclone.conf. With push, local is
    the source; otherwise remote is the source. The source is listed first so an
    empty source does not require a destination directory. Only paths with equal
    sizes and matching MD5 at both endpoints are recorded. Raise on a failed or
    invalid listing; matching metadata alone cannot verify a file.
    """
    origin, destination = (local, remote) if push else (remote, local)
    source = list_files(binary, config_dir, origin)
    if not any(
        file.md5 is not None and file.size_bytes is not None
        for file in source.files.values()
    ):
        return {}
    target = list_files(binary, config_dir, destination)
    return (
        verified_entries(source, target) if push else verified_entries(target, source)
    )


def verified_entry(local: ListedFile, remote: ListedFile) -> BaselineEntry | None:
    """Return the two stamps only when size and MD5 prove the contents match."""
    if local.md5 is None or local.md5 != remote.md5:
        return None
    left, right = local.stamp(), remote.stamp()
    if left is None or right is None or left["size_bytes"] != right["size_bytes"]:
        return None
    return {"local": left, "remote": right}


def verified_entries(
    local: FileInventory, remote: FileInventory
) -> dict[str, BaselineEntry]:
    """Return verified stamps for matching files uniquely listed on both sides."""
    return {
        path: entry
        for path in local.files.keys() & remote.files.keys()
        if (entry := verified_entry(local.files[path], remote.files[path])) is not None
    }
