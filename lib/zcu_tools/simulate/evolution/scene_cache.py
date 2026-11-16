"""Versioned scene identities and atomic numeric cache I/O, without Git policy.

Callers supply explicit index/cache paths. No source, Engine or solver is opened.
Index order is preserved. Invalid versions, names or identities raise ValueError;
JSON and filesystem errors propagate. Returned records own only numeric data.
"""

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .frozen import (
    FrozenRecord,
    canonical_json,
    digest,
    read_frozen_manifest,
    write_frozen_manifest,
)

SCENE_CACHE_VERSION = 1


@dataclass(frozen=True)
class FrozenIndexEntry:
    """Pinned record name and full physical, oracle and opaque metadata digests.

    No numeric payload or solver policy is stored. Use from_record rather than
    calculating identity in callers. The entry also owns cache-file naming.
    """

    name: str
    physical_sha256: str
    oracle_sha256: str
    metadata_sha256: str

    @classmethod
    def from_record(cls, record: FrozenRecord) -> "FrozenIndexEntry":
        """Return this record's identity using the public canonical digest."""
        return cls(
            record.name,
            record.physical_sha256,
            record.oracle_sha256,
            digest(record.metadata),
        )

    def cache_path(self, directory: Path) -> Path:
        """Return the version-1 single-record path in an explicit cache directory."""
        return directory / (
            f"{self.name}-{self.physical_sha256[:16]}-"
            f"{self.metadata_sha256[:16]}.json.gz"
        )


def _validate_name(name: object) -> None:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("scene index name must be a nonempty filename component")


def _validate_hash(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"invalid scene index hash: {name}")


def _entries(entries: Iterable[FrozenIndexEntry]) -> tuple[FrozenIndexEntry, ...]:
    result = tuple(entries)
    names: set[str] = set()
    for entry in result:
        _validate_name(entry.name)
        if entry.name in names:
            raise ValueError(f"duplicate scene index name: {entry.name}")
        names.add(entry.name)
        for value in (
            entry.physical_sha256,
            entry.oracle_sha256,
            entry.metadata_sha256,
        ):
            _validate_hash(value, entry.name)
    return result


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_frozen_index(path: Path, entries: Iterable[FrozenIndexEntry]) -> None:
    """Atomically replace a version-1 JSON index after validating all entries.

    Create parents; serialize fully before publication. An invalid entry or I/O
    failure before replace leaves the previous index readable. No cache is read.
    """
    payload = {
        "version": SCENE_CACHE_VERSION,
        "inputs": [asdict(entry) for entry in _entries(entries)],
    }
    _publish(path, (canonical_json(payload) + "\n").encode())


def read_frozen_index(path: Path) -> tuple[FrozenIndexEntry, ...]:
    """Read and validate ordered identities; unknown versions fail, never fall back."""
    payload = json.loads(path.read_text())
    if payload["version"] != SCENE_CACHE_VERSION:
        raise ValueError("unsupported frozen scene index version")
    return _entries(FrozenIndexEntry(**entry) for entry in payload["inputs"])


def read_scene_records(
    directory: Path, entries: Iterable[FrozenIndexEntry]
) -> tuple[FrozenRecord, ...]:
    """Read exactly the indexed single-record gzip files and verify all identities.

    Missing files raise FileNotFoundError naming the record and hashes. Other
    container/hash errors raise ValueError. No fallback, model or solver runs.
    Consumers may select entries first to avoid touching unselected cache files.
    """
    result = []
    for entry in _entries(entries):
        path = entry.cache_path(directory)
        try:
            records = read_frozen_manifest(path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing frozen scene {entry.name}: physical_sha256={entry.physical_sha256} "
                f"metadata_sha256={entry.metadata_sha256}; {path}"
            ) from error
        if len(records) != 1 or FrozenIndexEntry.from_record(records[0]) != entry:
            raise ValueError(f"frozen scene index hash/name mismatch: {entry.name}")
        result.append(records[0])
    return tuple(result)


def write_scene_record(directory: Path, record: FrozenRecord) -> FrozenIndexEntry:
    """Atomically publish one manifest and return its pinned index identity.

    Existing valid files are reused, never overwritten. A conflicting existing
    file fails validation. Serialization completes in the same directory before
    replace; failures preserve existing pinned files and clean temporary files.
    """
    (entry,) = _entries((FrozenIndexEntry.from_record(record),))
    path = entry.cache_path(directory)
    if path.exists():
        read_scene_records(directory, (entry,))
        return entry
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=directory, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        write_frozen_manifest(temporary, (record,))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return entry
