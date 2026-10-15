"""Private simulation inputs, independent of normal measurement persistence.

Only environment authors opt in. Files contain model truth and must not be put in
RunMetadata. Waveforms are detached numeric timeline contents, not asset paths.
This owner checks JSON/storage; it does not interpret opaque numeric manifests.
"""

from hashlib import sha256
from os import fsync, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from zcu_tools.core.simulation import AcquireRequest, EnvironmentSnapshot

from .ftf import FTFSettings


class ReplayAcquire(BaseModel):
    """Fixed private acquire inputs shared by independently replayable rounds.

    version: exact private schema version 1; no compatibility reader. settings:
    full immutable FTF resource values, including model/calibration/solver/seed.
    request: detached quantized controls, owned complex envelopes, windows/layout
    and RNG identity. versions: ordered (component, exact version) pairs supplied
    by the backend, checked by replay execution. library_path: explicit absolute
    numeric-library root, or None when no numeric persistence was used.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal[1] = 1
    settings: FTFSettings
    request: AcquireRequest
    versions: tuple[tuple[str, str], ...]
    library_path: Path | None = None


class ReplayRound(BaseModel):
    """One detached round; no generated IQ, densities or Python callables.

    version: exact private schema 1. acquire_identity: complete acquire file
    SHA256 resolved inside the archive. snapshot: current native device values
    and model identity. index: nonnegative original software round number.
    numeric_manifest: opaque numeric-owner receipt string or None; resources
    neither decodes its internal format nor substitutes current publications.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal[1] = 1
    acquire_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot: EnvironmentSnapshot
    index: int = Field(ge=0)
    numeric_manifest: str | None = None


class ReplayInputs(BaseModel):
    """Resolved author-only inputs: acquire content and its selected round."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    acquire: ReplayAcquire
    round: ReplayRound


def _write(directory: Path, payload: bytes) -> str:
    identity = sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identity}.json"
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"private replay integrity failure: {target}")
        return identity
    with NamedTemporaryFile(dir=directory, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            temporary.write(payload)
            temporary.flush()
            fsync(temporary.fileno())
            temporary.close()
            replace(path, target)
        finally:
            path.unlink(missing_ok=True)
    return identity


def _read(path: Path) -> bytes:
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != path.stem:
        raise ValueError(f"private replay integrity failure: {path}")
    return payload


def save_replay(
    root: Path,
    acquire: ReplayAcquire,
    snapshot: EnvironmentSnapshot,
    index: int,
    numeric_manifest: str | None = None,
) -> Path:
    """Persist one round and deduplicated acquire; return its immutable JSON path.

    root must be an explicit absolute private archive directory. No output signal
    is saved. acquire owns all timeline/waveform/configuration content. snapshot
    and index identify the selected round. numeric_manifest is the numeric
    owner's opaque reference envelope, not decoded here. Missing dependencies,
    schema/checksum errors and I/O failures propagate. Writes use sibling temp,
    fsync and atomic replace; the two files are not a crash transaction. An
    unreferenced acquire may remain if writing the round fails. Never overwrites
    a damaged content-addressed file. No arbitrary object/callable serialization.
    """
    if not root.is_absolute():
        raise ValueError("private replay requires an explicit absolute root")
    identity = _write(root / "acquires", acquire.model_dump_json().encode())
    record = ReplayRound(
        acquire_identity=identity,
        snapshot=snapshot,
        index=index,
        numeric_manifest=numeric_manifest,
    )
    digest = _write(root / "rounds", record.model_dump_json().encode())
    return root / "rounds" / f"{digest}.json"


def read_replay(path: Path) -> ReplayInputs:
    """Resolve/check a selected private round and its immutable acquire content.

    path is a round path returned by save_replay/replay_paths. Both JSON file
    hashes and exact schemas are checked. Missing/corrupt files and unsupported
    versions raise instead of using current YAML/assets. Numeric references are
    returned opaque for their owner to check before execution. No devices or
    programs are opened, and no previous round/acquire is evaluated.
    """
    record = ReplayRound.model_validate_json(_read(path))
    acquire_path = path.parent.parent / "acquires" / f"{record.acquire_identity}.json"
    acquire = ReplayAcquire.model_validate_json(_read(acquire_path))
    return ReplayInputs(acquire=acquire, round=record)


def replay_paths(root: Path) -> tuple[Path, ...]:
    """List saved round paths in hash order, without loading or executing them.

    root is the explicitly selected private archive directory. A missing root
    returns an empty tuple. read_replay performs content/schema verification.
    """
    return tuple(sorted((root / "rounds").glob("*.json")))
