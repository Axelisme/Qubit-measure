"""Immutable, checked numeric archives. Directory policy belongs to callers."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from os import fsync, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class ArchiveHeader(BaseModel):
    """Strict numeric envelope: version/kind/key identify the stored calculation.

    family/lower/upper/degree describe a plateau interpolation domain. evidence
    contains measured validation numbers for published tables, empty otherwise.
    duration_ns is the positive quantized plateau length. ratios contains one
    (real,imaginary) dimensionless source amplitude per ordered model source.
    They are zero/empty for direct records. All versions are exact; there is no
    permissive compatibility reader.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal[1] = 1
    kind: Literal["partial", "table", "direct"]
    key: str
    family: str = ""
    lower: float = 0
    upper: float = 0
    degree: int = 0
    evidence: tuple[tuple[str, float], ...] = ()
    duration_ns: float = 0
    ratios: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class NumericArchive:
    """Header plus owned arrays; nodes/present are empty for exact density records.

    values is complex128 grid matrices (table/partial) or one density (direct).
    present is the boolean completed-node mask for resumable construction.
    """

    header: ArchiveHeader
    nodes: NDArray[np.float64]
    values: NDArray[np.complex128]
    present: NDArray[np.bool_]

    @property
    def nbytes(self) -> int:
        """Bytes in all resident ndarray payloads (host only)."""
        return self.nodes.nbytes + self.values.nbytes + self.present.nbytes


def encode_archive(archive: NumericArchive) -> bytes:
    """Encode exact values with a strict header, without pickle/object arrays."""
    stream = BytesIO()
    np.savez(
        stream,
        header=archive.header.model_dump_json(),
        nodes=archive.nodes,
        values=archive.values,
        present=archive.present,
    )
    payload = stream.getvalue()
    return sha256(payload).hexdigest().encode() + b"\n" + payload


def read_archive(path: Path, digest: str | None = None) -> NumericArchive:
    """Read/check a numeric file; corrupt data is ValueError, I/O errors propagate.

    digest is an optional SHA256 of the complete file. Payloads are immutable.
    This validates numerical storage, not physical applicability/convergence.
    """
    payload = path.read_bytes()
    if digest is not None and sha256(payload).hexdigest() != digest:
        raise ValueError(f"numeric archive integrity failure: {path}")
    checksum, separator, numeric = payload.partition(b"\n")
    if not separator or checksum != sha256(numeric).hexdigest().encode():
        raise ValueError(f"numeric archive integrity failure: {path}")
    try:
        with np.load(BytesIO(numeric), allow_pickle=False) as data:
            header = ArchiveHeader.model_validate_json(str(data["header"].item()))
            nodes = np.asarray(data["nodes"], dtype=np.float64)
            values = np.asarray(data["values"], dtype=np.complex128)
            present = np.asarray(data["present"], dtype=np.bool_)
    except (ValueError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(f"corrupt numeric archive: {path}") from error
    archive = NumericArchive(header, nodes, values, present)
    _validate_payload(archive)
    for array in (nodes, values, present):
        array.setflags(write=False)
    return archive


def _validate_payload(archive: NumericArchive) -> None:
    if not np.isfinite(archive.values).all() or not np.isfinite(archive.nodes).all():
        raise ValueError("corrupt numeric archive: nonfinite values")
    if archive.header.kind in ("table", "partial"):
        _validate_grid(archive)
    elif archive.values.ndim != 2 or archive.values.shape[0] != archive.values.shape[1]:
        raise ValueError("corrupt numeric archive: density dimensions")


def _validate_grid(archive: NumericArchive) -> None:
    header, nodes, values, present = (
        archive.header,
        archive.nodes,
        archive.values,
        archive.present,
    )
    if (
        header.degree not in (4, 6)
        or not header.lower < header.upper
        or nodes.shape != (header.degree + 1,)
        or present.shape != nodes.shape
        or values.ndim != 3
        or values.shape[0] != len(nodes)
        or values.shape[1] != values.shape[2]
        or np.any(np.diff(nodes) <= 0)
    ):
        raise ValueError("corrupt numeric archive: table dimensions")
    if header.kind == "table" and (not present.all() or not header.evidence):
        raise ValueError("corrupt numeric archive: incomplete publication")


def write_archive(path: Path, archive: NumericArchive) -> str:
    """Atomically replace an explicit file; fsync before publish, failures propagate.

    Temporary files stay beside the destination and are removed on all paths.
    Caller owns locking when mutable partial state may have several writers.
    Return complete-file SHA256, suitable for immutable content references.
    """
    payload = encode_archive(archive)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".numeric-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(payload)
            file.flush()
            fsync(file.fileno())
        replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256(payload).hexdigest()
