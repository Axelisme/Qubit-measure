"""Layout-independent numeric input codec (ADR-0078).

Canonical physics and its hashes retain the original family-table format.
JSON decoding is confined here; numeric record constructors validate the arrays.
No model builder, routing history, device layout or solver is persisted.
"""

import gzip
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .engine import DriveTrack, EnvelopeSpec, TimelineBatch
from .secular import (
    FluxExpansion,
    FluxSchedule,
    FluxTerm,
    FluxWaveform,
    ModelSource,
    SecularDrive,
    SecularModel,
    SegmentShift,
)

FROZEN_PHYSICAL_VERSION = 1
"""Physical decoding contract shared by manifests and standalone inputs.

Bump when fields, units, dtypes, defaults or Engine-input reconstruction change.
This is not SEGMENT_NUMERIC_VERSION: numerical algorithm changes alone do not
change the serialized input contract. Producers own migration after a bump.
"""


@dataclass(frozen=True)
class FrozenInput:
    """Decoded Engine inputs; the caller closes source after use.

    source contains only the enumerated frozen flux levels and a fresh owned
    routing history. Requesting an unlisted level raises KeyError.
    batch preserves original point order and the public record array dtypes.
    """

    source: ModelSource
    batch: TimelineBatch


@dataclass(frozen=True)
class FrozenRecord:
    """Named numeric physics with opaque caller-owned JSON metadata.

    Hash properties reflect the current physical mapping: full identity for
    physical_sha256 and only model namespaces removed for oracle_sha256.
    Writers calculate both; readers verify stored hashes before returning.
    Metadata is stored without interpreting caller fields. It carries no codec
    or solver policy. Constructing/reading a record does not construct a model.
    """

    name: str
    physical: dict[str, object]
    metadata: object = None

    @property
    def physical_sha256(self) -> str:
        return digest(self.physical)

    @property
    def oracle_sha256(self) -> str:
        physical = json.loads(canonical_json(self.physical))
        for model in physical["models"]:
            del model["identity"]
        return digest(physical)


def canonical_json(value: object) -> str:
    """Encode deterministic numeric records; reject nonfinite/unsupported values.

    Arrays become nested lists, complex scalars become real/imag objects, keys
    are sorted, and compact separators preserve historical physical hashes.
    """

    def encode(item: object) -> object:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, complex):
            return {"real": item.real, "imag": item.imag}
        raise TypeError(f"unsupported manifest value: {type(item).__name__}")

    return json.dumps(
        value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: object) -> str:
    """Return SHA-256 of canonical_json(value), including ordered numeric arrays."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def physical_input(
    source: ModelSource, batch: TimelineBatch, *, include_identity: bool = True
) -> dict[str, object]:
    """Snapshot complete Engine physics without evolving or closing source.

    Include scheduled flux levels and the all-zero DC level, initial density,
    optional continuous-flux expansion and slow-envelope subdivisions. False
    removes only model identity namespaces, allowing reference sharing. Arrays
    follow their public numeric-record dtypes. Registered source shifts and
    routing history are not persisted; batch segment corrections are included.
    Model construction and missing expansion failures propagate.
    """
    levels = sorted(
        {tuple(int(v) for v in row) for row in batch.flux.level}
        | {(0,) * batch.flux.level.shape[1]}
    )
    models = [asdict(source.model(level)) for level in levels]
    if not include_identity:
        for model in models:
            model.pop("identity")
    return {
        "batch": asdict(batch),
        "initial_density": source.initial_density,
        "models": models,
        "expansion": asdict(source.flux_expansion())
        if batch.flux_waveform is not None
        else None,
        "slow_envelope_segments_per_period": source.slow_envelope_segments_per_period,
    }


def _complex_object(value: dict[str, Any]) -> object:
    if value.keys() == {"real", "imag"}:
        return complex(value["real"], value["imag"])
    return value


def _track(value: dict[str, Any]) -> DriveTrack:
    envelopes = tuple(
        EnvelopeSpec(
            e["kind"], np.asarray(e["samples"], complex), np.asarray(e["sample_ns"])
        )
        for e in value["envelopes"]
    )
    return DriveTrack(
        value["port"],
        np.asarray(value["start_ns"]),
        np.asarray(value["duration_ns"]),
        np.asarray(value["frequency_ghz"]),
        np.asarray(value["phase_rad"]),
        np.asarray(value["amplitude"]),
        envelopes,
        np.asarray(value["envelope_index"]),
        np.asarray(value["active"]),
    )


def _batch(value: dict[str, Any], loops: int, size: int) -> TimelineBatch:
    flux = value["flux"]
    waveform = value["flux_waveform"]
    # Empty JSON lists lose dtype and trailing dimensions; DC model levels
    # retain the loop count even when the batch has no scheduled segments.
    # The source density retains M for empty continuation densities (0,M,M).
    points = np.asarray(
        value["points"], dtype=np.int64 if value["points"] == [] else None
    )
    levels = (
        np.empty((0, loops), dtype=np.int64)
        if value["points"] == [] and flux["level"] == []
        else np.asarray(flux["level"])
    )
    return TimelineBatch(
        points,
        tuple(_track(track) for track in value["tracks"]),
        FluxSchedule(
            np.asarray(flux["offsets"]),
            np.asarray(flux["start_ns"]),
            levels,
        ),
        np.asarray(value["end_ns"]),
        None
        if waveform is None
        else FluxWaveform(
            np.asarray(waveform["offsets"]),
            np.asarray(waveform["time_ns"]),
            np.empty((0, loops), dtype=np.float64)
            if value["points"] == [] and waveform["flux"] == []
            else np.asarray(waveform["flux"]),
        ),
        tuple(tuple(times) for times in value["segment_boundaries"]),
        tuple(
            tuple(SegmentShift(**shift) for shift in shifts)
            for shifts in value["segment_shifts"]
        ),
        None if value["start_ns"] is None else np.asarray(value["start_ns"]),
        None
        if value["initial_densities"] is None
        else np.empty((0, size, size), dtype=np.complex128)
        if value["points"] == [] and value["initial_densities"] == []
        else np.asarray(value["initial_densities"], complex),
    )


def _model(value: dict[str, Any]) -> SecularModel:
    return SecularModel(
        tuple(value["level"]),
        value["identity"],
        np.asarray(value["energies_ghz"]),
        np.asarray(value["eigenvectors"], complex),
        tuple(
            SecularDrive(
                d["name"],
                np.asarray(d["in_phase_ghz"], complex),
                np.asarray(d["quadrature_ghz"], complex),
            )
            for d in value["drives"]
        ),
        np.asarray(value["rates_per_ns"]),
        np.asarray(value["dephasing_per_ns"]),
    )


def read_physical_input(value: dict[str, object]) -> FrozenInput:
    """Reconstruct a physical_input snapshot, including JSON-decoded snapshots.

    Numeric record constructors reject invalid shape, ordering and physics.
    Missing fields raise KeyError; malformed JSON values raise TypeError or
    ValueError. Identity-free snapshots cannot reconstruct a source. No model
    is built outside the finite supplied mapping and no solver is invoked.
    """
    decoded = json.loads(canonical_json(value), object_hook=_complex_object)
    models = {model.level: model for model in map(_model, decoded["models"])}
    expansion = decoded["expansion"]
    restored_expansion = (
        None
        if expansion is None
        else FluxExpansion(
            np.asarray(expansion["static"], complex),
            tuple(
                FluxTerm(t["loop"], t["kind"], np.asarray(t["matrix"], complex))
                for t in expansion["terms"]
            ),
        )
    )
    return FrozenInput(
        ModelSource(
            np.asarray(decoded["initial_density"], complex),
            models.__getitem__,
            restored_expansion,
            slow_envelope_segments_per_period=decoded[
                "slow_envelope_segments_per_period"
            ],
        ),
        _batch(
            decoded["batch"],
            len(decoded["models"][0]["level"]),
            len(decoded["initial_density"]),
        ),
    )


def _unique_names(records: tuple[FrozenRecord, ...]) -> None:
    # JSON callers can supply invalid runtime values despite the record hint.
    names: list[object] = [record.name for record in records]
    if any(not isinstance(name, str) for name in names):
        raise TypeError("frozen manifest names must be strings")
    if len(set(names)) != len(names):
        raise ValueError("duplicate frozen manifest name")


def write_frozen_manifest(path: Path, records: Iterable[FrozenRecord]) -> None:
    """Write records with FROZEN_PHYSICAL_VERSION inside canonical stdlib gzip.

    Calculate both physical hashes, reject duplicate/non-string names and keep
    caller metadata opaque. gzip has mtime=0 and no filename, so identical inputs
    produce identical bytes regardless of path. Create parents and replace the
    destination. Serialization and I/O errors propagate. No model or solver runs.
    """
    records = tuple(records)
    _unique_names(records)
    payload = {
        "version": FROZEN_PHYSICAL_VERSION,
        "inputs": [
            {
                "name": record.name,
                "physical": record.physical,
                "physical_sha256": record.physical_sha256,
                "oracle_sha256": record.oracle_sha256,
                "metadata": record.metadata,
            }
            for record in records
        ],
    }
    data = (canonical_json(payload) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0))


def read_frozen_manifest(path: Path) -> tuple[FrozenRecord, ...]:
    """Read a gzip manifest, preserving order and opaque JSON metadata.

    Check FROZEN_PHYSICAL_VERSION, unique names and both physical hashes before
    returning numeric records. Bad versions/hashes/duplicates raise ValueError.
    Missing fields, invalid JSON/gzip and I/O errors propagate. Reconstruct a
    caller-owned source separately with read_physical_input(record.physical).
    """
    payload = json.loads(gzip.decompress(path.read_bytes()))
    if payload["version"] != FROZEN_PHYSICAL_VERSION:
        raise ValueError("unsupported frozen manifest version")
    records = tuple(
        FrozenRecord(record["name"], record["physical"], record["metadata"])
        for record in payload["inputs"]
    )
    _unique_names(records)
    for record, stored in zip(records, payload["inputs"], strict=True):
        if record.physical_sha256 != stored["physical_sha256"]:
            raise ValueError(f"frozen manifest physical hash mismatch: {record.name}")
        if record.oracle_sha256 != stored["oracle_sha256"]:
            raise ValueError(f"frozen manifest oracle hash mismatch: {record.name}")
    return records


def write_frozen_input(path: Path, source: ModelSource, batch: TimelineBatch) -> None:
    """Write a FROZEN_PHYSICAL_VERSION input and its physical hash to path.

    Parent directories are created and an existing file is replaced. Serialization
    and I/O errors propagate. No solver or Engine work runs.
    """
    physical = physical_input(source, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        canonical_json(
            {
                "version": FROZEN_PHYSICAL_VERSION,
                "physical_sha256": digest(physical),
                "physical": physical,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def read_frozen_input(path: Path) -> FrozenInput:
    """Read a standalone input, checking FROZEN_PHYSICAL_VERSION and content hash.

    Raise ValueError for unknown versions or altered content. Missing/malformed
    files and invalid numeric records fail without building a model or evolving.
    The returned source is caller-owned and must be closed.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["version"] != FROZEN_PHYSICAL_VERSION:
        raise ValueError("unsupported frozen input version")
    if digest(record["physical"]) != record["physical_sha256"]:
        raise ValueError("frozen input physical hash mismatch")
    return read_physical_input(record["physical"])
