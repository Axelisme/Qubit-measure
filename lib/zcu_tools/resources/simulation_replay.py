"""Private simulation inputs, independent of normal measurement persistence.

Only environment authors opt in. Files contain model truth and must not be put in
RunMetadata. Waveforms are detached numeric timeline contents, not asset paths.
This owner checks JSON/storage; it does not interpret opaque numeric manifests.
"""

from hashlib import sha256
from os import fsync, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidateAs,
)

from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    EnvironmentSnapshot,
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    ReadoutWindow,
    TimelineTemplate,
)

from .ftf import FTFSettings


class _NumericJSON(BaseModel):
    """Strict wire records; from_attributes adapts core numeric dataclasses."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, from_attributes=True
    )


class _ArrayJSON(_NumericJSON):
    """Flat JSON values with explicit dtype and shape, never object arrays."""

    dtype: Literal["float64", "int64", "bool"]
    shape: tuple[int, ...]
    values: tuple[float | int | bool, ...]

    def array(self, dtype: np.dtype):
        if self.dtype != dtype.name:
            raise ValueError(
                f"expected replay array dtype {dtype.name}, got {self.dtype}"
            )
        if any(size < 0 for size in self.shape):
            raise ValueError("replay array shape must be nonnegative")
        return np.asarray(self.values, dtype=dtype).reshape(self.shape)


def _array_json(value: object) -> object:
    """Convert only ndarrays; JSON dictionaries go to the typed wire validator."""
    if isinstance(value, np.ndarray):
        return _ArrayJSON(
            dtype=value.dtype.name, shape=value.shape, values=tuple(value.flat)
        )
    return value


def _float_array(value: _ArrayJSON) -> NDArray[np.float64]:
    return value.array(np.dtype(np.float64))


def _int_array(value: _ArrayJSON) -> NDArray[np.int64]:
    return value.array(np.dtype(np.int64))


def _bool_array(value: _ArrayJSON) -> NDArray[np.bool_]:
    return value.array(np.dtype(np.bool_))


_Floats = Annotated[
    NDArray[np.float64],
    ValidateAs(_ArrayJSON, _float_array),
    BeforeValidator(_array_json),
    PlainSerializer(_array_json, return_type=_ArrayJSON),
]
_Ints = Annotated[
    NDArray[np.int64],
    ValidateAs(_ArrayJSON, _int_array),
    BeforeValidator(_array_json),
    PlainSerializer(_array_json, return_type=_ArrayJSON),
]
_Bools = Annotated[
    NDArray[np.bool_],
    ValidateAs(_ArrayJSON, _bool_array),
    BeforeValidator(_array_json),
    PlainSerializer(_array_json, return_type=_ArrayJSON),
]


class _EventJSON(_NumericJSON):
    """EventSlot wire layout with typed array codecs and detached envelopes."""

    gen_ch: int
    source: str
    iterations: tuple[tuple[str, int], ...]
    start_us: _Floats
    duration_us: _Floats
    frequency_mhz: _Floats
    phase_deg: _Floats
    gain: _Floats
    envelopes: tuple[PulseEnvelope | None, ...]
    envelope_index: _Ints
    active: _Bools
    phase_reference_us: float

    def record(self) -> EventSlot:
        return EventSlot(
            gen_ch=self.gen_ch,
            source=self.source,
            iterations=self.iterations,
            start_us=self.start_us,
            duration_us=self.duration_us,
            frequency_mhz=self.frequency_mhz,
            phase_deg=self.phase_deg,
            gain=self.gain,
            envelopes=self.envelopes,
            envelope_index=self.envelope_index,
            active=self.active,
            phase_reference_us=self.phase_reference_us,
        )


class _ReadoutJSON(_NumericJSON):
    """ReadoutSlot wire layout; array shape retains point and read axes."""

    channel: int
    samples: int
    sample_rate_mhz: float
    trigger_us: _Floats
    frequency_mhz: _Floats
    phase_deg: _Floats
    generator_channels: _Ints

    def record(self) -> ReadoutSlot:
        return ReadoutSlot(
            channel=self.channel,
            samples=self.samples,
            sample_rate_mhz=self.sample_rate_mhz,
            trigger_us=self.trigger_us,
            frequency_mhz=self.frequency_mhz,
            phase_deg=self.phase_deg,
            generator_channels=self.generator_channels,
        )


class _TemplateJSON(_NumericJSON):
    """TimelineTemplate wire layout; arrays keep their original dtype and shape."""

    points: _Ints
    events: tuple[_EventJSON, ...]
    readouts: tuple[_ReadoutJSON, ...]
    duration_us: _Floats

    def record(self) -> TimelineTemplate:
        return TimelineTemplate(
            self.points,
            tuple(event.record() for event in self.events),
            tuple(readout.record() for readout in self.readouts),
            self.duration_us,
        )


class _RequestJSON(_NumericJSON):
    """Template acquire wire layout with typed per-point arrays."""

    mode: AcquisitionMode
    loop_dims: tuple[int, ...]
    avg_level: int
    readouts: tuple[ReadoutWindow, ...]
    root_seed: int
    sequence: int
    rng_version: str
    point_dims: tuple[int, ...] = ()
    point_names: tuple[str, ...] = ()
    reps: int = 1
    templates: tuple[_TemplateJSON, ...] = ()

    def record(self) -> AcquireRequest:
        return AcquireRequest(
            mode=self.mode,
            loop_dims=self.loop_dims,
            avg_level=self.avg_level,
            readouts=self.readouts,
            root_seed=self.root_seed,
            sequence=self.sequence,
            rng_version=self.rng_version,
            point_dims=self.point_dims,
            point_names=self.point_names,
            reps=self.reps,
            templates=tuple(template.record() for template in self.templates),
        )


def _request_json(value: AcquireRequest) -> _RequestJSON:
    return _RequestJSON.model_validate(value)


_ReplayRequest = Annotated[
    AcquireRequest,
    ValidateAs(_RequestJSON, _RequestJSON.record),
    PlainSerializer(_request_json, return_type=_RequestJSON),
]


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
    request: _ReplayRequest
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
