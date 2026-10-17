"""Acquire-level segment engine and immutable ns/GHz timeline records."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Event, Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .kernel import Kernel

import numpy as np
from numpy.typing import NDArray

from ._envelope import sample_values
from ._parallel import execution_pool, native_limit
from ._validation import integers, owned, shape
from .secular import FluxSchedule, FluxWaveform, ModelSource

# Replay identity of the segment engine numerical algorithms, independent of records.
SEGMENT_NUMERIC_VERSION = "segment-engine-v3"


@dataclass(frozen=True)
class EnvelopeSpec:
    """Immutable envelope description; identity hashes its complete content.

    kind: constant (unity) or sampled. Analytic pulse shapes arrive sampled.
    samples: complex128 (n,) dimensionless amplitudes, with DRAG Q in the
        imaginary part. sample_ns: float64 (n,) increasing pulse-local ns
        starting at zero. Both arrays are empty for constant envelopes.
    identity: automatically computed stable SHA256 of kind, samples and sample
        times, not caller metadata or a Python object identity.
    Arrays own read-only copies.

    Raises:
        ValueError: message names invalid kind, finite values, shape, zero
        origin, increasing times, or nonempty constant samples.
    """

    kind: Literal["constant", "sampled"]
    samples: NDArray[np.complex128] = field(
        default_factory=lambda: np.empty(0, dtype=np.complex128)
    )
    sample_ns: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in ("constant", "sampled"):
            raise ValueError("unknown envelope kind")
        samples = owned(self.samples, np.complex128, "envelope samples")
        times = owned(self.sample_ns, np.float64, "envelope sample times")
        if samples.ndim != 1:
            raise ValueError("envelope samples must be a vector")
        shape(times, samples.shape, "envelope sample times")
        if self.kind == "sampled":
            if not len(times) or times[0] != 0:
                raise ValueError("sampled envelope times must start at zero")
            if np.any(np.diff(times) <= 0):
                raise ValueError("sampled envelope times must be increasing")
        elif len(samples):
            raise ValueError("constant envelopes cannot contain samples")
        digest = sha256(self.kind.encode("ascii"))
        digest.update(samples.astype("<c16").tobytes())
        digest.update(times.astype("<f8").tobytes())
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_ns", times)
        object.__setattr__(self, "identity", digest.hexdigest())

    def values(self, local_ns: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Evaluate finite nonnegative pulse-local times, retaining input shape.

        Interpolate real and imaginary parts linearly and hold the final sample
        through pulse end. Constant envelopes return unity. Pulse support is
        enforced by tracks/segments, not this evaluator.

        Raises:
            ValueError: times are not finite or are negative.
        """
        times = np.asarray(local_ns, dtype=np.float64)
        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("envelope evaluation times must be finite and nonnegative")
        return sample_values(times, self.sample_ns, self.samples)


@dataclass(frozen=True)
class DriveTrack:
    """One drive slot across points; every per-point array has shape (n,).

    port: nonempty SecularDrive name.
    start_ns, duration_ns: nonnegative float64 support [start,start+duration).
    frequency_ghz: signed float64 carrier in cycles/ns.
    phase_rad: float64 phase at common absolute t=0, including virtual Z.
    amplitude: signed float64 real gain times the port's input scale.
    envelopes: candidate EnvelopeSpec tuple; identities enter prepare keys.
    envelope_index: int64 index into envelopes for each point.
    active: bool mask per point; false means no drive in this slot.
    Arrays own read-only copies. Invalid shapes, times or indices raise
    ValueError. Readout truncation belongs to Engine, not this record.

    Raises:
        ValueError: message names shape, nonnegative support, or envelope
        index when that condition fails.
    """

    port: str
    start_ns: NDArray[np.float64]
    duration_ns: NDArray[np.float64]
    frequency_ghz: NDArray[np.float64]
    phase_rad: NDArray[np.float64]
    amplitude: NDArray[np.float64]
    envelopes: tuple[EnvelopeSpec, ...]
    envelope_index: NDArray[np.int64]
    active: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if not self.port or np.asarray(self.start_ns).ndim != 1:
            raise ValueError("drive requires a port and per-point vectors")
        expected = self.start_ns.shape
        for name in (
            "start_ns",
            "duration_ns",
            "frequency_ghz",
            "phase_rad",
            "amplitude",
        ):
            array = owned(getattr(self, name), np.float64, name)
            shape(array, expected, name)
            object.__setattr__(self, name, array)
        if np.any(self.start_ns < 0) or np.any(self.duration_ns < 0):
            raise ValueError("drive support must be nonnegative")
        indices = integers(self.envelope_index, "envelope indices")
        shape(indices, expected, "envelope indices")
        if np.any(indices < 0) or np.any(indices >= len(self.envelopes)):
            raise ValueError("envelope index out of range")
        if np.asarray(self.active).dtype.kind != "b":
            raise ValueError("active must be boolean")
        active = owned(self.active, np.bool_, "active mask")
        shape(active, expected, "active mask")
        object.__setattr__(self, "envelope_index", indices)
        object.__setattr__(self, "active", active)


@dataclass(frozen=True)
class TimelineBatch:
    """All points of one template, in original flat point order.

    points: unique nonnegative int64 (n,) C1 point indices.
    tracks: DriveTrack tuple, one slot per drive with n per-point values.
    flux: FluxSchedule containing n ragged point spans.
    end_ns: nonnegative float64 (n,) readout start times. Engine ignores tracks
        and flux segments beyond these times, returning DC-basis densities.
    flux_waveform: optional continuous absolute flux samples in batch point
        order, with the same loop count as flux. Required by flux_dop853.
        None means no continuous samples were supplied.
    Arrays own read-only copies.

    Raises:
        ValueError: message names shape for misaligned per-point arrays,
        point indices for invalid points, or readout times/flux for bad spans.
    """

    points: NDArray[np.int64]
    tracks: tuple[DriveTrack, ...]
    flux: FluxSchedule
    end_ns: NDArray[np.float64]
    flux_waveform: FluxWaveform | None = None

    def __post_init__(self) -> None:
        points = integers(self.points, "point indices")
        if (
            points.ndim != 1
            or np.any(points < 0)
            or len(np.unique(points)) != len(points)
        ):
            raise ValueError("point indices must be a unique nonnegative vector")
        end = owned(self.end_ns, np.float64, "readout times")
        shape(end, points.shape, "readout times")
        if np.any(end < 0) or len(self.flux.offsets) != len(points) + 1:
            raise ValueError("readout times/flux must match points")
        if self.flux_waveform is not None:
            shape(
                self.flux_waveform.offsets, (len(points) + 1,), "flux waveform offsets"
            )
            if self.flux_waveform.flux.shape[1] != self.flux.level.shape[1]:
                raise ValueError("flux waveform loop shape must match schedule")
        for track in self.tracks:
            shape(track.start_ns, points.shape, "drive track")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "end_ns", end)


@dataclass(frozen=True)
class PathDiagnostics:
    """Author-only aggregate counts, not acquisition results.

    segments_by_kernel: kernel name -> nonnegative segment count.
    points_by_class: dispatch path-class name -> nonnegative point count.
    exceeded_by_kernel: kernel name -> post-apply accuracy exceedance segment
        count, with only positive counts retained. Exceeded results remain
        committed without correction or retry; these are not solver failures.
    max_trace_error: maximum absolute complex trace-minus-one of final
        readout densities; zero for an empty evolve. Report-only, with no
        correction, rejection or readout-specific tolerance comparison.
    All mappings default empty, own immutable copies and reject empty names
    or negative counts with ValueError. A point may participate in several
    path classes.
    """

    segments_by_kernel: Mapping[str, int] = field(default_factory=dict)
    points_by_class: Mapping[str, int] = field(default_factory=dict)
    exceeded_by_kernel: Mapping[str, int] = field(default_factory=dict)
    max_trace_error: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_trace_error) or self.max_trace_error < 0:
            raise ValueError("max_trace_error must be finite and nonnegative")
        for name, counts in (
            ("segments_by_kernel", self.segments_by_kernel),
            ("points_by_class", self.points_by_class),
            ("exceeded_by_kernel", self.exceeded_by_kernel),
        ):
            if any(not key or value < 0 for key, value in counts.items()):
                raise ValueError("diagnostics require names and nonnegative counts")
            copied = dict(counts)
            if name == "exceeded_by_kernel":
                copied = {key: value for key, value in copied.items() if value > 0}
            object.__setattr__(self, name, MappingProxyType(copied))


@dataclass(frozen=True)
class EvolutionResult:
    """Densities at readout start, always in the DC (level zero) eigenbasis.

    points: unique nonnegative int64 (n,), union of input batch point indices.
    densities: finite complex128 (n,M,M), including basis conversion at end_ns
        for timelines ending at another flux level. No clipping/normalization.
    paths: author-only PathDiagnostics of segments and point classes.
    Arrays own read-only copies.

    Raises:
        ValueError: message names shape, finite data or point indices.
    """

    points: NDArray[np.int64]
    densities: NDArray[np.complex128]
    paths: PathDiagnostics

    def __post_init__(self) -> None:
        points = integers(self.points, "result point indices")
        if (
            points.ndim != 1
            or np.any(points < 0)
            or len(np.unique(points)) != len(points)
        ):
            raise ValueError("result points must be a unique nonnegative vector")
        densities = owned(self.densities, np.complex128, "result densities")
        if (
            densities.ndim != 3
            or densities.shape[0] != len(points)
            or densities.shape[1] == 0
            or densities.shape[1] != densities.shape[2]
        ):
            raise ValueError("result densities must have shape (points, M, M)")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "densities", densities)


class Cancelled(Exception):
    """Evolution was cancelled between chunks; no partial result is returned."""


class Engine:
    """Segment engine with inline or threaded execution and persistent prepare LRU.

    workers: positive sole parallel width; kernels remain single-threaded.
    cache_bytes: nonnegative resident-array byte limit, not a process RAM cap.
    kernels: optional algorithm collection replacing the production registry;
        each stateless collaborator implements the internal Kernel contract and
        has a unique name. Missing required names raise, never silently reroute.
        Terminal families must cover their full dispatch domain: t4 handles all
        no-tone rows, t6 all single constant-tone rows, gate_cache repeated
        shaped rows, and period_table the remaining single shaped-tone rows.
        If a terminal guard rejects a dispatched row, evolve raises RuntimeError
        with "no valid kernel"; it does not reroute that row to DOP853.
    chunks_per_worker: positive chunk multiplier; only scheduling, not physics.

    The zero-level model identity names the entire ModelSource: all physical
    parameters and absolute DC flux. Kernel keys include row flux levels.
    Cache namespaces also include the error budget, so stricter requests cannot
    reuse looser preparations. Calls on one instance serialize; close is
    idempotent and waits for active work before clearing the cache. No worker
    resource survives an evolve call. The prepare cache persists across evolve
    calls; multi-timeline work opens and closes its own joblib threading pool.
    Empty work, a single timeline, and workers=1 run inline without a pool.
    Native BLAS uses one thread throughout evolve, coordinated across engines.
    The last overlapping evolve restores the original process-wide BLAS limits.
    """

    def __init__(
        self,
        workers: int,
        cache_bytes: int,
        *,
        kernels: Sequence[Kernel] | None = None,
        chunks_per_worker: int = 6,
    ) -> None:
        """Create the prepare cache; reject invalid sizes or duplicate kernel names."""
        from ._schedule import PrepareCache
        from .kernels import REGISTRY

        if workers < 1 or cache_bytes < 0 or chunks_per_worker < 1:
            raise ValueError(
                "workers/chunks must be positive and cache_bytes nonnegative"
            )
        collection = REGISTRY if kernels is None else kernels
        self._kernels = {kernel.name: kernel for kernel in collection}
        if len(self._kernels) != len(collection):
            raise ValueError("kernel names must be unique")
        self._cache: PrepareCache = PrepareCache(cache_bytes)
        self._chunks = workers * chunks_per_worker
        self._lock = Lock()
        self._closed = False
        self._workers = workers

    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: Event,
    ) -> EvolutionResult:
        """Evolve all batches to readout, forwarding per-segment budget to kernels.

        source: this round's model cache and initial DC density. batches: all
        template batches for the acquire. budget: positive absolute population
        and low-four error allowance. cancel: event checked between chunks;
        when set, raises Cancelled, without returning partial results.
        Model ValueErrors propagate; numerical failure raises RuntimeError
        naming the point and segment. Results do not depend on chunking or
        worker width except for floating-point rounding. Classification may
        borrow this call's worker pool for eight or more shared prepare keys.
        Local guards run in timeline workers and hand their tables to apply.
        Guard decisions remain independent of worker width and execution order.
        Post-apply accuracy
        exceedances stay in the result's paths.exceeded_by_kernel and produce
        exactly one RuntimeWarning per call, without correction or retry.
        """
        from ._dispatch import classify
        from ._planning import plan
        from ._prefix import PrefixResolver
        from ._schedule import Schedule

        with self._lock, native_limit.acquire():
            if self._closed:
                raise RuntimeError("engine is closed")
            if not np.isfinite(budget) or budget <= 0:
                raise ValueError("budget must be finite and positive")
            if cancel.is_set():
                raise Cancelled()
            if not batches or not any(len(batch.points) for batch in batches):
                size = source.initial_density.shape[0]
                return EvolutionResult(
                    np.empty(0, dtype=np.int64),
                    np.empty((0, size, size), dtype=np.complex128),
                    PathDiagnostics(),
                )
            geometry = plan(source, batches)
            workers = (
                self._workers
                if len(geometry.points) > 1 and len(geometry.table.timeline)
                else 1
            )
            with execution_pool(workers) as pool:
                geometry = PrefixResolver(
                    geometry, self._kernels, source, budget, cancel
                ).resolve(self._cache, pool, self._chunks)
                dispatch = classify(geometry, self._kernels, source, budget, pool=pool)
                schedule = Schedule(geometry, dispatch, source, budget, cancel)
                result = schedule.run(self._kernels, self._cache, pool, self._chunks)
            if cancel.is_set():
                raise Cancelled()
            exceeded = result.paths.exceeded_by_kernel
            if exceeded:
                details = ", ".join(
                    f"{name}: {count}" for name, count in sorted(exceeded.items())
                )
                warnings.warn(
                    f"segment accuracy check exceeded in {sum(exceeded.values())} segments "
                    f"({details}); results use the uncorrected approximation",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return result

    def close(self) -> None:
        """Wait for active work, clear the cache and reject later evolve calls; idempotent."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._cache.clear()
