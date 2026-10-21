"""Acquire-level segment engine and immutable ns/GHz timeline records."""

from __future__ import annotations

import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
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
from ._validation import density, integers, owned, shape
from .secular import FluxSchedule, FluxWaveform, ModelSource, SegmentShift
from .telemetry import FamilyRouting

# Replay identity of the segment engine numerical algorithms, independent of records.
SEGMENT_NUMERIC_VERSION = "segment-engine-v6"


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
    segment_boundaries: optional point-aligned tuples of strictly increasing,
        finite absolute ns in [0, end_ns]. Empty outer tuple means none.
    segment_shifts: optional point-aligned tuples of SegmentShift in increasing
        span order. Each span must exactly match a segment_spans result; shifts
        never introduce boundaries. Unlisted spans have zero shift. Empty outer
        tuple means no shifts. State order is the physical flux model basis.
    start_ns: optional finite float64 (n,) absolute ns, paired with
        initial_densities. Each value must be 0 or a final segment_spans boundary
        no later than end_ns. Planning retains the common origin; only spans in
        [start_ns,end_ns] evolve. Invalid starts raise ValueError naming point
        and value. None starts every point at zero using ModelSource's density.
    initial_densities: optional complex128 (n,M,M) DC-energy-basis densities in
        point order, paired with start_ns. Each obeys ModelSource's existing
        finite, square, Hermitian, trace-one and positive density invariants,
        without normalization or clipping. M must match the source at evolve.
        Unpaired fields or invalid shapes/invariants raise ValueError.
    Arrays own read-only copies. Invalid boundaries/spans fail at construction.

    Raises:
        ValueError: message names shape for misaligned per-point arrays,
        point indices for invalid points, or readout times/flux for bad spans.
    """

    points: NDArray[np.int64]
    tracks: tuple[DriveTrack, ...]
    flux: FluxSchedule
    end_ns: NDArray[np.float64]
    flux_waveform: FluxWaveform | None = None
    segment_boundaries: tuple[tuple[float, ...], ...] = ()
    segment_shifts: tuple[tuple[SegmentShift, ...], ...] = ()
    start_ns: NDArray[np.float64] | None = None
    initial_densities: NDArray[np.complex128] | None = None

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
        self._validate_segments()
        self._validate_continuation()

    def _validate_continuation(self) -> None:
        if (self.start_ns is None) != (self.initial_densities is None):
            raise ValueError("start_ns and initial_densities must be supplied together")
        if self.start_ns is None or self.initial_densities is None:
            return
        starts = np.array(self.start_ns, dtype=np.float64, copy=True)
        shape(starts, self.points.shape, "start_ns")
        for index, spans in enumerate(segment_spans(self)):
            value = starts[index]
            boundaries = {0.0, *(t for span in spans for t in span)}
            if (
                not np.isfinite(value)
                or value not in boundaries
                or value > self.end_ns[index]
            ):
                raise ValueError(f"point {self.points[index]} invalid start_ns {value}")
        initial = owned(self.initial_densities, np.complex128, "initial densities")
        if initial.ndim != 3 or len(initial) != len(self.points):
            raise ValueError("initial densities shape must be (n,M,M)")
        if initial.shape[1] == 0 or initial.shape[1] != initial.shape[2]:
            raise ValueError("initial densities shape must be (n,M,M)")
        for matrix in initial:
            density(matrix)
        starts.setflags(write=False)
        object.__setattr__(self, "start_ns", starts)
        object.__setattr__(self, "initial_densities", initial)

    def _validate_segments(self) -> None:
        boundaries = tuple(
            tuple(float(t) for t in times) for times in self.segment_boundaries
        )
        if boundaries and len(boundaries) != len(self.points):
            raise ValueError("segment boundaries must match points")
        for point, times in enumerate(boundaries):
            values = np.asarray(times)
            if (
                not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > self.end_ns[point])
                or np.any(np.diff(values) <= 0)
            ):
                raise ValueError(
                    "segment boundaries must increase within readout times"
                )
        shifts = tuple(tuple(point) for point in self.segment_shifts)
        if shifts and len(shifts) != len(self.points):
            raise ValueError("segment shifts must match points")
        object.__setattr__(self, "segment_boundaries", boundaries)
        object.__setattr__(self, "segment_shifts", shifts)
        if shifts:
            for supplied, spans in zip(shifts, segment_spans(self), strict=True):
                previous_end = -1.0
                for shift in supplied:
                    if (
                        shift.start_ns,
                        shift.end_ns,
                    ) not in spans or shift.start_ns < previous_end:
                        raise ValueError(
                            "shift span must match a final segment without overlap"
                        )
                    previous_end = shift.end_ns


def segment_spans(batch: TimelineBatch) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Query final half-open ns spans, in batch point order, without model/cache work.

    Spans touch without overlap and cover [0, end_ns]; zero duration gives ().
    Uses exactly evolve's flux, active microwave/plateau and neutral boundaries.
    SegmentShift values do not change this partition. No kernel is prepared and
    neither a ModelSource nor an Engine is needed. Invalid batch input raises
    ValueError at TimelineBatch construction.
    """
    from ._planning import batch_boundaries

    events, ends, _ = batch_boundaries(batch, {})
    return tuple(
        tuple(
            (float(a), float(b))
            for a, b in zip(
                events["time"][events["timeline"] == point],
                ends[events["timeline"] == point],
                strict=True,
            )
        )
        for point in range(len(batch.points))
    )


@dataclass(frozen=True)
class PathDiagnostics:
    """Author-only aggregate counts, not acquisition results.

    segments_by_kernel: kernel name -> nonnegative timeline-segment count.
        A shared prefix segment counts once per adopting timeline under the
        kernel that computed it. Its exceedances scale by the same rule.
    points_by_class: dispatch path-class name -> nonnegative point count.
    exceeded_by_kernel: kernel name -> post-apply accuracy exceedance segment
        count, with only positive counts retained. Exceeded results remain
        committed without correction or retry; these are not solver failures.
    max_trace_error: maximum absolute complex trace-minus-one of final
        readout densities; zero for an empty evolve. Report-only, with no
        correction, rejection or readout-specific tolerance comparison.
    routing: optional family work reports, empty when telemetry is disabled.
        Timings describe this call only, never the call that populated a cache.
    All mappings default empty, own immutable copies and reject empty names
    or negative counts with ValueError. A point may participate in several
    path classes.
    """

    segments_by_kernel: Mapping[str, int] = field(default_factory=dict)
    points_by_class: Mapping[str, int] = field(default_factory=dict)
    exceeded_by_kernel: Mapping[str, int] = field(default_factory=dict)
    max_trace_error: float = 0.0
    routing: Mapping[str, FamilyRouting] = field(default_factory=dict)
    frontend_cpu_s: float = 0.0
    """Calling-thread geometry and WorkPlan CPU; zero when telemetry is off."""
    point_members: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    """Class -> increasing unique nonnegative TimelineBatch.points indices.

    Keys equal points_by_class keys and each tuple's length equals its count.
    Engine fills caller point IDs, not internal timeline offsets. The mapping
    and member tuples own immutable copies. Invalid membership raises ValueError.
    """

    def for_reuse(self, *, telemetry: bool) -> PathDiagnostics:
        """Preserve routes but report no kernel calls for an exact-result hit.

        Enabled reports count each logical segment as a cache hit, with zero
        CPU, rejections and estimated work. Disabled reports have no routing.
        """
        return replace(
            self,
            frontend_cpu_s=0.0,
            routing={
                name: FamilyRouting(rows=count, cache_hits=count, estimated_cpu_s=0.0)
                for name, count in self.segments_by_kernel.items()
            }
            if telemetry
            else {},
        )

    def warn_exceeded(self, *, stacklevel: int = 2) -> None:
        """Warn once for all report-only accuracy exceedances, if any.

        Counts and results remain unchanged. stacklevel selects the warning's
        caller location using the standard warnings.warn convention.
        """
        exceeded = self.exceeded_by_kernel
        if exceeded:
            details = ", ".join(
                f"{name}: {count}" for name, count in sorted(exceeded.items())
            )
            warnings.warn(
                f"segment accuracy check exceeded in {sum(exceeded.values())} segments "
                f"({details}); results use the uncorrected approximation",
                RuntimeWarning,
                stacklevel=stacklevel,
            )

    def __post_init__(self) -> None:
        object.__setattr__(self, "routing", MappingProxyType(dict(self.routing)))
        members = {key: tuple(values) for key, values in self.point_members.items()}
        if members.keys() != self.points_by_class.keys():
            raise ValueError("point membership keys must match points_by_class")
        for name, values in members.items():
            if (
                len(values) != self.points_by_class[name]
                or any(
                    not np.issubdtype(type(value), np.integer) or value < 0
                    for value in values
                )
                or any(a >= b for a, b in zip(values[:-1], values[1:], strict=True))
            ):
                raise ValueError(
                    "point members must be increasing unique indices matching counts"
                )
        object.__setattr__(self, "point_members", MappingProxyType(members))
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


def merge_path_diagnostics(parts: Sequence[PathDiagnostics]) -> PathDiagnostics:
    """Merge author-only evolve/span diagnostics without double-counting points.

    parts is a sequence of validated PathDiagnostics. Per-class point_members
    are unioned and sorted; points_by_class is the resulting distinct count.
    segments_by_kernel, exceeded_by_kernel and frontend_cpu_s are summed.
    max_trace_error is the maximum, or zero for empty parts. Routing sums each
    family's rows, CPU times, hits, rejections, costs, builds and admission rows;
    estimated_cpu_s is None if any contributing family estimate is unknown.
    Reuse-hit diagnostics participate exactly as returned, without adding hits
    or replacing their counts. Empty parts returns empty PathDiagnostics.
    Inputs are unchanged; the result owns immutable mappings and member tuples.
    """
    segments: Counter[str] = Counter()
    exceeded: Counter[str] = Counter()
    members: dict[str, set[int]] = {}
    routing: dict[str, list[FamilyRouting]] = {}
    for part in parts:
        segments.update(part.segments_by_kernel)
        exceeded.update(part.exceeded_by_kernel)
        for name, indices in part.point_members.items():
            members.setdefault(name, set()).update(indices)
        for name, report in part.routing.items():
            routing.setdefault(name, []).append(report)
    return PathDiagnostics(
        segments,
        {name: len(indices) for name, indices in members.items()},
        exceeded,
        max((part.max_trace_error for part in parts), default=0.0),
        {name: _merge_family(reports) for name, reports in routing.items()},
        sum(part.frontend_cpu_s for part in parts),
        {name: tuple(sorted(indices)) for name, indices in members.items()},
    )


def _merge_family(parts: Sequence[FamilyRouting]) -> FamilyRouting:
    return FamilyRouting(
        rows=sum(part.rows for part in parts),
        prepare_cpu_s=sum(part.prepare_cpu_s for part in parts),
        apply_cpu_s=sum(part.apply_cpu_s for part in parts),
        cache_hits=sum(part.cache_hits for part in parts),
        rejected_rows=sum(part.rejected_rows for part in parts),
        estimated_cost=sum(part.estimated_cost for part in parts),
        estimated_cpu_s=(
            sum(
                part.estimated_cpu_s
                for part in parts
                if part.estimated_cpu_s is not None
            )
            if all(part.estimated_cpu_s is not None for part in parts)
            else None
        ),
        prepare_builds=sum(part.prepare_builds for part in parts),
        admission_rows=sum(part.admission_rows for part in parts),
    )


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
    cache_bytes: nonnegative unified resident/pending array and estimated metadata
        byte limit (4608 metadata bytes per whole allocation), not a
        process RAM cap. Active nonretained products and scratch are separate.
        Exact final results use a separate LRU capped at min(64 MiB, cache_bytes),
        including metadata. Zero disables both caches. No ADC/sampling is cached.
    kernels: optional algorithm collection replacing the production registry;
        each stateless collaborator implements the internal Kernel contract and
        has a unique name. Missing required names raise, never silently reroute.
        Terminal families must cover their full dispatch domain: t4 handles all
        no-tone rows and t6 all single constant-tone rows. If either terminal
        guard rejects a dispatched row, evolve raises RuntimeError with
        "no valid kernel". Shaped rows try gate_cache, period_table and dop853
        in amortized-cost order; a guard rejection tries the next candidate.
        Preparation and apply failures propagate without reclassification.
    chunks_per_worker: optional positive fixed chunk multiplier. None selects
        timeline batching from the complete acquire inventory and keeps six
        prepare chunks per worker. Explicit values fix both stages. Scheduling
        never changes numerical admission or physics.

    The zero-level model identity names the entire ModelSource: all physical
    parameters and absolute DC flux. Kernel keys include row flux levels.
    Cache namespaces also include the error budget, so stricter requests cannot
    reuse looser preparations. Calls on one instance serialize; close is
    idempotent and waits for active work before clearing the cache. No worker
    resource survives an evolve call. The prepare cache persists across evolve
    calls; multi-timeline work opens and closes its own joblib threading pool.
    Exact-input memo hits skip geometry only for inventories without cumulative
    single-tone shaped gate or multitone candidates: their routes do not depend
    on history.
    Other inventories count distinct acquire content's complete row inventories
    and select gate keys/multitone routes before memo lookup. Caller-owned
    preparations can use reuse() to reprice retained immutable facts without
    entering geometry, WorkPlan construction or numerical preparation.
    Acquires sharing a source must serialize, including across Engine instances.
    Empty work, a single timeline, and workers=1 run inline without a pool.
    Native BLAS uses one thread throughout evolve, coordinated across engines.
    The last overlapping evolve restores the original process-wide BLAS limits.
    Kernels must be deterministic and pure for identical numerical inputs, both
    for prepare reuse and exact-result memoization. The kernel collection and
    numerical implementation are fixed for this Engine's process-local lifetime;
    memo entries are never shared across Engines or persisted. Exact-input hits
    bypass planning and all kernel calls, returning independent owned copies.
    """

    def __init__(
        self,
        workers: int,
        cache_bytes: int,
        *,
        kernels: Sequence[Kernel] | None = None,
        chunks_per_worker: int | None = None,
    ) -> None:
        """Create the prepare cache; reject invalid sizes or duplicate kernel names."""
        from ._result_memo import RESULT_MEMO_BYTES, ResultMemo
        from ._routing_reuse import RoutingReuse
        from .cache import PrepareCache
        from .kernels import REGISTRY

        if (
            workers < 1
            or cache_bytes < 0
            or (chunks_per_worker is not None and chunks_per_worker < 1)
        ):
            raise ValueError(
                "workers/chunks must be positive and cache_bytes nonnegative"
            )
        collection = REGISTRY if kernels is None else kernels
        self._kernels = {kernel.name: kernel for kernel in collection}
        if len(self._kernels) != len(collection):
            raise ValueError("kernel names must be unique")
        self._cache: PrepareCache = PrepareCache(cache_bytes)
        self._memo = ResultMemo(min(RESULT_MEMO_BYTES, cache_bytes))
        self._memo_enabled = cache_bytes > 0
        self._routing_reuse = RoutingReuse(min(4 * 1024**2, cache_bytes))
        self._chunks = workers * (6 if chunks_per_worker is None else chunks_per_worker)
        self._inventory_chunks = chunks_per_worker is None
        self._lock = Lock()
        self._closed = False
        self._workers = workers

    def reuse(
        self,
        source: ModelSource,
        key: bytes,
        *,
        cancel: Event | None = None,
        budget: float = 1e-3,
    ) -> bool:
        """Validate a caller's exact-content result against current source usage.

        key is the same caller-owned content digest passed as reuse_key to the
        successful evolve that produced the result. The caller must invalidate
        its result on any content, source, budget or Engine change. True records
        this acquire's complete inventory and permits reuse with no numerical
        work. False records nothing: evolve the full acquire instead. Eviction,
        a different source, disabled residency and a changed route return False.
        Calls sharing a source must serialize the check and subsequent evolve.
        Optional cancel raises Cancelled when set; None has no cancellation
        token. A closed engine raises RuntimeError.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("engine is closed")
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            return self._routing_reuse.reuse(key, source, budget)

    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: Event,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
    ) -> EvolutionResult:
        """Evolve all batches to readout, forwarding per-segment budget to kernels.

        source: this round's model cache and initial DC density. batches: all
        template batches for the acquire. reuse_key: optional bytes digest of
        every input used by a caller-retained result, including any downstream
        inputs not present in batches. None skips registration for reuse(). A
        successful nonempty evolve registers inventory/pricing facts under this
        key, scoped to this Engine, source instance and budget. The caller must
        invalidate its retained result on any content, source, budget or Engine
        change; evolve does not verify the caller's digest. Registration does not
        change the numerical solve or its result. Facts use a separate LRU capped
        at min(4 MiB, cache_bytes). Empty work, disabled residency or oversized
        facts may leave no entry. A later reuse() returns False after eviction or
        a route change; the caller must then evolve the full acquire again.
        budget: finite positive absolute population and low-four error allowance;
        otherwise raises ValueError. A closed engine raises RuntimeError.
        cancel: event checked between chunks;
        when set, raises Cancelled, without returning partial results.
        telemetry enables per-family calling-thread CPU and routing accounting
        without I/O; False avoids CPU clock reads and returns empty routing.
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
        from ._dispatch import classify, create_work_plan, resident_inventory
        from ._planning import plan
        from ._prefix import PrefixResolver
        from ._resident import ResidentAdmission
        from ._result_memo import input_key, route_key
        from ._schedule import Schedule
        from .kernel import KernelContext

        with self._lock, native_limit.acquire():
            self._check_evolve_ready(budget, cancel)
            if not batches or not any(len(batch.points) for batch in batches):
                size = source.initial_density.shape[0]
                return EvolutionResult(
                    np.empty(0, dtype=np.int64),
                    np.empty((0, size, size), dtype=np.complex128),
                    PathDiagnostics(),
                )
            content_key = input_key(source, batches, budget)
            key = content_key if self._memo_enabled else None
            if key is not None:
                # Only inventories without cumulative gate candidates use the
                # raw input key. Their routes are independent of source history.
                cached = self._cached_result(key, cancel, telemetry=telemetry)
                if cached is not None:
                    self._routing_reuse.remember_independent(
                        reuse_key, source, budget, content_key
                    )
                    return cached
            frontend_started = time.thread_time() if telemetry else 0.0
            geometry = plan(source, batches)
            workers = (
                self._workers
                if len(geometry.points) > 1 and len(geometry.table.timeline)
                else 1
            )
            # Drain workers before releasing evolve-local ownership.
            with closing(
                create_work_plan(
                    geometry.table,
                    source,
                    budget,
                    telemetry=telemetry,
                    content_key=content_key,
                )
            ) as work_plan:
                frontend_cpu_s = (
                    time.thread_time() - frontend_started if telemetry else 0.0
                )
                context = KernelContext(source, budget, work_plan)
                reuse_snapshot = self._routing_reuse.snapshot(
                    reuse_key, geometry.table, context
                )
                if key is not None:
                    key = route_key(key, geometry.table, context)
                    cached = self._cached_result(key, cancel, telemetry=telemetry)
                    if cached is not None:
                        self._routing_reuse.add(reuse_key, reuse_snapshot)
                        return cached
                kernels = self._kernels
                inventory = resident_inventory(geometry, kernels, context, self._cache)
                with (
                    closing(ResidentAdmission(self._cache, inventory)) as resident,
                    execution_pool(workers) as pool,
                ):
                    geometry = PrefixResolver(
                        geometry, kernels, context, cancel, resident
                    ).resolve(self._cache, pool, self._chunks)
                    dispatch = classify(
                        geometry,
                        kernels,
                        context,
                        pool=pool,
                        admission=resident,
                    )
                    schedule = Schedule(geometry, dispatch, context, cancel, resident)
                    result = schedule.run(
                        kernels,
                        self._cache,
                        pool,
                        self._chunks,
                        timeline_workers=workers if self._inventory_chunks else None,
                    )
            if telemetry:
                result = replace(
                    result, paths=replace(result.paths, frontend_cpu_s=frontend_cpu_s)
                )
            if cancel.is_set():
                raise Cancelled()
            result.paths.warn_exceeded(stacklevel=3)
            if key is not None:
                self._memo.add(key, result)
            self._routing_reuse.add(reuse_key, reuse_snapshot)
            return result

    def _check_evolve_ready(self, budget: float, cancel: Event) -> None:
        """Check evolve preconditions without side effects under caller-held resources.

        The caller holds the engine lock and native limit; this acquires neither.
        Raise for a closed engine, invalid budget, then cancellation, in that order.
        """
        if self._closed:
            raise RuntimeError("engine is closed")
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("budget must be finite and positive")
        if cancel.is_set():
            raise Cancelled()

    def _cached_result(
        self, key: bytes, cancel: Event, *, telemetry: bool
    ) -> EvolutionResult | None:
        """Look up an exact result and preserve cancellation/warning semantics."""
        cached = self._memo.find(key, telemetry=telemetry)
        if cached is not None:
            if cancel.is_set():
                raise Cancelled()
            cached.paths.warn_exceeded(stacklevel=3)
        return cached

    def close(self) -> None:
        """Wait for active work, clear the cache and reject later evolve calls; idempotent."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._cache.clear()
                self._memo.clear()
                self._routing_reuse.clear()
