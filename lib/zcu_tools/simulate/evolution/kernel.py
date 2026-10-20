"""Internal prepare/apply seam shared by algorithm families and scheduling.

This is deliberately not re-exported by the evolution package. Kernels own
opaque prepared payloads; the engine only sees their resident array byte size.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Protocol, SupportsIndex

import numpy as np
from numpy.typing import NDArray

from ._validation import integers, owned, shape
from .engine import EnvelopeSpec
from .secular import FluxWaveform, ModelSource
from .workplan import WorkPlan


@dataclass(frozen=True)
class SegmentTable:
    """Structure-of-arrays segments and evolve-local certified prefix inputs.

    timeline: nonnegative int64 (rows,), batch-local timeline index.
    start_ns, duration_ns: nonnegative float64 (rows,), segment support.
    level: int64 (rows,loops), constant quantized flux level.
    tones: int64 (rows,), number of playing tones, from 0 to max_tones.
    tone_port: int64 (rows,max_tones), index into the model's drives, -1 pad.
    tone_frequency_ghz: float64 (rows,max_tones), carrier cycles/ns.
    tone_amplitude: float64 (rows,max_tones), signed real port-scaled gain.
    tone_theta_rad: float64 (rows,max_tones), effective phase at segment start
        2*pi*f*start_ns + common-origin phase, including virtual Z.
    tone_envelope: int64 (rows,max_tones), index into envelopes; -1 pad.
    tone_envelope_offset_ns: float64 (rows,max_tones), envelope-local time
        at segment start (a segment may cut an existing pulse).
    envelopes: shared EnvelopeSpec tuple. Only slots before tones[row] play;
        floating padding is finite and ignored.
    flux_waveform: optional FluxWaveform indexed by batch-local timeline, not
        row. Its offsets are (timelines+1,), samples (total,loops), with the
        same loop count as level. Linear interpolation holds endpoint samples.
        None means all rows lack continuous samples. In combined batches a
        timeline without samples has a single zero-time placeholder; only
        quantized kernels may consume such a timeline.
    flux_waveform_available: optional bool (timelines,) mask. False marks
        placeholder timelines without continuous samples. None means all
        waveform timelines have samples (or no waveform is supplied).
    prefix_group: optional immutable tuple (rows,) of nonnegative group IDs
        or None per row. Equal IDs certify identical nonempty prefixes within
        one source/evolve, including timing, levels, tones and flux samples.
        Only the planner assigns groups; kernels must not infer them from level.
    prefix_no_drive: group-aligned tuple. True preserves no-drive B2 exemption.
    prefix_inputs: immutable group-ID to read-only DC density mapping. The
        engine computes these known inputs before dependent rows are classified.
        Workers adopt the latest saved input and skip its preceding rows.
        These densities never enter model-only preparations or PrepareCache.
        Absent groups have no known input yet.
    Arrays own read-only copies. Invalid shapes/counts/indices raise ValueError.
    Adjacent rows of one timeline may be merged by DOP853 kernels into a
    multi-interval work item: stop at boundaries and reset each row's step seed.

    Raises:
        ValueError: message names shape for misaligned arrays, tone counts
        for excess tones, or nonnegative for invalid segment support.
    """

    timeline: NDArray[np.int64]
    start_ns: NDArray[np.float64]
    duration_ns: NDArray[np.float64]
    level: NDArray[np.int64]
    tones: NDArray[np.int64]
    tone_port: NDArray[np.int64]
    tone_frequency_ghz: NDArray[np.float64]
    tone_amplitude: NDArray[np.float64]
    tone_theta_rad: NDArray[np.float64]
    tone_envelope: NDArray[np.int64]
    tone_envelope_offset_ns: NDArray[np.float64]
    envelopes: tuple[EnvelopeSpec, ...]
    flux_waveform: FluxWaveform | None = None
    flux_waveform_available: NDArray[np.bool_] | None = None
    prefix_group: tuple[int | None, ...] | None = None
    prefix_no_drive: tuple[bool, ...] = ()
    prefix_inputs: Mapping[int, NDArray[np.complex128]] = field(default_factory=dict)
    shift_content: NDArray[np.bytes_] = field(
        default_factory=lambda: np.empty(0, dtype="S64")
    )

    def __post_init__(self) -> None:
        timeline = integers(self.timeline, "segment timelines")
        if timeline.ndim != 1 or np.any(timeline < 0):
            raise ValueError("segment timelines must be a nonnegative vector")
        rows = len(timeline)
        object.__setattr__(self, "timeline", timeline)
        for name in ("start_ns", "duration_ns"):
            array = owned(getattr(self, name), np.float64, name)
            shape(array, (rows,), name)
            if np.any(array < 0):
                raise ValueError("segment support must be nonnegative")
            object.__setattr__(self, name, array)
        levels = integers(self.level, "segment levels")
        if levels.ndim != 2 or levels.shape[0] != rows or levels.shape[1] == 0:
            raise ValueError("segment levels must have shape (rows, loops)")
        object.__setattr__(self, "level", levels)
        if self.flux_waveform is not None:
            if self.flux_waveform.flux.shape[1] != levels.shape[1]:
                raise ValueError("flux waveform loop shape must match segment levels")
            if rows and timeline.max() >= len(self.flux_waveform.offsets) - 1:
                raise ValueError("flux waveform must cover segment timelines")
        if self.flux_waveform_available is not None:
            if self.flux_waveform is None:
                raise ValueError("waveform availability requires a flux waveform")
            mask = owned(
                self.flux_waveform_available, np.bool_, "waveform availability"
            )
            shape(mask, (len(self.flux_waveform.offsets) - 1,), "waveform availability")
            object.__setattr__(self, "flux_waveform_available", mask)
        shifts = (
            np.zeros(rows, dtype="S64")
            if not len(self.shift_content)
            else np.array(self.shift_content, dtype="S64", copy=True)
        )
        shape(shifts, (rows,), "segment shift content")
        shifts.setflags(write=False)
        object.__setattr__(self, "shift_content", shifts)
        self._validate_prefixes(rows)
        self._validate_tones(rows)

    def model(self, source: ModelSource, row: SupportsIndex):
        """Resolve a row by physical flux and its independent shift identity."""
        return source.shifted_model(
            tuple(int(v) for v in self.level[int(row)]), self.shift(row)
        )

    def shift(self, row: SupportsIndex) -> bytes:
        """Return the normalized model-only shift fingerprint."""
        return bytes(self.shift_content[int(row)])

    def constant_envelopes(self, rows: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Return row-aligned bools: every active envelope has kind constant.

        rows contains valid global row indices, in any order or subset, including
        empty selections. Only slots before tones[row] are active. Padding never
        affects the result; a zero-tone row is constant. This table-only rule
        does not validate numerical applicability or inspect envelope samples.
        """
        kinds = np.array(
            [envelope.kind == "constant" for envelope in self.envelopes], dtype=bool
        )
        constant = np.ones(len(rows), dtype=bool)
        for slot in range(self.tone_port.shape[1]):
            playing = self.tones[rows] > slot
            constant[playing] &= kinds[self.tone_envelope[rows[playing], slot]]
        return constant

    def _validate_prefixes(self, rows: int) -> None:
        if self.prefix_group is not None:
            groups = tuple(self.prefix_group)
            if len(groups) != rows or any(
                g is not None and not 0 <= g < len(self.prefix_no_drive) for g in groups
            ):
                raise ValueError(
                    "prefix groups must align with rows and name a valid group"
                )
            object.__setattr__(self, "prefix_group", groups)
        object.__setattr__(self, "prefix_no_drive", tuple(self.prefix_no_drive))
        inputs = {}
        for group, initial in self.prefix_inputs.items():
            if group < 0 or group >= len(self.prefix_no_drive):
                raise ValueError("prefix input requires a valid group")
            copied = owned(initial, np.complex128, "prefix input")
            if copied.ndim != 2 or copied.shape[0] != copied.shape[1]:
                raise ValueError("prefix input must be a square density")
            inputs[group] = copied
        object.__setattr__(self, "prefix_inputs", MappingProxyType(inputs))

    def _validate_tones(self, rows: int) -> None:
        tones = integers(self.tones, "tone counts")
        shape(tones, (rows,), "tone counts")
        ports = integers(self.tone_port, "tone ports")
        if ports.ndim != 2 or ports.shape[0] != rows:
            raise ValueError("tone ports must have shape (rows, max_tones)")
        width = ports.shape[1]
        if np.any(tones < 0) or np.any(tones > width):
            raise ValueError("tone counts exceed table width")
        envelopes = integers(self.tone_envelope, "tone envelopes")
        shape(envelopes, ports.shape, "tone envelopes")
        playing = np.arange(width)[None, :] < tones[:, None]
        if np.any(ports[playing] < 0) or np.any(ports[~playing] != -1):
            raise ValueError(
                "tone ports require nonnegative playing indices and -1 padding"
            )
        if (
            np.any(envelopes[playing] < 0)
            or np.any(envelopes[playing] >= len(self.envelopes))
            or np.any(envelopes[~playing] != -1)
        ):
            raise ValueError("tone envelope index out of range or invalid padding")
        for name in (
            "tone_frequency_ghz",
            "tone_amplitude",
            "tone_theta_rad",
            "tone_envelope_offset_ns",
        ):
            array = owned(getattr(self, name), np.float64, name)
            shape(array, ports.shape, name)
            object.__setattr__(self, name, array)
        if np.any(self.tone_envelope_offset_ns[playing] < 0):
            raise ValueError("envelope-local times must be nonnegative")
        object.__setattr__(self, "tones", tones)
        object.__setattr__(self, "tone_port", ports)
        object.__setattr__(self, "tone_envelope", envelopes)


def prepared_key_indices(
    available: NDArray[np.void], required: NDArray[np.void]
) -> dict[bytes, int] | None:
    """Index opaque prepared keys by bytes if they cover every required key.

    Return None for a domain miss. Duplicate available keys use the last index.
    """
    indices = {key.tobytes(): i for i, key in enumerate(available)}
    if any(key.tobytes() not in indices for key in required):
        return None
    return indices


class Prepared(Protocol):
    """Opaque kernel payload, shared read-only by workers.

    prepare and guard_prepared return state-independent data eligible for
    PrepareCache under the ordinary shared-key rules. bind_input
    may wrap it in an input-bound payload used only within one evolve call;
    that wrapper never enters PrepareCache.
    Each kernel defines its own concrete payload. nbytes counts resident array
    bytes owned by the whole batch of prepared entries, not Python overhead,
    models, solver workspaces or process RSS. It must be a nonnegative integer
    and must not change while the payload resides in the engine's LRU cache.
    """

    @property
    def nbytes(self) -> int:
        """Resident array bytes charged to the prepare cache for this payload."""
        ...


class RowPlan(Protocol):
    """Read-only per-row decisions dispatch made for one kernel family in one evolve.

    Indexed by SegmentTable row. Only the kernel family that produced it reads
    it. It lives for one evolve and never enters PrepareCache or kernel state.
    """


@dataclass(frozen=True)
class KernelContext:
    """Model source, allowance and shared ownership for one evolve call.

    source owns immutable models and the source input, not a worker density.
    budget is the finite positive absolute per-segment allowance. work_plan
    belongs to this evolve and is shared by prefix and timeline workers.
    Standalone callers supply a fresh WorkPlan and close it after use.
    """

    source: ModelSource
    budget: float
    work_plan: WorkPlan

    def __post_init__(self) -> None:
        if not np.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("kernel budget must be finite and positive")


@dataclass(frozen=True)
class ApplyContext:
    """Evolve-level inputs that every apply call of one kernel family shares.

    budget: positive absolute per-segment allowance.
    work_plan: shared evolve ownership. The default is only for standalone
        calls; Engine always passes the same WorkPlan to every apply.
    plan: this family's RowPlan from dispatch, or None when dispatch made none.
        Kernels that never receive a plan raise ValueError for a non-None plan;
        kernels that need one raise ValueError for None.
    """

    budget: float
    plan: RowPlan | None = None
    work_plan: WorkPlan = field(default_factory=WorkPlan)

    def __post_init__(self) -> None:
        if not np.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("apply budget must be finite and positive")

    def require_no_plan(self) -> None:
        """Raise ValueError if a plan is supplied to a family that accepts none."""
        if self.plan is not None:
            raise ValueError("this kernel does not accept a row plan")


@dataclass(frozen=True)
class GuardPreparation:
    """Model-only preparation returned by a guard for this evolve's source/budget.

    keys: read-only unique fixed-width prepare keys, in the payload's entry order.
    prepared: opaque read-only payload equivalent to prepare(keys, ...), with
        no density access. Keys may include rejected rows. Only accepted rows
        receive apply bindings; fully rejected model-only domains may enter
        low-priority resident storage for fresh guard admission on later calls.
        No kernel may retain this record in its own state.
    Owns a read-only key copy; a non-vector/non-void key array raises ValueError.
    """

    keys: NDArray[np.void]
    prepared: Prepared

    @cached_property
    def nbytes(self) -> int:
        """Array charge read once for transfer into whole-payload ownership."""
        return self.prepared.nbytes

    def __post_init__(self) -> None:
        keys = self.keys.copy()
        if keys.ndim != 1 or keys.dtype.kind != "V":
            raise ValueError("guard preparation keys must be a fixed-width vector")
        keys.setflags(write=False)
        object.__setattr__(self, "keys", keys)


@dataclass(frozen=True)
class GuardResult:
    """Row-local guard decision plus optional model-only preparations.

    accepted: read-only bool (len(rows),) mask identical to guard for the same arguments.
    preparations: disjoint key batches constructed with that call's source and
        budget. Empty means no handoff. Accepted products use ordinary resident
        admission; fully rejected products can only fill unused capacity and
        yield to ordinary products. The accepted mask never enters the cache.
    plan: optional kernel-defined per-row numerical decisions for this call's
        accepted rows. Dispatch merges them into the family plan for apply.
        They live only within this evolve and never enter PrepareCache.
    Owns a read-only mask copy; a non-vector/non-bool mask raises ValueError.
    """

    accepted: NDArray[np.bool_]
    preparations: tuple[GuardPreparation, ...] = ()
    plan: RowPlan | None = None

    def __post_init__(self) -> None:
        if self.accepted.ndim != 1 or self.accepted.dtype != np.bool_:
            raise ValueError("guard acceptance must be a bool vector")
        object.__setattr__(
            self, "accepted", owned(self.accepted, np.bool_, "guard acceptance")
        )


class Kernel(Protocol):
    """One stateless algorithm family; all persistent data lives in Prepared.

    name: stable algorithm name for diagnostics and cache partitioning.
    guard_builds: True when guard_prepared builds and hands off validated model
        payloads. False families return None and use ordinary batched guards.
        The engine uses this declaration to defer local table builds; the
        dispatch planner can account for guard construction without probing.
        True families return GuardResult, including for rejected/empty rows.
    Multiple numerical families share this seam so the scheduler need not know
    their representation. All index arrays below are int64 vectors.
    Engine may call guards concurrently on its existing threading pool. Guards
    must be pure with respect to model/table inputs, with decisions independent
    of worker width and execution order. Kernels never open a numerical pool.
    """

    name: str
    guard_builds: bool

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Pure validity mask (len(rows),) at positive per-segment error budget.

        Each row's mask depends only on that row's table data, source and
        budget, never on other rows in the call. Splitting rows into arbitrary
        segments, calling guard separately and concatenating results in the
        original row order must match a whole call bit for bit, regardless of
        call order.

        Known input means the source initial density or the engine-computed
        density for this row's certified prefix group. Guard may read that
        immutable input, but never evolve density or inspect worker states.
        The engine owns branch order and cross-family cost comparisons. Family
        wrappers may also require immutable complete-acquire key counts, as
        multitone_gate does. Prefix and worker calls borrow those same counts;
        subset reuse and resident cache warmth never decide admission.
        """
        ...

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> GuardResult | None:
        """Optionally return guard decisions and the preparations they required.

        Classification calls this only when guard_builds is True, instead of
        guard. It must then return GuardResult, including for empty or rejected
        rows; None raises RuntimeError rather than retrying guard. When
        guard_builds is False, classification calls guard directly and does not
        invoke this hook. Acceptance must obey guard's row locality, including
        empty/reversed subsets. Only known inputs may be read; no density is
        evolved or mutated. Returned preparations remain model-only.

        The engine reuses returned preparations only for accepted rows with
        the same kernel, prepare key, source and budget in this evolve. They
        reach A-stage before input binding and are also reused for singleton
        rows without cache admission. Shared-key batches follow normal LRU
        rules; otherwise references expire with this evolve. No kernel state
        or separate cross-evolve store is permitted. Failure semantics match
        guard/prepare, including RuntimeError on numerical failure.
        """
        ...

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> GuardResult | None:
        """Admit current rows using an existing immutable model-only payload.

        Return None if the payload's domain does not cover every required row
        product, so dispatch uses guard_prepared unchanged. Otherwise return
        current row admissions and RowPlan, without building frames, periods or
        certificates. Rejection is an accepted=False decision, not a domain miss.
        Existing payloads never replace row-local numerical admission.
        """
        ...

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Fixed-width keys (len(rows),), equal keys sharing one preparation.

        context shares immutable workload facts with guard and preparation.
        Keys include the model identity and exclude apply-only parameters;
        the exact algorithm determines which parameters belong in each half.
        Resident inventory calls this before guard on structural candidates.
        Numerically ineligible rows may use inventory-only identities disjoint
        from preparable keys. Only guard-accepted rows reach prepare and apply.
        """
        ...

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return nonnegative finite relative row costs for chunk interleaving."""
        ...

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build data for unique keys in their given order; never read density.

        rows: one representative row of table per key, aligned with keys;
        prepare_keys(table, rows) equals keys. Content a fixed-width key cannot
        carry (e.g. envelope samples) is read from these rows; the result must
        depend only on the keys, so any representative gives the same payload.
        budget is the positive absolute per-segment error allowance.
        Numerical failure raises RuntimeError. Prepared data is immutable to
        apply workers and reports its resident array bytes through nbytes.
        """
        ...

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> Prepared | None:
        """Optionally contract a common input into an evolve-local read-only payload.

        Called in A-stage only for shared prepared batches (repeated keys).
        key_index selects one entry in prepared. rows share that prepare key,
        first tone port and input density, in arbitrary order. They either all
        start at zero or belong to one certified prefix_group.
        source_initial is complex128 (M,M) in these rows' model basis: either
        the source initial density or the engine-computed prefix output,
        never a worker's evolving density. No input may be modified. budget is
        the positive absolute per-segment allowance.

        None declines sharing and keeps the ordinary per-batch apply path.
        A returned payload replaces prepared in apply for these rows, with
        key_index all zero. It must be immutable to workers and usable with
        arbitrary subsets/order (including empty rows) from this group. It
        lives only for this evolve, never in PrepareCache or kernel state.
        Numerical failure raises RuntimeError.
        """
        ...

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Advance complex128 states (len(rows),M,M) in place through rows.

        prepared is either model-only prepare data or a bind_input payload;
        the latter requires the same input density supplied to bind_input.
        key_index maps each row to its prepared entry. context supplies the
        per-segment allowance and this family's optional dispatch plan. Called by a joblib worker thread;
        numba implementations must be nogil and single-threaded. Prepared
        data and table stay unchanged. Return a bool mask (len(rows),), aligned
        with rows: True marks a post-apply accuracy threshold exceedance, not
        a failure. Results are still committed without correction or retry.
        Kernels without a post-apply check return all False. Masks, like states,
        retain row order when independent rows are split across calls.
        Numerical failure raises RuntimeError.
        """
        ...
