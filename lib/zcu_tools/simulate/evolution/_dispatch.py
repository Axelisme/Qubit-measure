"""Cross-family order and costs; kernels own numerical guards and family-local reuse."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from joblib import delayed
from numpy.typing import NDArray

from ._carrier_plan import populate_carrier_plan
from ._dop853_dispatch import direct_plans
from ._gain_plan import plan_gain_groups
from ._parallel import Executor
from ._planning import Geometry
from ._prepare_keys import group_keys
from ._resident import ResidentAdmission
from ._shaped_dispatch import (
    populate_single_gate_plan,
    preference,
    single_gate_preferred,
)
from ._two_tone_dispatch import direct_preferred
from .cache import RESIDENT_METADATA_BYTES, PrepareCache, PreparedPayload
from .engine import PathDiagnostics
from .kernel import (
    GuardPreparation,
    Kernel,
    KernelContext,
    Prepared,
    RowPlan,
    SegmentTable,
)
from .kernels.floquet import FloquetKernel, FloquetPlan
from .kernels.gate_cache import GateCacheKernel
from .kernels.multitone_gate import populate_gate_inventory
from .secular import ModelSource
from .workplan import WorkPlan


def create_work_plan(
    table: SegmentTable, source: ModelSource, budget: float = 1e-3
) -> WorkPlan:
    """Create ownership for the complete evolve before prefix/worker planning.

    table is the full immutable geometry and source owns model identity.
    Algorithm-specific group planners populate this plan before workers start;
    budget is the numerical admission budget for cold mixed gate plans.
    """
    plan = WorkPlan()
    populate_carrier_plan(table, source, plan)
    plan_gain_groups(table, source, plan)
    populate_gate_inventory(table, source, plan)
    populate_single_gate_plan(table, source, plan, budget)
    return plan


@dataclass(frozen=True)
class _CandidateDomain:
    mask: NDArray[np.bool_]
    families: tuple[str, ...]


def _candidate_domains(geometry: Geometry) -> dict[str, _CandidateDomain]:
    """Dispatch's structural domains and candidate inventory, before cost/guards.

    Inventory visits every family. Classification uses these same domains and
    reorders their candidates by acquire-wide cost or skips optional families.
    """
    table = geometry.table
    flux = geometry.flux_drive & (table.tones > 0)
    ordinary = ~flux
    constant = table.constant_envelopes(np.arange(len(table.timeline), dtype=np.int64))
    single = ordinary & (table.tones == 1)
    return {
        "idle": _CandidateDomain(table.tones == 0, ("t4",)),
        "flux": _CandidateDomain(flux, ("flux_dop853",)),
        "plateau": _CandidateDomain(
            single & constant, ("dyson", "single_tone_rwa", "floquet_period", "t6")
        ),
        "shaped": _CandidateDomain(
            single & ~constant,
            (
                "single_tone_gate",
                "shaped_sambe",
                "gate_cache",
                "period_table",
                "dop853",
            ),
        ),
        "two": _CandidateDomain(
            ordinary & (table.tones == 2) & constant,
            ("multitone_rwa", "floquet", "dop853"),
        ),
        "multiple": _CandidateDomain(
            ordinary & (((table.tones == 2) & ~constant) | (table.tones >= 3)),
            ("multitone_gate", "multitone_rwa", "dop853"),
        ),
    }


def resident_inventory(
    geometry: Geometry,
    kernels: Mapping[str, Kernel],
    context: KernelContext,
    cache: PrepareCache,
) -> dict[tuple[str, str, float], set[bytes]]:
    """Find all potentially borrowed resident keys before prefixes or guards.

    Inventory covers every candidate in the complete WorkPlan geometry, not
    just accepted rows or the current prefix. Only resident families need key
    computation. Candidate membership never invokes a numerical guard and does
    not change routing; whole-batch intersections remain eligible for borrowing.
    """
    table = geometry.table
    zero = tuple(0 for _ in range(table.level.shape[1]))
    identity = context.source.model(zero).identity
    current = cache.families(identity, context.budget)
    inventory = {}
    if current:
        domains = _candidate_domains(geometry)
        for domain in domains.values():
            mask, families = domain.mask, domain.families
            rows = np.flatnonzero(mask)
            if not len(rows):
                continue
            for name in families:
                if name in current:
                    keys = kernels[name].prepare_keys(table, rows, context)
                    inventory.setdefault(
                        (name, identity, context.budget), set()
                    ).update(key.tobytes() for key in keys)
    return inventory


@dataclass(frozen=True)
class ShapedPlan:
    """Acquire-wide shaped costs, fixed before any numerical guard.

    rows: all ordinary shaped global rows, in ascending order.
    order: (len(rows), 3) gate/period/direct index permutations.
    Workers borrow these arrays and never recalculate reuse from their subset.
    """

    rows: NDArray[np.int64]
    order: NDArray[np.int64]


@dataclass(frozen=True)
class Selection:
    """Resolve deferred global rows inside their single timeline worker.

    rows: global rows owned by this worker's timeline.
    shaped: acquire-wide cost plan, or None if no selected row is shaped.
    two_tone_direct: acquire-wide global-row preference, or None without two tones.
    names: caller-accepted family context for timeline-local frame planning.
    plans: caller-accepted decisions to retain in the timeline-local plan.
    cached: caller-looked-up model-only payload references for worker admission.
    A selection disables further deferral and retains accepted guard products.
    """

    rows: NDArray[np.int64]
    shaped: ShapedPlan | None
    rejected: Mapping[str, NDArray[np.bool_]]
    two_tone_direct: NDArray[np.bool_] | None = None
    names: NDArray[np.str_] | None = None
    plans: Mapping[str, RowPlan] = field(default_factory=dict)
    cached: Mapping[str, tuple[PreparedPayload, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Dispatch:
    """Call-local interval execution plan, consumed by Schedule.

    Within the classified scope, every pending interval is covered exactly once:
    all Geometry.pending rows for an acquire, or Selection.rows for one worker.
    Each has a nonempty family in names, or a True deferred bit with the contingent
    Selection(rows, shaped, rejected) resolved by its timeline worker. Already
    adopted prefix intervals are excluded. Schedule owns ordering and coverage;
    callers must not reclassify accepted rows or reconstruct guard products.
    This plan belongs to the same geometry, registry, source and budget for one
    evolve, not to shadow assignment or portable replay.

    names: family per global row; empty deferred slices are filled by their worker.
    diagnostics: counts for decisions resolved so far, refreshed after workers join.
    costs: nonnegative scheduling estimates, including provisional deferred costs.
    preparations: accepted caller guard products, grouped by family.
    rejected_preparations: model-only handoffs with no accepted rows, never apply
        bindings. Caller shared guards cap these by spare resident capacity and
        per-key quotas; Schedule consumes this mutable handoff queue when offering
        them below actual consumers. Only admission retains unused products then.
    deferred: worker-owned row mask, or None when no rows need worker classification.
    shaped: fixed acquire-wide cost plan, or None when no shaped work exists.
    rejected: caller guard decisions to skip in worker-local fallback chains.
    two_tone_direct: global-row cost preference fixed before either two-tone guard;
        None when no constant two-tone candidates exist.
    plans: immutable family-owned global-row decisions passed unchanged to apply;
        evolve-local plans never enter the model-only prepare cache.
    cached: strong resident references passed to deferred workers without cache IO.
    chain: global-row chain IDs; -1 for independent rows. A timeline worker
        replaces provisional decisions locally after its deferred guards finish.
    """

    names: NDArray[np.str_]
    diagnostics: PathDiagnostics
    costs: NDArray[np.float64]
    preparations: Mapping[str, tuple[GuardPreparation, ...]] = field(
        default_factory=dict
    )
    deferred: NDArray[np.bool_] | None = None
    shaped: ShapedPlan | None = None
    rejected: Mapping[str, NDArray[np.bool_]] = field(default_factory=dict)
    two_tone_direct: NDArray[np.bool_] | None = None
    plans: Mapping[str, RowPlan] = field(default_factory=dict)
    chain: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    cached: Mapping[str, tuple[PreparedPayload, ...]] = field(default_factory=dict)
    rejected_preparations: dict[str, tuple[GuardPreparation, ...]] = field(
        default_factory=dict
    )


def diagnostics(table: SegmentTable, names: NDArray[np.str_]) -> PathDiagnostics:
    """Count resolved global rows and distinct timeline/family pairs.

    Empty names denote deferred rows and are omitted until workers resolve them.
    This function does not run guards or alter names.
    """
    rows = np.flatnonzero(names != "")
    unique, counts = np.unique(names[rows], return_counts=True)
    pairs = np.empty(len(rows), dtype=[("name", "U16"), ("timeline", "i8")])
    pairs["name"], pairs["timeline"] = names[rows], table.timeline[rows]
    point_names, point_counts = np.unique(np.unique(pairs)["name"], return_counts=True)
    return PathDiagnostics(
        dict(zip(unique.tolist(), counts.tolist(), strict=True)),
        dict(zip(point_names.tolist(), point_counts.tolist(), strict=True)),
    )


def require(
    kernels: Mapping[str, Kernel],
    name: str,
    table: SegmentTable,
    rows: NDArray[np.int64],
) -> Kernel:
    """Resolve a required family or identify the first affected segment."""
    if name not in kernels:
        row = int(rows[0])
        raise RuntimeError(
            f"segment {row} (timeline {table.timeline[row]}, start {table.start_ns[row]} ns) requires kernel {name}"
        )
    return kernels[name]


def prepare(
    kernel: Kernel,
    keys: NDArray[np.void],
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
) -> Prepared:
    """Select a key-level builder and prepare an aligned batch of representatives.

    Called at the scheduler's normal build point, after cache lookup. Gate keys
    with duration in (0, 30] ns try phase-adaptive preparation; all other gate
    keys use sixteen nodes. Failed phase probes fall back inside the numerical
    build. Other families use their ordinary prepare entry point unchanged.
    Keys, representatives, budget and failure semantics follow Kernel.prepare.
    """
    if isinstance(kernel, GateCacheKernel):
        duration = keys["duration"]
        phase_adaptive = (duration > 0) & (duration <= 30.0)
        return kernel.prepare(keys, table, rows, context, phase_adaptive=phase_adaptive)
    return kernel.prepare(keys, table, rows, context)


class _GuardCacheHits(tuple[PreparedPayload, ...]):
    """Caller-indexed hits, borrowed unchanged by shared and deferred guards.

    Tuple contents retain the whole-payload references used by resident
    admission. by_key lists payload positions in hit order, including overlaps
    so a numerical domain miss can still try the next covering payload.
    """

    by_key: Mapping[bytes, tuple[int, ...]]

    def __new__(cls, payloads: Iterable[PreparedPayload]) -> Self:
        hits = super().__new__(cls, payloads)
        positions: dict[bytes, list[int]] = {}
        for index, payload in enumerate(hits):
            for key in dict.fromkeys(payload.keys):
                positions.setdefault(key, []).append(index)
        hits.by_key = {key: tuple(indices) for key, indices in positions.items()}
        return hits


_EMPTY_GUARD_HITS = _GuardCacheHits(())


@dataclass(frozen=True)
class _Guarded:
    rows: NDArray[np.int64]
    rejected: NDArray[np.int64]
    preparations: tuple[GuardPreparation, ...]
    plan: RowPlan | None = None
    rejected_preparations: tuple[GuardPreparation, ...] = ()


def _guard_key(
    kernel: Kernel,
    table: SegmentTable,
    context: KernelContext,
    rows: NDArray[np.int64],
) -> _Guarded:
    """Separate consumer handoffs from model-only products with no accepted row."""
    result = kernel.guard_prepared(table, rows, context)
    if result is None:
        raise RuntimeError(
            f"kernel {kernel.name} declares guard_builds without a handoff"
        )
    accepted = result.accepted
    selected = rows[accepted]
    preparations = result.preparations if len(selected) else ()
    return _Guarded(
        selected,
        rows[~accepted],
        preparations,
        result.plan,
        result.preparations if not len(selected) else (),
    )


def _guard_cached(
    kernel: Kernel,
    table: SegmentTable,
    context: KernelContext,
    rows: NDArray[np.int64],
    hits: _GuardCacheHits,
    rejected_limit: int | None = None,
) -> tuple[_Guarded, ...]:
    """Admit covered cached rows without building; build only domain misses."""
    remaining = np.ones(len(rows), dtype=bool)
    keys = kernel.prepare_keys(table, rows, context) if hits else ()
    positions: dict[int, list[int]] = {}
    for position, key in enumerate(keys):
        for index in hits.by_key.get(key.tobytes(), ()):
            positions.setdefault(index, []).append(position)
    results: list[_Guarded] = []
    # Keep hit order between payloads and original row order within each call.
    for index in sorted(positions):
        covered = np.array(positions[index], dtype=np.int64)
        covered = covered[remaining[covered]]
        if not len(covered):
            continue
        selected = rows[covered]
        result = kernel.admit_prepared(hits[index].prepared, table, selected, context)
        if result is None:
            continue
        remaining[covered] = False
        accepted = result.accepted
        results.append(
            _Guarded(
                selected[accepted],
                selected[~accepted],
                result.preparations if accepted.any() else (),
                result.plan,
            )
        )
    if remaining.any():
        built = _guard_key(kernel, table, context, rows[remaining])
        if rejected_limit is not None and built.rejected_preparations:
            retained: list[GuardPreparation] = []
            for preparation in built.rejected_preparations:
                charge = preparation.nbytes + RESIDENT_METADATA_BYTES
                if charge <= rejected_limit:
                    retained.append(preparation)
                    rejected_limit -= charge
            built = _Guarded(
                built.rows,
                built.rejected,
                built.preparations,
                built.plan,
                tuple(retained),
            )
        results.append(built)
    return tuple(results)


def _unshared_driven_input(table: SegmentTable) -> NDArray[np.bool_]:
    """Mark rows after a drive without a shared prefix input certificate.

    Rows are timeline-major and chronological. A preceding drive in another
    timeline does not change the source input. Certificates retain Dyson's
    shared input contraction; no-drive histories retain its cheap source path.
    """
    index = np.arange(len(table.timeline))
    first = np.searchsorted(table.timeline, table.timeline, side="left")
    last_drive = np.maximum.accumulate(np.where(table.tones > 0, index, -1))
    previous_drive = np.r_[-1, last_drive[:-1]]
    shared = (
        np.fromiter(
            (group is not None for group in table.prefix_group),
            dtype=bool,
            count=len(index),
        )
        if table.prefix_group is not None
        else np.zeros(len(index), dtype=bool)
    )
    return (previous_drive >= first) & ~shared


class _Classifier:
    def __init__(
        self,
        geometry: Geometry,
        kernels: Mapping[str, Kernel],
        context: KernelContext,
        pool: Executor | None = None,
        selection: Selection | None = None,
        admission: ResidentAdmission | None = None,
    ):
        self.admission = admission
        self.cache = admission.cache if admission is not None else None
        self.cached = (
            {
                name: hits
                if isinstance(hits, _GuardCacheHits)
                else _GuardCacheHits(hits)
                for name, hits in selection.cached.items()
            }
            if selection is not None
            else {}
        )
        self.pool, self.selection = pool, selection
        self.geometry, self.kernels = geometry, kernels
        self.context = context
        self.source, self.budget = context.source, context.budget
        self.table = geometry.table
        self.domains = _candidate_domains(geometry)
        self.names = np.full(len(self.table.timeline), "", dtype="U16")
        self.costs = np.zeros(len(self.names))
        self.deferred = np.zeros(len(self.names), dtype=bool)
        self.scope = geometry.pending
        if selection is not None:
            self.scope &= np.isin(np.arange(len(self.names)), selection.rows)
        self.shaped = selection.shaped if selection is not None else None
        self.two_tone_direct = (
            selection.two_tone_direct if selection is not None else None
        )
        self.preparations: dict[str, list[GuardPreparation]] = {}
        self.rejected_preparations: dict[str, list[GuardPreparation]] = {}
        self.rejected: dict[str, NDArray[np.bool_]] = {}
        self.guard_plans: dict[str, list[RowPlan]] = {}
        self.plans: dict[str, RowPlan] = {}

    def guard_keys(
        self, kernel: Kernel, rows: NDArray[np.int64]
    ) -> Iterable[_Guarded | None]:
        """Batch one timeline's local builders; guard shared keys with the pool."""
        if self.selection is not None:
            # This worker owns one timeline. A single batch lets the kernel
            # reuse call-local work across distinct prepare keys, then drop it.
            return _guard_cached(
                kernel,
                self.table,
                self.context,
                rows,
                self.cached.get(kernel.name, _EMPTY_GUARD_HITS),
            )
        groups = group_keys(kernel, self.table, rows, self.context)
        if self.cache is not None:
            dc = self.source.model(tuple(0 for _ in range(self.table.level.shape[1])))
            self.cached[kernel.name] = _GuardCacheHits(
                self.cache.find(
                    (kernel.name, dc.identity, self.budget),
                    {key.tobytes() for key in groups.keys},
                )
            )
        hits = self.cached.get(kernel.name, _EMPTY_GUARD_HITS)
        if isinstance(kernel, FloquetKernel) and self.cache is not None and hits:
            hits = self.extend_floquet_hits(kernel, hits)
            self.cached[kernel.name] = hits
        # Bound rejected handoffs even while pooled shared guards are returning.
        # Equal per-key quotas deliberately decline large non-revisiting scans.
        rejected_limit = (
            max(
                0,
                self.cache.limit
                - self.cache.charged_bytes
                + self.cache.reclaimable_bytes,
            )
            // max(1, int(np.count_nonzero(groups.shared)))
            if self.cache is not None
            else 0
        )
        local = rows[~groups.shared[groups.inverse]]
        self.deferred[local] = True
        if len(local):
            self.costs[local] = kernel.cost(self.table, local)
        selections = (
            selected for index, selected in groups.selections() if groups.shared[index]
        )
        if self.pool is not None and np.count_nonzero(groups.shared) >= 8:
            batches = self.pool(
                delayed(_guard_cached)(
                    kernel, self.table, self.context, selected, hits, rejected_limit
                )
                for selected in selections
            )
            results: list[_Guarded] = []
            for batch in batches:
                if batch is None:
                    raise RuntimeError("guard job returned no decision")
                results.extend(batch)
            return results
        return (
            result
            for selected in selections
            for result in _guard_cached(
                kernel, self.table, self.context, selected, hits, rejected_limit
            )
        )

    def extend_floquet_hits(
        self, kernel: FloquetKernel, hits: _GuardCacheHits
    ) -> _GuardCacheHits:
        """Extend complete cached domains in the caller before workers borrow them.

        Cache ownership is replaced, not added alongside the shared frame. The
        full-table inventory makes later shared/deferred guards read-only, so
        worker completion order cannot lose a concurrent bin extension.
        """
        if self.cache is None:
            raise RuntimeError("Floquet extension requires caller cache ownership")
        dc = self.source.model(tuple(0 for _ in range(self.table.level.shape[1])))
        namespace = kernel.name, dc.identity, self.budget
        extended: list[PreparedPayload] = []
        for payload in hits:
            prepared = kernel.extend_prepared(
                payload.prepared, self.table, self.context
            )
            if prepared is payload.prepared:
                extended.append(payload)
                continue
            replacement = PreparedPayload(prepared, payload.keys)
            if self.admission is None:
                raise RuntimeError("Floquet extension requires caller admission")
            self.admission.replace(namespace, payload, replacement)
            extended.append(replacement)
        return _GuardCacheHits(extended)

    def try_kernel(self, kernel: Kernel, rows: NDArray[np.int64]) -> NDArray[np.int64]:
        """Batch cheap guards in the caller and never retry its rejected rows."""
        eligible = rows
        if self.selection is not None:
            previous = self.selection.rejected.get(kernel.name)
            if previous is not None:
                eligible = rows[~previous[rows]]
        if not len(eligible):
            return rows
        if kernel.guard_builds:
            results = self.guard_keys(kernel, eligible)
        else:
            accepted = kernel.guard(self.table, eligible, self.context)
            results = (_Guarded(eligible[accepted], eligible[~accepted], ()),)
        for result in results:
            if result is None:
                raise RuntimeError("guard job returned no decision")
            self.names[result.rows] = kernel.name
            if len(result.rejected):
                self.rejected.setdefault(
                    kernel.name, np.zeros(len(self.names), dtype=bool)
                )[result.rejected] = True
            if result.plan is not None:
                self.guard_plans.setdefault(kernel.name, []).append(result.plan)
            if result.rejected_preparations:
                self.rejected_preparations.setdefault(kernel.name, []).extend(
                    result.rejected_preparations
                )
            if result.preparations:
                self.preparations.setdefault(kernel.name, []).extend(
                    result.preparations
                )
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def accept(self, rows: NDArray[np.int64], candidates: tuple[str, ...]) -> None:
        """Try numerical guards in fixed cost order, or defer the remaining chain."""
        for name in candidates:
            if not len(rows):
                return
            rows = self.try_kernel(require(self.kernels, name, self.table, rows), rows)
        if len(rows):
            raise RuntimeError(f"no valid kernel for segment {rows[0]}")

    def classify_shaped(self, rows: NDArray[np.int64]) -> None:
        """Share acquire-wide costs with workers, independent of spectral outcomes."""
        if self.shaped is None:
            self.shaped = ShapedPlan(
                rows,
                preference(self.table, rows, self.kernels, self.context),
            )
        plan = self.shaped
        families = self.domains["shaped"].families
        covariant = self.kernels.get(families[0])
        if covariant is not None:
            selected = plan.rows[self.scope[plan.rows]]
            preferred = single_gate_preferred(self.table, selected, self.context)
            if np.any(preferred):
                self.try_kernel(covariant, selected[preferred])
        spectral = self.kernels.get(families[1])
        if spectral is not None:
            selected = plan.rows[
                self.scope[plan.rows]
                & (self.names[plan.rows] == "")
                & ~self.deferred[plan.rows]
            ]
            if len(selected):
                self.try_kernel(spectral, selected)
        pending = (
            self.scope[plan.rows]
            & (self.names[plan.rows] == "")
            & ~self.deferred[plan.rows]
        )
        families = families[2:]
        pending_rows, order = plan.rows[pending], plan.order[pending]
        for permutation in np.unique(order, axis=0):
            selected = pending_rows[np.all(order == permutation, axis=1)]
            self.accept(selected, tuple(families[index] for index in permutation))

    def classify(self) -> Dispatch:
        table = self.table
        rows = np.flatnonzero(self.scope)
        for name in ("idle", "flux"):
            domain = self.domains[name]
            self.accept(rows[domain.mask[rows]], domain.families)
        domain = self.domains["plateau"]
        plateau = rows[domain.mask[rows]]
        # Distinct inputs cannot share Dyson paths; try constant RWA and period
        # propagation first within 10 us. Keep Dyson when both guards reject.
        distinct_input = _unshared_driven_input(table) & (table.duration_ns <= 10000.0)
        dyson, rwa, period, terminal = domain.families
        prefix = (rwa,) if rwa in self.kernels else ()
        self.accept(
            plateau[distinct_input[plateau]], (*prefix, period, dyson, terminal)
        )
        self.accept(
            plateau[~distinct_input[plateau]], (dyson, *prefix, period, terminal)
        )
        shaped = np.flatnonzero(self.domains["shaped"].mask & self.scope)
        if len(shaped):
            self.classify_shaped(shaped)
        two_tone = np.flatnonzero(self.domains["two"].mask)
        if len(two_tone):
            self.classify_two_tone(two_tone)
        domain = self.domains["multiple"]
        multiple = rows[domain.mask[rows]]
        # Each calibrated approximation is optional; the direct terminal is
        # required even when neither fast path is present in a custom registry.
        optional = tuple(name for name in domain.families[:-1] if name in self.kernels)
        self.accept(multiple, (*optional, domain.families[-1]))
        return self._result()

    def classify_two_tone(self, rows: NDArray[np.int64]) -> None:
        """Fix acquire-wide costs before either guard; workers borrow the plan."""
        if self.two_tone_direct is None:
            self.two_tone_direct = np.zeros(len(self.names), dtype=bool)
            self.two_tone_direct[rows] = direct_preferred(
                self.table, rows, self.context
            )
        selected = rows[self.scope[rows]]
        direct = self.two_tone_direct[selected]
        rwa, floquet, dop853 = self.domains["two"].families
        prefix = (rwa,) if rwa in self.kernels else ()
        # Production DOP853 accepts every valid row and is terminal here.
        # Keep the Floquet tail for injected same-name kernels that may reject.
        self.accept(selected[direct], (*prefix, dop853, floquet))
        self.accept(selected[~direct], (*prefix, floquet, dop853))
        self.retain_floquet_plan()

    def retain_floquet_plan(self) -> None:
        """Keep caller-accepted admissions alongside this timeline's guard results."""
        plans = list(self.guard_plans.get("floquet", ()))
        if self.selection is not None:
            previous = self.selection.plans.get("floquet")
            if previous is not None:
                if not isinstance(previous, FloquetPlan):
                    raise TypeError("Floquet requires its own row plan")
                timeline = self.table.timeline[self.selection.rows[0]]
                first, stop = np.searchsorted(
                    self.table.timeline, [timeline, timeline + 1]
                )
                plans.append(
                    FloquetPlan(
                        {
                            row: previous.admissions[row]
                            for row in range(int(first), int(stop))
                            if row in previous.admissions
                        }
                    )
                )
        if plans:
            self.plans["floquet"] = FloquetPlan.merge(plans)

    def _result(self) -> Dispatch:
        for name in np.unique(self.names[self.names != ""]):
            rows = np.flatnonzero(self.names == name)
            self.costs[rows] = self.kernels[str(name)].cost(self.table, rows)
        if not np.isfinite(self.costs).all() or np.any(self.costs < 0):
            raise ValueError("kernel costs must be finite and nonnegative")
        if self.selection is not None and self.selection.names is not None:
            point = int(self.table.timeline[self.selection.rows[0]])
            first, stop = (
                np.searchsorted(self.table.timeline, point, side="left"),
                np.searchsorted(self.table.timeline, point, side="right"),
            )
            context_rows = np.arange(first, stop, dtype=np.int64)
            context = self.names[context_rows].copy()
            unresolved = context == ""
            context[unresolved] = self.selection.names[context_rows[unresolved]]
            context[~self.geometry.pending[context_rows]] = ""
            plans, chain = direct_plans(self.table, context, context_rows)
        else:
            plans, chain = direct_plans(self.table, self.names)
        self.plans.update(plans)
        return Dispatch(
            self.names,
            diagnostics(self.table, self.names),
            self.costs,
            {name: tuple(items) for name, items in self.preparations.items()},
            self.deferred,
            self.shaped,
            self.rejected,
            self.two_tone_direct,
            self.plans,
            chain,
            self.cached,
            {name: tuple(items) for name, items in self.rejected_preparations.items()},
        )


def classify(
    geometry: Geometry,
    kernels: Mapping[str, Kernel],
    context: KernelContext,
    pool: Executor | None = None,
    selection: Selection | None = None,
    *,
    admission: ResidentAdmission | None = None,
) -> Dispatch:
    """Guard shared candidates now and resolve local candidates in their worker.

    Eight or more shared prepare keys may use a borrowed Engine threading pool.
    Smaller shared selections stay serial. Selection resolves one timeline's
    deferred rows without a nested pool, retaining guard products for apply.
    Both modes use the same acquire-wide cost plan and numerical guards.
    admission=None means this call does not look up resident payloads. Callers
    supply the evolve-wide admission under their call lock. Deferred workers
    borrow strong references through Selection.cached and never access the
    resident cache themselves.
    """
    return _Classifier(
        geometry, kernels, context, pool, selection, admission
    ).classify()
