"""Cross-family order and costs; kernels own numerical guards and family-local reuse."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
from joblib import delayed
from numpy.typing import NDArray

from zcu_tools.simulate.boundary import CancellationQuery, Cancelled

from ._capture import CaptureRound
from ._carrier_plan import populate_carrier_plan, ramp_families_for_rows
from ._common_frame_dispatch import preferred as common_frame_preferred
from ._dop853_dispatch import direct_plans, opencl_rows
from ._gain_plan import plan_gain_groups
from ._inspection import DispatchCandidate, GuardDecision, needs_prefix_input
from ._multitone_dispatch import (
    expanded_direct,
    gate_preferred,
    populate_gate_inventory,
    populate_multitone_inventory,
)
from ._parallel import Executor
from ._planning import Geometry
from ._prepare_keys import KeyGroups, group_keys
from ._shaped_dispatch import (
    populate_single_gate_plan,
    preference,
    single_gate_preferred,
)
from ._shared_multitone_dispatch import route_sampled, split_gate_outsiders
from ._slow_dispatch import direct_preferred as slow_direct_preferred
from ._two_tone_dispatch import first_candidate
from .engine import PathDiagnostics
from .exceptions import NoValidKernel
from .kernel import (
    GuardPreparation,
    Kernel,
    KernelContext,
    Prepared,
    RowPlan,
    SegmentTable,
)
from .kernels._multitone_spot import sampled_groups
from .kernels.common_frame import zero_mean_carrier
from .kernels.common_frame_kernel import CommonFrameKernel
from .kernels.dop853 import DOP853Kernel
from .kernels.floquet import FloquetPlan
from .kernels.gate_cache import GateCacheKernel
from .kernels.multitone_gate import MultitoneGateKernel
from .kernels.multitone_rwa import MultitoneRWAKernel
from .kernels.slow_envelope import low_carrier
from .secular import ModelSource
from .workplan import WorkPlan


def populate_work_plan(
    table: SegmentTable,
    source: ModelSource,
    plan: WorkPlan,
    budget: float = 1e-3,
    *,
    cancel: CancellationQuery | None = None,
) -> None:
    """Populate a caller-owned plan for the full immutable geometry.

    Run the group planners before workers start. budget is the numerical
    admission budget for cold mixed gate plans; source owns model identity
    and routing history. Raise Cancelled before recording any routing usage
    if cancel is set between inventory and single-gate planning. The caller
    closes the plan on success and on every failure.
    """
    populate_carrier_plan(table, source, plan)
    plan_gain_groups(table, source, plan)
    populate_gate_inventory(table, source, plan)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    populate_single_gate_plan(table, source, plan, budget)
    populate_multitone_inventory(table, source, plan, budget)


def _domains(geometry: Geometry) -> dict[str, NDArray[np.bool_]]:
    """Partition structural row domains before costs and numerical guards.

    Each domain's classifier owns its named candidate choices.
    """
    table = geometry.table
    secular_only = ~table.full_loss
    flux = geometry.flux_drive & (table.tones > 0) & secular_only
    ordinary = ~flux & secular_only
    constant = table.constant_envelopes(np.arange(len(table.timeline), dtype=np.int64))
    single = ordinary & (table.tones == 1)
    slow = single & constant
    if table.tone_frequency_ghz.shape[1]:
        slow &= low_carrier(table.tone_frequency_ghz[:, 0])
    adiabatic = geometry.flux_drive & (table.tones == 0) & secular_only
    return {
        "idle": (table.tones == 0) & ~adiabatic & secular_only,
        "adiabatic": adiabatic,
        "flux": flux,
        "slow": slow,
        "plateau": single & constant & ~slow,
        "shaped": single & ~constant,
        "two": ordinary & (table.tones == 2) & constant,
        "multiple": ordinary & (((table.tones == 2) & ~constant) | (table.tones >= 3)),
    }


@dataclass(frozen=True)
class ShapedPlan:
    """Acquire-wide shaped costs, fixed before any numerical guard.

    rows: all ordinary shaped global rows, in ascending order.
    order: (len(rows), 3) cheapest-first gate_cache/period_table/dop853 names.
    Workers borrow these arrays and never recalculate reuse from their subset.
    """

    rows: NDArray[np.int64]
    order: NDArray[np.str_]


@dataclass(frozen=True)
class Selection:
    """Resolve deferred global rows inside their single timeline worker.

    rows: global rows owned by this worker's timeline.
    shaped: acquire-wide cost plan, or None if no selected row is shaped.
    two_tone_first: global-row cheaper family names, or None without two tones.
    names: caller-accepted family context for timeline-local frame planning.
    plans: caller-accepted decisions to retain in the timeline-local plan.
    A selection disables further deferral and retains accepted guard products.
    """

    rows: NDArray[np.int64]
    shaped: ShapedPlan | None
    rejected: Mapping[str, NDArray[np.bool_]]
    two_tone_first: NDArray[np.str_] | None = None
    names: NDArray[np.str_] | None = None
    plans: Mapping[str, RowPlan] = field(default_factory=dict)


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
    deferred: worker-owned row mask, or None when no rows need worker classification.
    shaped: fixed acquire-wide cost plan, or None when no shaped work exists.
    rejected: caller guard decisions to skip in worker-local fallback chains.
    two_tone_first: global-row cheaper family names fixed before either guard;
        None when no constant two-tone candidates exist.
    plans: immutable family-owned global-row decisions passed unchanged to apply;
        plans live only for this evolve.
    chain: global-row chain IDs; -1 for independent rows. A timeline worker
        replaces provisional decisions locally after its deferred guards finish.
    opencl_rows: read-only global-row mask offering ready DOP853 endpoints to
        OpenCL, or None when no such routes are offered.
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
    two_tone_first: NDArray[np.str_] | None = None
    plans: Mapping[str, RowPlan] = field(default_factory=dict)
    chain: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    opencl_rows: NDArray[np.bool_] | None = None


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
        duration_ns_by_kernel={
            str(name): float(np.sum(table.duration_ns[names == name]))
            for name in unique
        },
        point_members={
            str(name): tuple(np.unique(table.timeline[names == name]).tolist())
            for name in point_names
        },
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

    Gate-cache keys with duration in (0, 30] ns try phase-adaptive preparation;
    all other gate-cache keys use sixteen nodes. Failed phase probes fall back
    inside that build. Keys, representatives, budget and failure semantics
    follow Kernel.prepare.
    """
    routing = context.work_plan.routing
    # The multitone gate counts only the producer of each evolve-local key.
    if not isinstance(kernel, MultitoneGateKernel):
        routing.prepared_keys(kernel.name, len(keys))
    with routing.measure(kernel.name, "prepare"):
        if isinstance(kernel, GateCacheKernel):
            duration = keys["duration"]
            phase_adaptive = (duration > 0) & (duration <= 30.0)
            return kernel.prepare(
                keys, table, rows, context, phase_adaptive=phase_adaptive
            )
        return kernel.prepare(keys, table, rows, context)


@dataclass(frozen=True)
class _Guarded:
    rows: NDArray[np.int64]
    rejected: NDArray[np.int64]
    preparations: tuple[GuardPreparation, ...]
    plan: RowPlan | None = None


def _guard_key(
    kernel: Kernel,
    table: SegmentTable,
    context: KernelContext,
    rows: NDArray[np.int64],
    handoff: GuardPreparation | None = None,
) -> _Guarded:
    """Keep guard products only when this call has accepted consumers.

    A common-frame handoff can cover new row admissions without another build.
    """
    routing = context.work_plan.routing
    with routing.measure(kernel.name, "prepare"):
        if handoff is None:
            result = kernel.guard_prepared(table, rows, context)
        elif isinstance(kernel, CommonFrameKernel):
            result = kernel.admit_prepared(handoff.prepared, table, rows, context)
        else:
            raise TypeError("only common frame accepts a coefficient handoff")
    if result is None:
        raise RuntimeError(
            f"kernel {kernel.name} declares guard_builds without a handoff"
        )
    if routing.enabled and handoff is None:
        routing.prepared_keys(
            kernel.name, sum(len(p.keys) for p in result.preparations)
        )
    accepted = result.accepted
    selected = rows[accepted]
    preparations = (handoff,) if handoff is not None else result.preparations
    return _Guarded(
        selected,
        rows[~accepted],
        preparations if len(selected) else (),
        result.plan,
    )


def _eager_sambe_keys(
    groups: KeyGroups,
    table: SegmentTable,
    context: KernelContext,
    seeded: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Expand eager local keys to their timeline-local spectral groups."""
    eager = seeded & ~groups.shared
    if not np.any(eager):
        return eager

    # The original worker call shared spectral work across duration keys. Do
    # not leave some of the same timeline's spectral group in that worker.
    timelines = table.timeline[groups.representative]
    local = np.flatnonzero(~groups.shared & np.isin(timelines, timelines[eager]))
    families = ramp_families_for_rows(table, groups.representative[local], context)
    # Use the identity of the immutable WorkPlan records, not their row lists.
    buckets = [
        (timeline, id(family))
        for timeline, family in zip(timelines[local], families, strict=True)
    ]
    advanced = {
        bucket
        for bucket, selected in zip(buckets, eager[local], strict=True)
        if selected
    }
    eager[local] = [bucket in advanced for bucket in buckets]
    return eager


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


class Classifier:
    """Resolve one dispatch scope while borrowing complete-acquire routing facts."""

    def __init__(
        self,
        geometry: Geometry,
        kernels: Mapping[str, Kernel],
        context: KernelContext,
        pool: Executor | None = None,
        selection: Selection | None = None,
        trace: dict[int, dict[str, DispatchCandidate]] | None = None,
    ):
        self.trace = trace
        self.capture: CaptureRound | None = None
        self.pool, self.selection = pool, selection
        self.geometry = geometry
        self.kernels = (
            kernels
            if context.families is None
            else {
                name: kernel
                for name, kernel in kernels.items()
                if name in context.families
            }
        )
        self.context = context
        self.source, self.budget = context.source, context.budget
        self.table = geometry.table
        self.domains = _domains(geometry)
        self.names = np.full(len(self.table.timeline), "", dtype="U16")
        self.costs = np.zeros(len(self.names))
        self.deferred = np.zeros(len(self.names), dtype=bool)
        self.scope = geometry.pending
        if trace is not None:
            # Query cannot evaluate a guard that needs an evolved prefix density.
            self.scope &= ~needs_prefix_input(self.table)
        if selection is not None:
            self.scope &= np.isin(np.arange(len(self.names)), selection.rows)
        self.shaped = selection.shaped if selection is not None else None
        self.two_tone_first = (
            selection.two_tone_first if selection is not None else None
        )
        self.preparations: dict[str, dict[int, GuardPreparation]] = {}
        # This classifier alone owns tentative pricing handoffs. They do not
        # enter Dispatch unless a later guard selects a consumer.
        self.common_handoffs: dict[bytes, GuardPreparation] = {}
        self.common_prices: dict[int, bool] = {}
        self.rejected: dict[str, NDArray[np.bool_]] = {}
        self.guard_plans: dict[str, list[RowPlan]] = {}
        self.plans: dict[str, RowPlan] = {}

    def scheduling_cost(
        self, kernel: Kernel, rows: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        if isinstance(kernel, DOP853Kernel):
            return kernel.cost(self.table, rows, budget=self.budget)
        return kernel.cost(self.table, rows)

    def observe(
        self, kernel: Kernel, rows: NDArray[np.int64], decision: GuardDecision
    ) -> None:
        """Observe actual decisions; only an inspection requests extra pricing."""
        if self.capture is not None:
            self.capture.observe(kernel.name, rows, decision)
        if self.trace is None or not len(rows):
            return
        scheduling_costs = (
            self.scheduling_cost(kernel, rows)
            if decision != "untested_after_short_circuit"
            else [None] * len(rows)
        )
        for row, scheduling_cost in zip(rows, scheduling_costs, strict=True):
            candidates = self.trace.setdefault(int(row), {})
            if decision == "untested_after_short_circuit" and kernel.name in candidates:
                continue
            candidates[kernel.name] = DispatchCandidate(
                kernel.name,
                decision,
                None if scheduling_cost is None else float(scheduling_cost),
            )

    def guard_keys(
        self,
        kernel: Kernel,
        rows: NDArray[np.int64],
        eager_guard_rows: NDArray[np.int64] | None,
    ) -> Iterable[_Guarded | None]:
        """Guard shared keys and preserve batching for any advanced local Sambe keys."""
        if self.selection is not None and not isinstance(kernel, CommonFrameKernel):
            # This worker owns one timeline. A single batch lets the kernel
            # reuse call-local work across distinct prepare keys, then drop it.
            return (_guard_key(kernel, self.table, self.context, rows),)
        groups = group_keys(kernel, self.table, rows, self.context)
        # A common refusal must reach the caller's complete legacy key groups.
        # Deferring a unique common key can otherwise split one shared Floquet
        # (or RWA) build into one build per timeline after the refusal.
        per_key = (
            np.ones(len(groups.keys), dtype=bool)
            if kernel.name == "common_frame"
            else groups.shared
        )
        resolved = per_key[groups.inverse]
        timeline_batches = []
        if eager_guard_rows is not None and len(eager_guard_rows):
            seeded = np.zeros(len(groups.keys), dtype=bool)
            seeded[groups.inverse[np.isin(rows, eager_guard_rows)]] = True
            eager = _eager_sambe_keys(groups, self.table, self.context, seeded)
            eager_rows = eager[groups.inverse]
            resolved |= eager_rows
            # Rows advanced only for Sambe reuse keep worker fallback if refused.
            self.deferred[rows[eager_rows & ~seeded[groups.inverse]]] = True
            selected = rows[eager_rows]
            if len(selected):
                timelines = self.table.timeline[selected]
                timeline_batches = np.split(
                    selected, np.flatnonzero(timelines[1:] != timelines[:-1]) + 1
                )
        local = rows[~resolved]
        self.deferred[local] = True
        self.observe(kernel, local, "deferred")
        if len(local):
            self.costs[local] = self.scheduling_cost(kernel, local)
        selections = chain(
            (selected for index, selected in groups.selections() if per_key[index]),
            timeline_batches,
        )

        def guard(selected: NDArray[np.int64]) -> _Guarded:
            handoff = None
            if self.common_handoffs and isinstance(kernel, CommonFrameKernel):
                key = kernel.prepare_keys(self.table, selected[:1], self.context)[0]
                handoff = self.common_handoffs.get(key.tobytes())
            return _guard_key(kernel, self.table, self.context, selected, handoff)

        if (
            self.pool is not None
            and np.count_nonzero(per_key) + len(timeline_batches) >= 8
        ):
            results = self.pool(delayed(guard)(selected) for selected in selections)
            if any(result is None for result in results):
                raise RuntimeError("guard job returned no decision")
            return results
        return (guard(selected) for selected in selections)

    def eligible_rows(
        self, kernel: Kernel, rows: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Keep structural candidates not rejected by caller or worker guards."""
        eligible = rows
        if kernel.name == "common_frame":
            # Its existing dispatch domain is two tones, even when cost is bypassed.
            eligible = eligible[self.table.tones[eligible] == 2]
            invalid_carrier = np.array(
                [
                    zero_mean_carrier(self.table.tone_frequency_ghz[row, :2])
                    for row in eligible
                ],
                dtype=bool,
            )
            eligible = eligible[~invalid_carrier]
        if self.context.families is not None:
            rejected = self.rejected.get(kernel.name)
            if rejected is not None:
                eligible = eligible[~rejected[eligible]]
        if self.selection is not None:
            previous = self.selection.rejected.get(kernel.name)
            if previous is not None:
                eligible = eligible[~previous[eligible]]
        return eligible

    def guard_outcomes(
        self,
        kernel: Kernel,
        rows: NDArray[np.int64],
        eager_sambe_rows: NDArray[np.int64] | None,
    ) -> Iterable[_Guarded | None]:
        """Evaluate non-build guards and eager group decisions in this scope."""
        if kernel.guard_builds:
            return self.guard_keys(kernel, rows, eager_sambe_rows)
        with self.context.work_plan.routing.measure(kernel.name, "prepare"):
            accepted = kernel.guard(self.table, rows, self.context)
        return (_Guarded(rows[accepted], rows[~accepted], ()),)

    def try_kernel(
        self,
        kernel: Kernel,
        rows: NDArray[np.int64],
        *,
        eager_guard_rows: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Guard rows once; advance Sambe groups needed for shared fallbacks."""
        eligible = self.eligible_rows(kernel, rows)
        if not len(eligible):
            return rows
        for result in self.guard_outcomes(kernel, eligible, eager_guard_rows):
            if result is None:
                raise RuntimeError("guard job returned no decision")
            self.names[result.rows] = kernel.name
            if eager_guard_rows is not None:
                self.deferred[result.rows] = False
                worker_fallback = result.rejected[self.deferred[result.rejected]]
                if len(worker_fallback):
                    self.costs[worker_fallback] = self.scheduling_cost(
                        kernel, worker_fallback
                    )
            self.observe(kernel, result.rows, "accepted")
            self.record_rejection(kernel, result.rejected)
            if result.plan is not None:
                self.guard_plans.setdefault(kernel.name, []).append(result.plan)
            if result.preparations:
                adopted = self.preparations.setdefault(kernel.name, {})
                for handoff in result.preparations:
                    # Reused coefficients share one handoff and one schedule binding.
                    adopted[id(handoff)] = handoff
                    if isinstance(kernel, CommonFrameKernel):
                        self.common_handoffs[handoff.keys[0].tobytes()] = handoff
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def record_rejection(self, kernel: Kernel, rows: NDArray[np.int64]) -> None:
        """Publish actual, in-scope guard refusals to telemetry and inspection."""
        if len(rows):
            self.observe(kernel, rows, "rejected")
            self.context.work_plan.routing.reject(kernel.name, len(rows))
            self.rejected.setdefault(
                kernel.name, np.zeros(len(self.names), dtype=bool)
            )[rows] = True

    def candidate_order(
        self, candidates: tuple[str, ...], cost_skipped: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Keep cost order; an allowlist adds its cost-skipped candidates last."""
        if self.context.families is None:
            return candidates
        return tuple(
            name
            for name in dict.fromkeys((*candidates, *cost_skipped))
            if name in self.context.families
        )

    def accept(
        self,
        domain: str,
        rows: NDArray[np.int64],
        candidates: tuple[str, ...],
        *,
        cost_skipped: tuple[str, ...] = (),
    ) -> None:
        """Try named candidates, exhausting cost-skipped allowed alternatives."""
        if len(rows):
            self.try_candidates(
                domain, rows, self.candidate_order(candidates, cost_skipped)
            )

    def try_candidates(
        self, domain: str, rows: NDArray[np.int64], candidates: tuple[str, ...]
    ) -> None:
        """Try the final ordered candidates or raise a refusal for this domain."""
        for index, name in enumerate(candidates):
            offered = rows
            rows = self.try_kernel(require(self.kernels, name, self.table, rows), rows)
            if self.trace is not None or self.capture is not None:
                stopped = offered[~np.isin(offered, rows)]
                for later in candidates[index + 1 :]:
                    kernel = self.kernels.get(later)
                    if kernel is not None:
                        self.observe(kernel, stopped, "untested_after_short_circuit")
            if not len(rows):
                return
        self.reject(domain, rows, candidates)

    def reject(
        self, domain: str, rows: NDArray[np.int64], attempted: tuple[str, ...]
    ) -> None:
        """Raise the domain and point failure for nonempty exhausted rows."""
        reason = {
            "full_loss": "; collapse operators require CPU dop853",
            "full_loss_flux": "; collapse operators require fixed flux and no supplied flux waveform",
        }.get(domain, "")
        points = np.unique(self.geometry.points[self.table.timeline[rows]]).tolist()
        raise NoValidKernel(
            f"no valid kernel for segment {rows[0]}; domain {domain}: "
            f"points {points}, segments {rows.tolist()}{reason}",
            domain=domain,
            points=tuple(points),
            segments=tuple(int(row) for row in rows),
            attempted=attempted,
        )

    def classify_shaped(self, rows: NDArray[np.int64]) -> None:
        """Share acquire-wide costs with workers, independent of spectral outcomes."""
        if self.shaped is None:
            self.shaped = ShapedPlan(
                rows,
                preference(self.table, rows, self.kernels, self.context),
            )
        plan = self.shaped
        covariant = self.kernels.get("single_tone_gate")
        if covariant is not None:
            selected = plan.rows[self.scope[plan.rows]]
            preferred = single_gate_preferred(self.table, selected, self.context)
            if np.any(preferred):
                self.try_kernel(covariant, selected[preferred])
        pending = (
            self.scope[plan.rows]
            & (self.names[plan.rows] == "")
            & ~self.deferred[plan.rows]
        )
        selected, order = plan.rows[pending], plan.order[pending]
        rounds = tuple(
            (
                selected[np.all(order == costs, axis=1)],
                self.candidate_order(
                    tuple(str(name) for name in costs),
                    ("single_tone_gate", "shaped_sambe"),
                ),
            )
            for costs in np.unique(order, axis=0)
        )
        spectral = self.kernels.get("shaped_sambe")
        if spectral is not None and len(selected):
            self.try_kernel(
                spectral, selected, eager_guard_rows=self.shared_period_fallback(rounds)
            )
        for offered, candidates in rounds:
            remaining = offered[(self.names[offered] == "") & ~self.deferred[offered]]
            if len(remaining):
                self.try_candidates("shaped", remaining, candidates)

    def shared_period_fallback(
        self, rounds: tuple[tuple[NDArray[np.int64], tuple[str, ...]], ...]
    ) -> NDArray[np.int64] | None:
        """Find rows whose first fallback can share a table across timelines."""
        period = self.kernels.get("period_table")
        if self.selection is not None or period is None:
            return None
        period_first = [rows for rows, order in rounds if order[:1] == (period.name,)]
        if not period_first:
            return None
        candidates = np.concatenate(period_first)
        if len(candidates) < 2:
            return None
        # A worker's Sambe refusal arrives too late for shared preparation.
        groups = group_keys(period, self.table, candidates, self.context)
        return candidates[groups.shared[groups.inverse]]

    def classify(self) -> Dispatch:
        rows = np.flatnonzero(self.scope)
        full = rows[self.table.full_loss[rows]]
        self.accept("full_loss_flux", full[self.geometry.flux_drive[full]], ())
        self.accept("full_loss", full, ("dop853",))
        self.accept("idle", rows[self.domains["idle"][rows]], ("t4",))
        self.accept(
            "adiabatic", rows[self.domains["adiabatic"][rows]], ("adiabatic_map",)
        )
        self.accept("flux", rows[self.domains["flux"][rows]], ("flux_dop853",))
        self.classify_slow(rows[self.domains["slow"][rows]])
        # Group shaped and constant consumers together before any legacy prepare.
        multiple = rows[self.domains["multiple"][rows]]
        rwa = self.kernels.get("multitone_rwa")
        sampled = np.zeros(len(multiple), dtype=bool)
        if len(multiple) and isinstance(rwa, MultitoneRWAKernel):
            complete = self.context.work_plan.complete_table
            if complete is None:
                raise RuntimeError("sampled RWA dispatch requires complete geometry")
            full_rows = multiple + self.context.complete_row_offset
            sampled = sampled_groups(complete, self.context).by_row[full_rows] >= 0
        common = self.domains["two"] | self.domains["multiple"]
        common[multiple[sampled]] = False
        self.try_common_frame(rows[common[rows]])
        self.classify_plateau(rows[self.domains["plateau"][rows]])
        shaped = np.flatnonzero(self.domains["shaped"] & self.scope)
        if len(shaped):
            self.classify_shaped(shaped)
        two_tone = np.flatnonzero(self.domains["two"])
        if len(two_tone):
            self.classify_two_tone(two_tone)
        early, late = split_gate_outsiders(self, multiple[~sampled])
        self.classify_multiple(early)
        if isinstance(rwa, MultitoneRWAKernel) and (np.any(sampled) or len(late)):
            diverted = route_sampled(self, multiple[sampled], late, rwa)
            self.classify_multiple(late, diverted)
        return self._result()

    def classify_slow(self, rows: NDArray[np.int64]) -> None:
        """Price dissipative DC channels over complete-acquire exact-key usage."""
        if not len(rows):
            return
        direct = slow_direct_preferred(
            self.table, np.flatnonzero(self.domains["slow"]), self.context
        )
        preferred = np.array([int(row) in direct for row in rows], dtype=bool)
        self.accept("slow", rows[preferred], ("dop853", "slow_envelope"))
        self.accept("slow", rows[~preferred], ("slow_envelope", "dop853"))

    def classify_plateau(self, rows: NDArray[np.int64]) -> None:
        """Prefer shared Dyson paths, or constant RWA/periods for distinct inputs."""
        distinct = _unshared_driven_input(self.table) & (
            self.table.duration_ns <= 10000.0
        )
        rwa = ("single_tone_rwa",) if "single_tone_rwa" in self.kernels else ()
        self.accept(
            "plateau", rows[distinct[rows]], (*rwa, "floquet_period", "dyson", "t6")
        )
        self.accept(
            "plateau", rows[~distinct[rows]], ("dyson", *rwa, "floquet_period", "t6")
        )

    def try_common_frame(self, rows: NDArray[np.int64]) -> NDArray[np.int64]:
        """Try the cold-priced variant without removing any original fallback."""
        kernel = self.kernels.get("common_frame")
        candidates = rows[self.table.tones[rows] == 2]
        if kernel is not None and len(candidates):
            rwa = self.kernels.get("multitone_rwa")
            preferred = common_frame_preferred(
                self.table,
                candidates,
                self.context,
                rwa if isinstance(rwa, MultitoneRWAKernel) else None,
            )
            offered = candidates[preferred]
            # Sampled pricing may already have refused these exact rows. No
            # handoff survives an all-refused key; do not rebuild it here.
            offset = self.context.complete_row_offset
            refused = np.fromiter(
                (
                    not self.common_prices.get(int(row) + offset, True)
                    for row in offered
                ),
                dtype=bool,
                count=len(offered),
            )
            self.record_rejection(kernel, offered[refused])
            self.try_kernel(kernel, offered[~refused])
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def try_multitone_gate(
        self, rows: NDArray[np.int64], diverted: Counter[bytes] | None = None
    ) -> NDArray[np.int64]:
        """Offer the existing gate only on its preferred rows."""
        gate = self.kernels.get("multitone_gate")
        if gate is not None and len(rows):
            preferred = gate_preferred(
                self.table, rows, self.context, tuple(self.kernels), diverted
            )
            self.try_kernel(gate, rows[preferred])
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def price_common_frame(
        self, kernel: Kernel, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Guard sorted complete-table contenders and retain accepted handoffs.

        Return the acceptance mask in contender order, without routing the rows.
        The original guard handoff can serve later pricing or actual routing.
        """
        missing = np.array(
            [row for row in rows if row not in self.common_prices], dtype=np.int64
        )
        if len(missing):
            accepted = self.price_new_common_frame(kernel, table, missing)
            self.common_prices.update(zip(missing, accepted.tolist(), strict=True))
        return np.array([self.common_prices[row] for row in rows], dtype=bool)

    def price_new_common_frame(
        self, kernel: Kernel, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Resolve not-yet-priced complete rows and retain each adopted key's handoff."""
        if not isinstance(kernel, CommonFrameKernel):
            with self.context.work_plan.routing.measure(kernel.name, "prepare"):
                return kernel.guard(table, rows, self.context)
        accepted = np.zeros(len(rows), dtype=bool)
        groups = group_keys(kernel, table, rows, self.context)
        for index, selected in groups.selections():
            key = groups.keys[index].tobytes()
            result = _guard_key(
                kernel, table, self.context, selected, self.common_handoffs.get(key)
            )
            accepted[np.searchsorted(rows, result.rows)] = True
            if result.preparations:
                (self.common_handoffs[key],) = result.preparations
        return accepted

    def classify_multiple(
        self, rows: NDArray[np.int64], diverted: Counter[bytes] | None = None
    ) -> None:
        """Try reusable gates, then the cheaper admitted RWA/direct alternative."""
        rows = rows[(self.names[rows] == "") & ~self.deferred[rows]]
        rows = self.try_multitone_gate(rows, diverted)
        preferred = expanded_direct(self.table, rows, self.context, self.kernels)
        rwa = ("multitone_rwa",) if "multitone_rwa" in self.kernels else ()
        self.accept(
            "multiple",
            rows[preferred],
            ("dop853", *rwa),
            cost_skipped=("common_frame", "multitone_gate"),
        )
        self.accept(
            "multiple",
            rows[~preferred],
            (*rwa, "dop853"),
            cost_skipped=("common_frame", "multitone_gate"),
        )

    def classify_two_tone(self, rows: NDArray[np.int64]) -> None:
        """Fix acquire-wide costs before either guard; workers borrow the plan."""
        selected = rows[
            self.scope[rows] & (self.names[rows] == "") & ~self.deferred[rows]
        ]
        if not len(selected):
            return
        if self.two_tone_first is None:
            self.two_tone_first = np.full(len(self.names), "", dtype="U16")
            self.two_tone_first[rows] = first_candidate(self.table, rows, self.context)
        expanded = expanded_direct(self.table, selected, self.context, self.kernels)
        prefix = ("multitone_rwa",) if "multitone_rwa" in self.kernels else ()
        self.accept(
            "two",
            selected[expanded],
            ("dop853", *prefix, "floquet"),
            cost_skipped=("common_frame",),
        )
        selected = selected[~expanded]
        direct = self.two_tone_first[selected] == "dop853"
        # Production DOP853 accepts every valid row and is terminal here.
        # Keep the Floquet tail for injected same-name kernels that may reject.
        self.accept(
            "two",
            selected[direct],
            (*prefix, "dop853", "floquet"),
            cost_skipped=("common_frame",),
        )
        self.accept(
            "two",
            selected[~direct],
            (*prefix, "floquet", "dop853"),
            cost_skipped=("common_frame",),
        )
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
            self.costs[rows] = self.scheduling_cost(self.kernels[str(name)], rows)
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
            plans, chain = direct_plans(
                self.table, context, context_rows, budget=self.context.budget
            )
        else:
            plans, chain = direct_plans(
                self.table, self.names, budget=self.context.budget
            )
        backend = self.context.opencl
        dimension = self.context.source.initial_density.shape[0]
        gpu = (
            opencl_rows(self.table, plans.get("dop853"), dimension)
            if backend is not None
            and backend.failure(dimension, dp54=self.context.opencl_dp54) is None
            and isinstance(self.kernels.get("dop853"), DOP853Kernel)
            else None
        )
        self.plans.update(plans)
        return Dispatch(
            self.names,
            diagnostics(self.table, self.names),
            self.costs,
            {name: tuple(items.values()) for name, items in self.preparations.items()},
            self.deferred,
            self.shaped,
            self.rejected,
            self.two_tone_first,
            self.plans,
            chain,
            opencl_rows=gpu,
        )


def classify(
    geometry: Geometry,
    kernels: Mapping[str, Kernel],
    context: KernelContext,
    pool: Executor | None = None,
    selection: Selection | None = None,
    *,
    trace: dict[int, dict[str, DispatchCandidate]] | CaptureRound | None = None,
) -> Dispatch:
    """Guard shared and eager candidates before scheduling, defer other local keys.

    Eight or more independent guard jobs may use a borrowed Engine threading pool.
    Smaller selections stay serial. Selection resolves one timeline's
    deferred rows without a nested pool, retaining guard products for apply.
    Both modes use the same acquire-wide cost plan and numerical guards.
    trace=None keeps ordinary evolve behavior. A query supplies an empty mapping
    to collect row/family observations and excludes rows that need an uncomputed
    driven prefix density. The mapping receives immutable DispatchCandidate
    records; no worker classification or prefix evolution runs for a query.
    A CaptureRound instead observes actual evolve decisions, without pricing
    or the query's prefix exclusion, and retains observations on failure.
    """
    query_trace = trace if isinstance(trace, dict) else None
    classifier = Classifier(geometry, kernels, context, pool, selection, query_trace)
    classifier.capture = trace if isinstance(trace, CaptureRound) else None
    try:
        return classifier.classify()
    finally:
        if classifier.capture is not None:
            classifier.capture.finish(classifier.names)
