"""Cross-family order and costs; kernels own numerical guards and family-local reuse."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from joblib import delayed
from numpy.typing import NDArray

from ._capture import CaptureRound
from ._carrier_plan import populate_carrier_plan
from ._common_frame_dispatch import preferred as common_frame_preferred
from ._dop853_dispatch import direct_plans, opencl_rows
from ._gain_plan import plan_gain_groups
from ._inspection import DispatchCandidate, GuardDecision, needs_prefix_input
from ._multitone_dispatch import (
    gate_preferred,
    populate_gate_inventory,
    populate_multitone_inventory,
    populate_multitone_plan,
)
from ._parallel import Executor
from ._planning import Geometry
from ._prepare_keys import group_keys
from ._shaped_dispatch import (
    populate_single_gate_plan,
    preference,
    single_gate_preferred,
)
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
from .kernels.common_frame import zero_mean_carrier
from .kernels.dop853 import DOP853Kernel
from .kernels.floquet import FloquetPlan
from .kernels.gate_cache import GateCacheKernel
from .kernels.multitone_rwa import MultitoneRWAKernel
from .kernels.slow_envelope import low_carrier
from .secular import ModelSource
from .workplan import WorkPlan


def create_work_plan(
    table: SegmentTable,
    source: ModelSource,
    budget: float = 1e-3,
    *,
    telemetry: bool = False,
    content_key: bytes = b"",
    history_mode: Literal["record", "project"] = "record",
) -> WorkPlan:
    """Create ownership for the complete evolve before prefix/worker planning.

    table is the full immutable geometry and source owns model identity.
    The plan retains table for lazy multitone and common-frame inventories.
    Algorithm-specific group planners populate this plan before workers start;
    budget is the numerical admission budget for cold mixed gate plans.
    telemetry enables calling-thread CPU accounting before those plans build.
    history_mode records complete usage for evolve or projects it without
    changing history for dispatch queries. Neither mode changes cost formulas.
    """
    plan = WorkPlan(
        content_key=content_key, complete_table=table, history_mode=history_mode
    )
    plan.routing.enabled = telemetry
    populate_carrier_plan(table, source, plan)
    plan_gain_groups(table, source, plan)
    populate_gate_inventory(table, source, plan)
    populate_single_gate_plan(table, source, plan, budget)
    populate_multitone_inventory(table, source, plan, budget)
    return plan


def _domains(geometry: Geometry) -> dict[str, NDArray[np.bool_]]:
    """Partition structural row domains before costs and numerical guards.

    Each domain's classifier owns its named candidate choices.
    """
    table = geometry.table
    flux = geometry.flux_drive & (table.tones > 0)
    ordinary = ~flux
    constant = table.constant_envelopes(np.arange(len(table.timeline), dtype=np.int64))
    single = ordinary & (table.tones == 1)
    slow = single & constant
    if table.tone_frequency_ghz.shape[1]:
        slow &= low_carrier(table.tone_frequency_ghz[:, 0])
    adiabatic = geometry.flux_drive & (table.tones == 0)
    return {
        "idle": (table.tones == 0) & ~adiabatic,
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

    Called at the scheduler's normal build point. Gate keys
    with duration in (0, 30] ns try phase-adaptive preparation; all other gate
    keys use sixteen nodes. Failed phase probes fall back inside the numerical
    build. Other families use their ordinary prepare entry point unchanged.
    Keys, representatives, budget and failure semantics follow Kernel.prepare.
    """
    with context.work_plan.routing.measure(kernel.name, "prepare"):
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
) -> _Guarded:
    """Keep guard products only when this call has accepted consumers."""
    with context.work_plan.routing.measure(kernel.name, "prepare"):
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
    )


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
        self.preparations: dict[str, list[GuardPreparation]] = {}
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
        self, kernel: Kernel, rows: NDArray[np.int64]
    ) -> Iterable[_Guarded | None]:
        """Batch one timeline's local builders; guard shared keys with the pool."""
        if self.selection is not None:
            # This worker owns one timeline. A single batch lets the kernel
            # reuse call-local work across distinct prepare keys, then drop it.
            return (_guard_key(kernel, self.table, self.context, rows),)
        groups = group_keys(kernel, self.table, rows, self.context)
        # A common refusal must reach the caller's complete legacy key groups.
        # Deferring a unique common key can otherwise split one shared Floquet
        # (or RWA/gate) build into one build per timeline after the refusal.
        caller_keys = (
            np.ones(len(groups.keys), dtype=bool)
            if kernel.name == "common_frame"
            else groups.shared
        )
        local = rows[~caller_keys[groups.inverse]]
        self.deferred[local] = True
        self.observe(kernel, local, "deferred")
        if len(local):
            self.costs[local] = self.scheduling_cost(kernel, local)
        selections = (
            selected for index, selected in groups.selections() if caller_keys[index]
        )
        if self.pool is not None and np.count_nonzero(caller_keys) >= 8:
            results = self.pool(
                delayed(_guard_key)(kernel, self.table, self.context, selected)
                for selected in selections
            )
            if any(result is None for result in results):
                raise RuntimeError("guard job returned no decision")
            return results
        return (
            _guard_key(kernel, self.table, self.context, selected)
            for selected in selections
        )

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

    def try_kernel(self, kernel: Kernel, rows: NDArray[np.int64]) -> NDArray[np.int64]:
        """Batch cheap guards in the caller and never retry its rejected rows."""
        eligible = self.eligible_rows(kernel, rows)
        if not len(eligible):
            return rows
        if kernel.guard_builds:
            results = self.guard_keys(kernel, eligible)
        else:
            with self.context.work_plan.routing.measure(kernel.name, "prepare"):
                accepted = kernel.guard(self.table, eligible, self.context)
            results = (_Guarded(eligible[accepted], eligible[~accepted], ()),)
        for result in results:
            if result is None:
                raise RuntimeError("guard job returned no decision")
            self.names[result.rows] = kernel.name
            self.observe(kernel, result.rows, "accepted")
            self.observe(kernel, result.rejected, "rejected")
            self.context.work_plan.routing.reject(kernel.name, len(result.rejected))
            if len(result.rejected):
                self.rejected.setdefault(
                    kernel.name, np.zeros(len(self.names), dtype=bool)
                )[result.rejected] = True
            if result.plan is not None:
                self.guard_plans.setdefault(kernel.name, []).append(result.plan)
            if result.preparations:
                self.preparations.setdefault(kernel.name, []).extend(
                    result.preparations
                )
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def accept(
        self,
        domain: str,
        rows: NDArray[np.int64],
        candidates: tuple[str, ...],
        *,
        cost_skipped: tuple[str, ...] = (),
    ) -> None:
        """Try named candidates, exhausting cost-skipped allowed alternatives.

        domain names the structural domain for refusal diagnostics.
        candidates lists cheapest-first choices. cost_skipped lists candidates
        omitted only for cost, to try last when a family allowlist is supplied.
        Numerical guards and input preconditions remain mandatory.
        """
        if not len(rows):
            return
        if self.context.families is not None:
            candidates = tuple(
                name
                for name in dict.fromkeys((*candidates, *cost_skipped))
                if name in self.context.families
            )
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
        points = np.unique(self.geometry.points[self.table.timeline[rows]]).tolist()
        raise NoValidKernel(
            f"no valid kernel for segment {rows[0]}; domain {domain}: "
            f"points {points}, segments {rows.tolist()}",
            domain=domain,
            points=tuple(points),
            segments=tuple(int(row) for row in rows),
            attempted=candidates,
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
        spectral = self.kernels.get("shaped_sambe")
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
        pending_rows, order = plan.rows[pending], plan.order[pending]
        for candidates in np.unique(order, axis=0):
            selected = pending_rows[np.all(order == candidates, axis=1)]
            self.accept(
                "shaped",
                selected,
                tuple(str(name) for name in candidates),
                cost_skipped=("single_tone_gate", "shaped_sambe"),
            )

    def classify(self) -> Dispatch:
        rows = np.flatnonzero(self.scope)
        self.accept("idle", rows[self.domains["idle"][rows]], ("t4",))
        self.accept(
            "adiabatic", rows[self.domains["adiabatic"][rows]], ("adiabatic_map",)
        )
        self.accept("flux", rows[self.domains["flux"][rows]], ("flux_dop853",))
        self.classify_slow(rows[self.domains["slow"][rows]])
        # Group shaped and constant consumers together before any legacy prepare.
        common = self.domains["two"] | self.domains["multiple"]
        self.try_common_frame(rows[common[rows]])
        self.classify_plateau(rows[self.domains["plateau"][rows]])
        shaped = np.flatnonzero(self.domains["shaped"] & self.scope)
        if len(shaped):
            self.classify_shaped(shaped)
        two_tone = np.flatnonzero(self.domains["two"])
        if len(two_tone):
            self.classify_two_tone(two_tone)
        self.classify_multiple(rows[self.domains["multiple"][rows]])
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
            if np.any(preferred):
                self.try_kernel(kernel, candidates[preferred])
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def expanded_direct(self, rows: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Borrow the cold comparison only when both families are available."""
        if (
            not len(rows)
            or not isinstance(self.kernels.get("multitone_rwa"), MultitoneRWAKernel)
            or ("dop853" not in self.kernels)
        ):
            return np.zeros(len(rows), dtype=bool)
        direct = populate_multitone_plan(
            self.table, self.source, self.context.work_plan, self.budget
        )
        return np.array([int(row) in direct for row in rows], dtype=bool)

    def classify_multiple(self, rows: NDArray[np.int64]) -> None:
        """Try reusable gates, then the cheaper admitted RWA/direct alternative."""
        rows = rows[(self.names[rows] == "") & ~self.deferred[rows]]
        gate = self.kernels.get("multitone_gate")
        if gate is not None and len(rows):
            preferred = gate_preferred(self.table, rows, self.context)
            self.try_kernel(gate, rows[preferred])
            rows = rows[(self.names[rows] == "") & ~self.deferred[rows]]
        preferred = self.expanded_direct(rows)
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
        expanded = self.expanded_direct(selected)
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
            {name: tuple(items) for name, items in self.preparations.items()},
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
    """Guard shared candidates now and resolve local candidates in their worker.

    Eight or more shared prepare keys may use a borrowed Engine threading pool.
    Smaller shared selections stay serial. Selection resolves one timeline's
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
    classifier = _Classifier(geometry, kernels, context, pool, selection, query_trace)
    classifier.capture = trace if isinstance(trace, CaptureRound) else None
    try:
        return classifier.classify()
    finally:
        if classifier.capture is not None:
            classifier.capture.finish(classifier.names)
