"""Cold prices for omission-expanded multitone RWA and full-drive integration."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel, KernelContext, SegmentTable
from .kernels._multitone_gate_phase import PHASE_GRID, PHASE_RTOL
from .kernels._rwa_gate import GATE_POLICY, canonical_keys
from .kernels._rwa_numeric import multitone_retention
from .kernels.multitone_rwa import MultitoneRWAKernel
from .secular import ModelSource
from .workplan import FactKey, WorkPlan

_GATE_INVENTORY = FactKey(("multitone-gate-inventory", GATE_POLICY), tuple)
# Full M=16 constant-pair preparation measured 0.24--0.28 ms. This reserve
# covers endpoint construction; the two dense applications stay in base work.
CONSTANT_KICK_PREPARE_S = 0.0003


def populate_gate_inventory(
    table: SegmentTable, source: ModelSource, plan: WorkPlan
) -> tuple[tuple[bytes, int], ...]:
    """Publish complete-acquire canonical gate counts before numerical guards.

    Return sorted (physics key, row count) pairs for sampled multitone rows.
    Prefix/worker subsets borrow these fixed counts; density and resident cache
    warmth do not participate. Counts determine cost, not numerical validity.
    """

    def build() -> tuple[tuple[bytes, int], ...]:
        rows = np.flatnonzero(
            (table.tones >= 2)
            & ~table.has_collapse_operators
            & ~table.constant_envelopes(np.arange(len(table.timeline)))
        )
        if not len(rows):
            return ()
        canonical = canonical_keys(table, KernelContext(source, 1e-3, plan))
        counts = Counter(canonical[row] for row in rows)
        return tuple(sorted(counts.items()))

    return plan.fact(_GATE_INVENTORY, build)


def gate_preferred(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    competitors: tuple[str, ...],
    diverted: Counter[bytes] | None = None,
) -> NDArray[np.bool_]:
    """Compare cold gate prices before building any phase coefficients.

    Only complete current-acquire canonical inventory counts amortize gate
    construction; the gate price never reads RoutingHistory.
    Eligible RWA retains its source-history cold price. An absent competitor
    cannot veto a pinned gate; the structural guard still checks eligibility.
    """
    plan = context.work_plan
    geometry = table if plan.complete_table is None else plan.complete_table
    counts = dict(populate_gate_inventory(geometry, context.source, plan))
    canonical = canonical_keys(table, context)
    rwa = (
        populate_multitone_inventory(geometry, context.source, plan, context.budget)
        if "multitone_rwa" in competitors
        else MultitoneInventory((), (), ())
    )
    rwa_uses = dict(rwa.uses)
    # RWA inventory rows refer to the complete table, even inside a worker.
    rwa_rows = {canonical_keys(geometry, context)[row]: row for row in rwa.admitted}
    preferred = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        key = canonical[row]
        uses = counts.get(key, 0) - (0 if diverted is None else diverted[key])
        if uses <= 0:
            continue
        cost = gate_cost(table, int(row), context)
        alternative = cost.dop853_competitor if "dop853" in competitors else np.inf
        if canonical[row] in rwa_rows:
            representative = rwa_rows[canonical[row]]
            other = multitone_cost(geometry, representative, context)
            alternative = min(
                alternative,
                other.guard
                + other.prepare / rwa_uses.get(representative, 1)
                + other.apply,
            )
        preferred[index] = cost.guard + cost.prepare / uses + cost.apply < alternative
    return preferred


@dataclass(frozen=True)
class GateCost:
    """Cold gate terms in estimated CPU seconds, independent of use count.

    direct is the uncalibrated full-drive work scale. dop853_competitor is
    the calibrated CPU alternative for this comparison only. endpoints and fourier
    are one key's build; guard (key/row bookkeeping) and apply are per use.
    """

    direct: float
    dop853_competitor: float
    endpoints: float
    fourier: float
    guard: float
    apply: float

    @property
    def prepare(self) -> float:
        """Return the combined estimate for one cold key."""
        return self.endpoints + self.fourier


def gate_cost(table: SegmentTable, row: int, context: KernelContext) -> GateCost:
    """Estimate one cold build and per-row work, once per key and budget.

    Gate preference amortizes the build over complete current-acquire uses.
    The estimate never reads time or RoutingHistory.
    """
    return context.work_plan.fact(
        FactKey(
            (
                "multitone-gate-cold-price",
                canonical_keys(table, context)[row],
                context.budget,
            ),
            GateCost,
        ),
        lambda: _price_gate(table, row, context),
    )


def _price_gate(table: SegmentTable, row: int, context: KernelContext) -> GateCost:
    model = table.model(context.source, row)
    size_ratio = len(model.energies_ghz) / 16
    scale = size_ratio**3
    nodes = int(np.prod(PHASE_GRID))
    direct = _direct_cost(table, row, context)
    # Price the current grid at its endpoint tolerance, independent of budget.
    baseline = direct / (1e-3 / context.budget) ** 0.125
    precision = (3e-5 / PHASE_RTOL) ** 0.125
    endpoints = baseline * nodes * 0.289 * precision
    fourier = float(2e-7 * 4 * nodes * np.log2(nodes) * size_ratio**2)
    apply = (2e-8 * 4 * nodes + 7.5e-6 * 4) * scale
    # Duration is a CPU pricing proxy, not a solver or frame selection rule.
    duration_ns = float(table.duration_ns[row])
    fraction = min(1.0, max(0.0, (duration_ns - 25.0) / 75.0))
    return GateCost(
        direct,
        (0.0891 + (0.412 - 0.0891) * fraction) * direct,
        endpoints,
        fourier,
        2e-6 * scale,
        apply,
    )


@dataclass(frozen=True)
class MultitoneInventory:
    """Complete admitted rows and expanded-row source usage, fixed before workers.

    admitted lists row indices of the complete Engine geometry, not a prefix
    slice. cheap lists admitted unexpanded small graphs. uses pairs an expanded
    row with its source-cumulative distinct-content row count. keys aligns with
    uses and names each row's source-history physics digest.
    Prices are evaluated lazily only when dispatch needs the legacy fallback.
    """

    admitted: tuple[int, ...]
    cheap: tuple[int, ...]
    uses: tuple[tuple[int, int], ...]
    keys: tuple[bytes, ...] = ()


def populate_multitone_inventory(
    table: SegmentTable, source: ModelSource, plan: WorkPlan, budget: float = 1e-3
) -> MultitoneInventory:
    """Record complete admission/usage once, independently of common selection.

    This preserves cumulative constant-pair fallback decisions for zero-cache
    and cached Engines. Sampled admission stays in the family guard and does
    not enter the source-cumulative history.
    """

    def build() -> MultitoneInventory:
        if budget < 1e-3:
            return MultitoneInventory((), (), ())
        context = KernelContext(source, budget, plan)
        kernel = MultitoneRWAKernel()
        pairs = np.flatnonzero((table.tones == 2) & ~table.has_collapse_operators)
        # Sampled guard work is lazy: never integrate probes to price constants
        # or to build caller-owned routing reuse facts.
        candidates = pairs[table.constant_envelopes(pairs)]
        admitted = candidates[kernel.guard(table, candidates, context)]
        expanded, cheap = [], []
        for row in admitted:
            retained = multitone_retention(table, int(row), context)
            if any(retained.edges):
                expanded.append(int(row))
            elif cheap_rwa_graph(table, int(row), context):
                cheap.append(int(row))
        rows = np.array(expanded, dtype=np.int64)
        physics = kernel.prepare_keys(table, rows, context) if len(rows) else ()
        keys = []
        inventory: dict[bytes, int] = {}
        for row, key in zip(rows, physics, strict=True):
            model = table.model(source, row)
            identity = sha256(
                b"multitone-cold" + model.identity.encode() + key.tobytes()
            ).digest()
            keys.append(identity)
            inventory[identity] = inventory.get(identity, 0) + 1
        history = source.history
        usage: dict[bytes, int] = {}
        if inventory:
            usage = (
                history.record_gate_usage(inventory, content=plan.content_key)
                if plan.history_mode == "record"
                else history.projected_gate_usage(inventory, content=plan.content_key)
            )
        return MultitoneInventory(
            tuple(int(row) for row in admitted),
            tuple(cheap),
            tuple((int(row), usage[key]) for row, key in zip(rows, keys, strict=True)),
            tuple(keys),
        )

    return plan.fact(
        FactKey(("multitone-cold-inventory", budget), MultitoneInventory), build
    )


def cheap_rwa_graph(table: SegmentTable, row: int, context: KernelContext) -> bool:
    """Classify an already-admitted unexpanded small graph for cold ordering.

    Complete inventories and prefix/worker subsets share this single predicate.
    This is a price shortcut, not a numerical admission rule.
    """
    retained = multitone_retention(table, row, context)
    return (
        retained.dimension is not None
        and retained.dimension
        <= max(9, 2 * len(table.model(context.source, row).energies_ghz))
        and not any(retained.edges)
    )


def populate_multitone_plan(
    table: SegmentTable, source: ModelSource, plan: WorkPlan, budget: float = 1e-3
) -> frozenset[int]:
    """Return global rows whose expanded RWA costs more than direct DOP853.

    Uses the complete acquire once, before prefix and worker subsets. Numerical
    admission remains the kernel's responsibility. Only admitted rows that
    restore omitted edges need a comparison; unexpanded RWA retains its order.
    Prepare work is amortized over source-cumulative exact physics-key usage,
    including this inventory. No resident-cache state or timing probes enter
    the decision. The frozen row selection is shared by all dispatch stages.

    Calibration uses the 16-level half-flux close-carrier mux model, 256--512 ns
    constant/Gaussian rows at pi and twice-pi gains. Expanded dimensions 174/256
    and fastest restored frequencies 7.84/8.66 GHz measured RWA apply around
    0.33--0.41/0.85--0.93 CPU-s. The 60 us per ns-GHz coefficient leaves headroom
    over these measured coefficients, since thermal/native and pure-state
    inventories have different adaptive-step costs. Guard allowance is 3 ms
    per occupied block unit, prepare 2 ms amortized per exact key. Direct uses
    the existing two-tone 80 us bandwidth/action model, with a 0.7 factor for
    sampled envelopes (measured Gaussian/constant direct ratio 0.65--0.69).
    Full constant-pair calibration adds a 0.30 ms M=16 endpoint-prepare
    reserve, with cubic size scaling. Restored-edge apply retains the 60 us per
    ns-GHz headroom and adds 3 us per ns for step-ceiling work, before action/occupancy/size
    scaling. The fixed M=16 feature star spans dimensions 22--40, 32--512 ns
    and restored bandwidths .075--.1 GHz; a retained .06-GHz-detuning control
    restored counterrotating edges at 2.065 GHz. These cold prices are
    scheduling estimates, not universal runtime bounds.
    """

    def build() -> frozenset[int]:
        complete = plan.complete_table
        geometry = table if complete is None else complete
        inventory = populate_multitone_inventory(geometry, source, plan, budget)
        context = KernelContext(source, budget, plan)
        return frozenset(
            row
            for row, uses in inventory.uses
            if multitone_cost(geometry, row, context).direct_cheaper(uses)
        )

    return plan.fact(FactKey(("multitone-cold-plan", budget), frozenset), build)


def expanded_direct(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    competitors: Mapping[str, Kernel],
) -> NDArray[np.bool_]:
    """Use the cold expanded-RWA comparison only if both families are available."""
    if (
        not len(rows)
        or not isinstance(competitors.get("multitone_rwa"), MultitoneRWAKernel)
        or "dop853" not in competitors
    ):
        return np.zeros(len(rows), dtype=bool)
    direct = populate_multitone_plan(
        table, context.source, context.work_plan, context.budget
    )
    return np.array([int(row) in direct for row in rows], dtype=bool)


@dataclass(frozen=True)
class MultitoneCost:
    """Usage-independent prices for one row's gate or expanded-RWA candidate.

    All fields are finite nonnegative CPU-second estimates, not measurements
    or runtime guarantees. direct is one full-drive density evolution of the
    same row. prepare is one cold construction of this candidate's reusable
    product. guard and apply are this candidate's per-use admission and playback
    work. Amortizing prepare requires a positive integer use count.
    """

    direct: float
    prepare: float
    guard: float
    apply: float

    def direct_cheaper(self, uses: int) -> bool:
        """Compare direct against cold candidate work for positive integer uses."""
        return self.direct < self.guard + self.prepare / uses + self.apply


def multitone_cost(
    table: SegmentTable, row: int, context: KernelContext
) -> MultitoneCost:
    """Borrow one row's immutable pricing, retaining no numerical payloads."""
    return context.work_plan.fact(
        FactKey(("multitone-row-cost", id(table), row, context.budget), MultitoneCost),
        lambda: _price_row(table, row, context),
    )


def _direct_cost(table: SegmentTable, row: int, context: KernelContext) -> float:
    """Raw full-drive work scale for gate construction and the RWA comparison."""
    model = table.model(context.source, row)
    size = len(model.energies_ghz)
    scale = (size / 16) ** 3
    count = int(table.tones[row])
    frequencies = table.tone_frequency_ghz[row, :count]
    strength = sum(
        abs(table.tone_amplitude[row, tone])
        * (
            np.linalg.norm(model.drives[port].in_phase_ghz, 2)
            + np.linalg.norm(model.drives[port].quadrature_ghz, 2)
        )
        for tone, port in enumerate(table.tone_port[row, :count])
    )
    duration = table.duration_ns[row]
    bandwidth = float(np.ptp(model.energies_ghz)) + max(abs(frequencies)) + 2 * strength
    action = 1 + 4 * np.sqrt(strength) + 4 * strength
    tolerance = (1e-3 / context.budget) ** 0.125
    sampled = any(
        table.envelopes[e].kind != "constant" for e in table.tone_envelope[row, :count]
    )
    return float(
        8e-5
        * duration
        * bandwidth
        * action
        * scale
        * tolerance
        * (0.7 if sampled else 1.0)
    )


def _price_row(table: SegmentTable, row: int, context: KernelContext) -> MultitoneCost:
    model = table.model(context.source, row)
    retained = multitone_retention(table, row, context)
    assert retained.dimension is not None
    size = len(model.energies_ghz)
    scale = (size / 16) ** 3
    count = int(table.tones[row])
    frequencies = table.tone_frequency_ghz[row, :count]
    strength = sum(
        abs(table.tone_amplitude[row, tone])
        * (
            np.linalg.norm(model.drives[port].in_phase_ghz, 2)
            + np.linalg.norm(model.drives[port].quadrature_ghz, 2)
        )
        for tone, port in enumerate(table.tone_port[row, :count])
    )
    duration = table.duration_ns[row]
    action = 1 + 4 * np.sqrt(strength) + 4 * strength
    tolerance = (1e-3 / context.budget) ** 0.125
    restored_frequency = max(
        abs(model.energies_ghz[i] - model.energies_ghz[j] + frequency)
        for frequency, edges in zip(frequencies, retained.edges, strict=True)
        for i, j in edges
    )
    occupancy = retained.dimension / size**2
    # Expanded graphs incur fast interaction-picture oscillations and additional
    # coefficient/guard work even when their omission certificate is excellent.
    # Full constant-pair preparation measured 0.24--0.28 ms at M=16.
    # Reserve 0.30 ms for the endpoint exponentials on top of the legacy
    # allowance. The two dense endpoint applications already belong to base W.
    prepare = 0.002 * scale * (1 + occupancy) + CONSTANT_KICK_PREPARE_S * scale
    guard = 0.003 * scale * (1 + occupancy)
    apply = (
        duration
        * (6e-5 * max(0.05, restored_frequency) + 3e-6)
        * action
        * (1 + occupancy)
        * scale
        * tolerance
    )
    return MultitoneCost(
        _direct_cost(table, row, context), float(prepare), float(guard), float(apply)
    )
