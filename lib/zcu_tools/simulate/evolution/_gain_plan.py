"""Complete-table CF4 coordinates, fixed dependencies and anchor admission."""

from bisect import bisect_left
from collections.abc import Hashable, Iterable

import numpy as np
from numpy.typing import NDArray

from .kernel import KernelContext, SegmentTable
from .kernels._cf4_grid import coarse_count, finer_count, spectral_width
from .kernels._cf4_spectral import CF4Grids, CF4Nodes, carrier_grids
from .kernels.floquet_period import PeriodAnchor, PeriodBuilder
from .secular import ModelIdentity, ModelSource, SecularModel
from .workplan import (
    CF4CarrierGroupPlan,
    FactKey,
    GainBuildPlan,
    GainGroupPlan,
    ResourceHandle,
    WorkPlan,
)


def period_builder(
    model: SecularModel, port: int, frequency: float, gain: float
) -> PeriodBuilder:
    """Select one constant-tone builder and retain its CF4 work ceiling.

    Carriers at or above 1 GHz use dense DOP853. Below that, calibrated
    cutoff-2/3 Sambe eigensolves cost 234 N-cubed units and must beat CF4's
    coarse plus first-fine estimate. A declined Sambe numerical certificate
    uses independent CF4, never gain anchors. No density or scheduling enters.
    """
    if abs(frequency) >= 1.0:
        return PeriodBuilder("dop853")
    drive = model.drives[port]
    width = spectral_width(
        model.energies_ghz, drive.in_phase_ghz, drive.quadrature_ghz, gain
    )
    coarse = coarse_count(1.5 * width / abs(frequency))
    work = float(coarse + finer_count(coarse))
    return PeriodBuilder("sambe" if work > 234.0 else "cf4", work)


def period_gain_coordinates(
    table: SegmentTable,
) -> dict[tuple[ModelIdentity, int, float], tuple[float, ...]]:
    """Collect complete-table low-carrier gain groups, before model selection.

    Only constant single tones with .03 <= |f| < 1 GHz participate; groups
    need at least two distinct gains before removing spectral candidates.
    """
    groups: dict[tuple[ModelIdentity, int, float], set[float]] = {}
    rows = np.flatnonzero(table.tones == 1)
    rows = rows[table.constant_envelopes(rows)]
    for row in rows:
        frequency = float(table.tone_frequency_ghz[row, 0])
        if not 0.03 <= abs(frequency) < 1.0:
            continue
        key = (
            ModelIdentity(tuple(int(v) for v in table.level[row]), table.shift(row)),
            int(table.tone_port[row, 0]),
            frequency,
        )
        groups.setdefault(key, set()).add(float(table.tone_amplitude[row, 0]))
    return {
        key: tuple(sorted(gains)) for key, gains in groups.items() if len(gains) >= 2
    }


def period_builders(
    model: SecularModel, port: int, frequency: float, gains: tuple[float, ...]
) -> dict[float, PeriodBuilder]:
    """Select each group key once and fix its non-spectral CF4 dependencies.

    gains is the complete sorted gain group, or a single standalone key.
    Every 24th CF4 gain and both endpoints are independent anchors. Other CF4
    keys use at most four nearest anchors, with lower gain winning distance
    ties. Sambe candidates never participate, even if their certificate fails.
    """
    builders = {gain: period_builder(model, port, frequency, gain) for gain in gains}
    ordered = [gain for gain in gains if builders[gain].kind == "cf4"]
    if len(gains) < 2 or not ordered:
        return builders
    anchors = ordered[::24]
    if anchors[-1] != ordered[-1]:
        anchors.append(ordered[-1])
    for gain in ordered:
        dependencies = (
            (gain,)
            if gain in anchors
            else tuple(sorted(sorted(anchors, key=lambda a: (abs(a - gain), a))[:4]))
        )
        builders[gain] = PeriodBuilder("cf4", builders[gain].work_limit, dependencies)
    return builders


_FULL_TABLE_ROWS = FactKey("gain-full-table-rows", int)


def plan_gain_groups(table: SegmentTable, source: ModelSource, plan: WorkPlan) -> None:
    """Publish gain and narrow CF4 carrier inventories before worker planning.

    Gain records cover multi-gain physical groups; carrier records cover narrow
    constant-CF4 inventories not owned by those gain groups. Builder selection
    and gain spacing retain their original rules. No periods build here.
    Prefix representatives are strict row subsets, so only a schedule
    with the complete row count may seal main resources. Content-preserving
    SegmentTable replacements (prefix inputs) still denote the full table.
    """
    plan.fact(_FULL_TABLE_ROWS, lambda: len(table.timeline))
    groups = []
    for (identity, port, frequency), gains in period_gain_coordinates(table).items():
        level = identity.physical_flux
        model = source.shifted_model(level, identity.shift_content)
        builders = period_builders(model, port, frequency, gains)
        groups.append(
            GainGroupPlan(
                model.identity,
                level,
                port,
                frequency,
                gains,
                tuple(
                    GainBuildPlan(gain, b.kind, b.work_limit, b.anchors)
                    for gain, b in builders.items()
                ),
                identity.shift_content,
            )
        )
    plan.publish_gain_groups(tuple(groups))
    _plan_cf4_carriers(table, source, plan)


def _plan_cf4_carriers(
    table: SegmentTable, source: ModelSource, plan: WorkPlan
) -> None:
    inventories: dict[tuple[ModelIdentity, int, float], set[float]] = {}
    rows = np.flatnonzero(table.tones == 1)
    rows = rows[table.constant_envelopes(rows)]
    gain_groups = {
        (ModelIdentity(g.level, g.shift_content), g.port, g.frequency_ghz)
        for g in plan.gain_groups
    }
    for row in rows:
        identity = ModelIdentity(
            tuple(int(v) for v in table.level[row]), table.shift(row)
        )
        port = int(table.tone_port[row, 0])
        frequency = float(table.tone_frequency_ghz[row, 0])
        # Keep gain sharing authoritative for every overlapping physical key.
        if (
            not 0.03 <= abs(frequency) < 1.0
            or (identity, port, frequency) in gain_groups
        ):
            continue
        gain = float(table.tone_amplitude[row, 0])
        inventories.setdefault((identity, port, gain), set()).add(frequency)
    groups = []
    for (identity, port, gain), frequencies in inventories.items():
        level = identity.physical_flux
        carriers = tuple(sorted(frequencies))
        magnitudes = np.abs(carriers)
        if (
            len(carriers) < 16
            or carriers[0] * carriers[-1] <= 0
            or max(magnitudes) > 1.01 * min(magnitudes)
        ):
            continue
        model = source.shifted_model(level, identity.shift_content)
        # Do not recruit spectral or high-carrier keys into CF4 dependencies.
        if any(period_builder(model, port, f, gain).kind != "cf4" for f in carriers):
            continue
        anchors = tuple(
            carriers[i]
            for i in np.linspace(0, len(carriers) - 1, 4).round().astype(int)
        )
        groups.append(
            CF4CarrierGroupPlan(
                model.identity,
                level,
                port,
                gain,
                carriers,
                anchors,
                identity.shift_content,
            )
        )
    plan.publish_cf4_carrier_groups(tuple(groups))


def carrier_build_plan(
    context: KernelContext,
    model: SecularModel,
    port: int,
    frequency: float,
    gain: float,
) -> CF4CarrierGroupPlan | None:
    """Find the fixed carrier dependencies for this exact physical CF4 key."""
    for group in context.work_plan.cf4_carrier_groups:
        if (group.model_identity, group.level, group.port, group.gain) == (
            model.identity,
            model.level,
            port,
            gain,
        ) and frequency in group.carriers_ghz:
            return group
    return None


def carrier_anchor_resource(
    context: KernelContext,
    group: CF4CarrierGroupPlan,
    frequency: float,
) -> ResourceHandle[PeriodAnchor]:
    """Borrow a carrier anchor with the same admission/handoff rules as gain anchors.

    The product includes IP prefixes, owned only by this evolve's resource;
    resident PeriodData never retains those extra arrays. Anchor keys consume
    only themselves; non-anchor keys consume all four supports.
    """
    return context.work_plan.resource(
        FactKey(
            (
                "cf4-carrier-anchor",
                group.model_identity,
                group.level,
                group.port,
                group.gain,
                context.budget,
                frequency,
            ),
            PeriodAnchor,
        ),
        planned_consumers=frozenset[Hashable](
            f
            for f in group.carriers_ghz
            if f == frequency or f not in group.anchors_ghz
        ),
    )


def carrier_node_resources(
    context: KernelContext, group: CF4CarrierGroupPlan
) -> dict[int, ResourceHandle[CF4Nodes]]:
    """Return single-flight coarse-node resources, partitioned by exact grid.

    Only non-anchor consumers participate. Model, gain, port, carrier sign and
    grid are physical identities; admission and handoff follow anchor lifetime.
    No numerical products build here or enter the resident cache.
    """

    def inventory() -> CF4Grids:
        model = context.source.shifted_model(group.level, group.shift_content)
        drive = model.drives[group.port]
        return carrier_grids(
            model.energies_ghz,
            group.gain * drive.in_phase_ghz,
            group.gain * drive.quadrature_ghz,
            tuple(f for f in group.carriers_ghz if f not in group.anchors_ghz),
        )

    # This WorkPlan's complete carrier inventory is fixed before any consumers.
    grids = context.work_plan.fact(
        FactKey(
            (
                "cf4-carrier-grids",
                group.model_identity,
                group.level,
                group.port,
                group.gain,
            ),
            CF4Grids,
        ),
        inventory,
    )
    return {
        count: context.work_plan.resource(
            FactKey(
                (
                    "cf4-carrier-nodes",
                    group.model_identity,
                    group.level,
                    group.port,
                    group.gain,
                    context.budget,
                    np.sign(group.carriers_ghz[0]),
                    count,
                ),
                CF4Nodes,
            ),
            planned_consumers=frozenset[Hashable](carriers),
        )
        for count, carriers in grids.by_count
    }


def admit_carrier_nodes(
    context: KernelContext, group: CF4CarrierGroupPlan, frequency: float
) -> ResourceHandle[CF4Nodes] | None:
    """Admit one actual non-anchor key to its grid's node resource.

    Anchor, released, sealed-out and already delivered consumers use the
    independent coarse guard. The caller must hand off every returned handle
    after its key build, even when incompatible anchors skip interpolation.
    """
    if frequency in group.anchors_ghz:
        return None
    for handle in carrier_node_resources(context, group).values():
        if frequency not in handle.planned_consumers or handle.released:
            continue
        if not handle.sealed and frequency not in handle.actual_consumers:
            handle.admit(frequency)
        if frequency in handle.remaining_consumers:
            return handle
    return None


def build_plan(
    context: KernelContext,
    model: SecularModel,
    port: int,
    frequency: float,
    gain: float,
) -> tuple[GainGroupPlan, PeriodBuilder] | None:
    """Find the fixed complete-table instructions, or None for standalone keys.

    Physical identity includes model content and level, signed carrier and port.
    No worker-local subset or previously prepared key changes the dependencies.
    """
    for group in context.work_plan.gain_groups:
        if (group.model_identity, group.level, group.port, group.frequency_ghz) != (
            model.identity,
            model.level,
            port,
            frequency,
        ):
            continue
        index = bisect_left(group.gains, gain)
        if index < len(group.gains) and group.gains[index] == gain:
            builder = group.builders[index]
            return group, PeriodBuilder(
                builder.kind, builder.work_limit, builder.anchors
            )
    return None


def gain_anchor_resource(
    context: KernelContext, group: GainGroupPlan, gain: float
) -> ResourceHandle[PeriodAnchor]:
    """Return the typed anchor handle with its complete planned gain consumers.

    Identity includes content, quantized coordinates, drive, carrier and budget.
    Callers admit only real key builds and hand off once their result is owned.
    No numerical construction occurs here.
    """
    return context.work_plan.resource(
        FactKey(
            (
                "gain-anchor",
                group.model_identity,
                group.level,
                group.port,
                group.frequency_ghz,
                context.budget,
                gain,
            ),
            PeriodAnchor,
        ),
        planned_consumers=frozenset[Hashable](
            b.gain for b in group.builders if gain in b.anchors
        ),
    )


def admit_shared_builds(
    context: KernelContext,
    table: SegmentTable,
    pending_rows: NDArray[np.int64],
    open_rows: NDArray[np.int64],
) -> None:
    """Admit cache-missing caller builds and seal caller-only anchor groups.

    pending_rows are representatives of actual shared Floquet-period build jobs,
    after resident lookup. open_rows are deferred or timeline-local candidates;
    matching groups stay open through evolve close for incremental admission.
    Prefix tables cannot close the complete table's admission. No numerical
    choices depend on cache warmth or on these actual consumers.
    """
    plan = context.work_plan
    if plan.fact(_FULL_TABLE_ROWS, lambda: len(table.timeline)) != len(table.timeline):
        return
    for group in plan.gain_groups:
        pending = _gains_at(table, pending_rows, group)
        stays_open = bool(_gains_at(table, open_rows, group))
        _admit_anchors(
            (
                gain_anchor_resource(context, group, coordinate)
                for coordinate in sorted({a for b in group.builders for a in b.anchors})
            ),
            pending,
            stays_open=stays_open,
        )
    for group in plan.cf4_carrier_groups:
        pending = _carriers_at(table, pending_rows, group)
        stays_open = bool(_carriers_at(table, open_rows, group))
        _admit_anchors(
            (
                carrier_anchor_resource(context, group, frequency)
                for frequency in group.anchors_ghz
            ),
            pending,
            stays_open=stays_open,
        )
        _admit_anchors(
            carrier_node_resources(context, group).values(),
            pending,
            stays_open=stays_open,
        )


def _admit_anchors[T](
    handles: Iterable[ResourceHandle[T]],
    pending: set[float],
    *,
    stays_open: bool,
) -> None:
    # Both physical axes have the same cache-miss admission and handoff lifetime.
    for handle in handles:
        if handle.released or handle.sealed:
            continue
        for consumer in pending.intersection(handle.planned_consumers):
            if consumer not in handle.actual_consumers:
                handle.admit(consumer)
        if not stays_open:
            handle.seal()


def _carriers_at(
    table: SegmentTable, rows: NDArray[np.int64], group: CF4CarrierGroupPlan
) -> set[float]:
    selected = rows[
        (table.tones[rows] == 1)
        & table.constant_envelopes(rows)
        & np.all(table.level[rows] == group.level, axis=1)
        & (table.tone_port[rows, 0] == group.port)
        & (table.shift_content[rows] == group.shift_content)
        & (table.tone_amplitude[rows, 0] == group.gain)
    ]
    return set(table.tone_frequency_ghz[selected, 0].tolist()).intersection(
        group.carriers_ghz
    )


def _gains_at(
    table: SegmentTable, rows: NDArray[np.int64], group: GainGroupPlan
) -> set[float]:
    selected = rows[
        (table.tones[rows] == 1)
        & table.constant_envelopes(rows)
        & np.all(table.level[rows] == group.level, axis=1)
        & (table.tone_port[rows, 0] == group.port)
        & (table.shift_content[rows] == group.shift_content)
        & (table.tone_frequency_ghz[rows, 0] == group.frequency_ghz)
    ]
    return set(table.tone_amplitude[selected, 0].tolist()).intersection(group.gains)
