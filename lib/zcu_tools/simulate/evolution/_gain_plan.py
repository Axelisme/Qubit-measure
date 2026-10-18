"""Complete-table gain coordinates, fixed dependencies and anchor admission."""

from bisect import bisect_left
from collections.abc import Hashable

import numpy as np
from numpy.typing import NDArray

from .kernel import KernelContext, SegmentTable
from .kernels._cf4_grid import coarse_count, finer_count, spectral_width
from .kernels.floquet_period import PeriodBuilder, PeriodData
from .secular import ModelSource, SecularModel
from .workplan import FactKey, GainBuildPlan, GainGroupPlan, ResourceHandle, WorkPlan


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
) -> dict[tuple[tuple[int, ...], int, float], tuple[float, ...]]:
    """Collect complete-table low-carrier gain groups, before model selection.

    Only constant single tones with .03 <= |f| < 1 GHz participate; groups
    need at least two distinct gains before removing spectral candidates.
    """
    groups: dict[tuple[tuple[int, ...], int, float], set[float]] = {}
    rows = np.flatnonzero(table.tones == 1)
    rows = rows[table.constant_envelopes(rows)]
    for row in rows:
        frequency = float(table.tone_frequency_ghz[row, 0])
        if not 0.03 <= abs(frequency) < 1.0:
            continue
        key = (
            tuple(int(v) for v in table.level[row]),
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
    """Publish complete low-carrier inventories before prefix/worker planning.

    Only multi-gain physical groups need records. Builder selection and anchor
    spacing retain the original cost and interpolation rules. No periods build
    here. Prefix representatives are strict row subsets, so only a schedule
    with the complete row count may seal main resources. Content-preserving
    SegmentTable replacements (prefix inputs) still denote the full table.
    """
    plan.fact(_FULL_TABLE_ROWS, lambda: len(table.timeline))
    groups = []
    for (level, port, frequency), gains in period_gain_coordinates(table).items():
        model = source.model(level)
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
            )
        )
    plan.publish_gain_groups(tuple(groups))


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
) -> ResourceHandle[PeriodData]:
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
            PeriodData,
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
    if not plan.gain_groups or plan.fact(
        _FULL_TABLE_ROWS, lambda: len(table.timeline)
    ) != len(table.timeline):
        return
    for group in plan.gain_groups:
        pending = _gains_at(table, pending_rows, group)
        stays_open = bool(_gains_at(table, open_rows, group))
        for coordinate in sorted({a for b in group.builders for a in b.anchors}):
            handle = gain_anchor_resource(context, group, coordinate)
            if handle.released or handle.sealed:
                continue
            for gain in pending.intersection(handle.planned_consumers):
                if gain not in handle.actual_consumers:
                    handle.admit(gain)
            if not stays_open:
                handle.seal()


def _gains_at(
    table: SegmentTable, rows: NDArray[np.int64], group: GainGroupPlan
) -> set[float]:
    selected = rows[
        (table.tones[rows] == 1)
        & table.constant_envelopes(rows)
        & np.all(table.level[rows] == group.level, axis=1)
        & (table.tone_port[rows, 0] == group.port)
        & (table.tone_frequency_ghz[rows, 0] == group.frequency_ghz)
    ]
    return set(table.tone_amplitude[selected, 0].tolist()).intersection(group.gains)
