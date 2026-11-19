"""Evolve-local sampled RWA prices, including serial full-drive spot checks."""

from collections.abc import Callable, Mapping
from functools import cache

import numpy as np
from numpy.typing import NDArray

from ._common_frame_dispatch import populate_common_frame_plan, preferred
from ._dop853_dispatch import (
    direct_plans,
    opencl_rows,
    opencl_wave_prices,
    two_tone_knot_counts,
)
from .kernel import KernelContext, SegmentTable
from .kernels._dop853 import ROW_INITIAL_STEP_NS
from .kernels._multitone_spot import SampledGroup, sampled_groups
from .kernels._rwa_fourier import fourier_coefficient
from .kernels._rwa_numeric import STEP_NS, multitone_retention
from .kernels.common_frame import zero_mean_carrier
from .kernels.dop853 import cpu_work_scale
from .kernels.multitone_rwa import MultitoneRWAKernel
from .workplan import FactKey


def _parallel_seconds(
    table: SegmentTable,
    rows: NDArray[np.int64],
    costs: NDArray[np.float64],
    workers: int,
) -> float:
    """Estimate elapsed work without parallelizing within a timeline."""
    if not len(rows):
        return 0.0
    # Complete-table rows are ordered by timeline. Only allocate for timelines
    # represented in the group, regardless of their global indices.
    starts = np.r_[0, np.flatnonzero(np.diff(table.timeline[rows])) + 1]
    totals = np.add.reduceat(costs, starts)
    return max(float(totals.sum()) / workers, float(totals.max()))


def _full_drive_bandwidth(
    table: SegmentTable, row: int, context: KernelContext
) -> float:
    model = table.model(context.source, row)
    key = FactKey(
        (
            "sampled-drive-bandwidth",
            model.identity,
            model.level,
            tuple(table.tone_port[row, :2]),
            tuple(table.tone_frequency_ghz[row, :2]),
        ),
        float,
    )

    def build() -> float:
        gaps = model.energies_ghz[:, None] - model.energies_ghz
        bandwidth = 0.0
        for port, carrier in zip(
            table.tone_port[row, :2], table.tone_frequency_ghz[row, :2], strict=True
        ):
            drive = model.drives[port]
            mask = fourier_coefficient(drive.in_phase_ghz, drive.quadrature_ghz, 1) != 0
            bandwidth = max(
                bandwidth, float(np.max(np.abs(gaps[mask] + carrier), initial=0))
            )
        return bandwidth

    return context.work_plan.fact(key, build)


def _direct_seconds(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.float64]:
    """Estimate serial DOP853 endpoint work using knot restarts and active bandwidth."""
    duration = table.duration_ns[rows]
    # Native knot count is a cheap upper estimate for clipped or offset controls.
    intervals = np.maximum(1, two_tone_knot_counts(table, rows) - 1)
    seed_ratio = np.maximum(1.0, duration / intervals / ROW_INITIAL_STEP_NS)
    restarts = intervals * (1.0 + np.ceil(np.log(seed_ratio) / np.log(4.0)))
    bandwidth = np.array(
        [_full_drive_bandwidth(table, int(row), context) for row in rows]
    )
    trials = np.maximum(restarts, 2 * duration * bandwidth)
    d = context.source.initial_density.shape[0]
    trial_seconds = 15e-6 + 12 * (8 * d**3 + 40 * d**2) / 17e9
    tolerance = (1e-3 / context.budget) ** 0.125
    return 25e-6 + trials * trial_seconds * tolerance


def _rwa_apply_seconds(table: SegmentTable, row: int, context: KernelContext) -> float:
    retained = multitone_retention(table, row, context)
    assert retained.dimension is not None
    model = table.model(context.source, row)
    bandwidth = max(
        (
            abs(model.energies_ghz[i] - model.energies_ghz[j] + frequency)
            for frequency, edges in zip(
                table.tone_frequency_ghz[row, :2], retained.edges, strict=True
            )
            for i, j in edges
        ),
        default=0.0,
    )
    steps = max(4.0, table.duration_ns[row] * max(1 / STEP_NS, 2 * bandwidth))
    tolerance = (1e-3 / context.budget) ** 0.125
    return float(steps * (10 + 0.4 * retained.dimension) * 1e-6 * tolerance)


def _rwa_seconds(
    table: SegmentTable,
    rows: NDArray[np.int64],
    direct: NDArray[np.float64],
    spot_rows: tuple[int, ...],
    context: KernelContext,
    *,
    total_cpu: bool = False,
) -> float:
    # Calibration: 3/4-level short and sparse/dense-knot rows plus the 24-level
    # simultaneous-drive scan. The spot is serial, uses its own coherent state,
    # and builds fresh payloads. Ordinary rows share preparation within the
    # current inventory but each pays a per-row allowance. Clock readings and
    # source history do not enter dispatch.
    d = context.source.initial_density.shape[0]
    apply = np.array([_rwa_apply_seconds(table, int(row), context) for row in rows])
    indices = np.searchsorted(rows, spot_rows)
    probe = float(np.sum(direct[indices] + apply[indices] + 260e-6 + 2.5e-6 * d))
    playback = apply + 65e-6 + 1e-6 * d
    elapsed = (
        float(playback.sum())
        if total_cpu
        else _parallel_seconds(table, rows, playback, context.workers)
    )
    return probe + elapsed


def sampled_rwa_cpu_seconds(
    table: SegmentTable, group: SampledGroup, context: KernelContext
) -> float:
    """Return one group's fixed CPU probes plus serial playback for all its rows."""
    rows = np.asarray(group.rows, dtype=np.int64)
    return _rwa_seconds(
        table,
        rows,
        _direct_seconds(table, rows, context),
        group.spot_rows,
        context,
        total_cpu=True,
    )


def sampled_common_prices(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    rwa: MultitoneRWAKernel | None,
) -> NDArray[np.float64]:
    """Return existing CPU prices for offered common frame; infinity otherwise."""
    plan = populate_common_frame_plan(table, context, price_rwa=rwa is not None)
    prices = {price.key: price.common for price in plan.prices}
    offered = preferred(table, rows, context, rwa)
    if context.families is not None and "dop853" not in context.families:
        # A whitelisted common frame is still a competitor after a cost skip.
        offered = np.ones_like(offered)
    estimates = np.full(len(rows), np.inf)
    for index in np.flatnonzero(offered):
        row = rows[index]
        if not zero_mean_carrier(table.tone_frequency_ghz[row, :2]):
            estimates[index] = prices[plan.keys[row].tobytes()]
    return estimates


def _gpu_seconds(
    table: SegmentTable,
    rows: NDArray[np.int64],
    direct: NDArray[np.float64],
    mask: NDArray[np.bool_],
    context: KernelContext,
    *,
    whole_group: bool = False,
) -> float:
    backend = context.opencl
    if backend is None:
        return np.inf
    dimension = context.source.initial_density.shape[0]
    remaining = np.ones(len(rows), dtype=bool)
    eligible = np.flatnonzero(mask[rows])
    timelines = table.timeline[rows[eligible]]
    _, first, counts = np.unique(timelines, return_index=True, return_counts=True)
    ordinal = np.arange(len(eligible)) - np.repeat(first, counts)
    gpu = 0.0
    for wave in np.unique(ordinal):
        indices = eligible[ordinal == wave]
        cpu_wave, gpu_wave = opencl_wave_prices(
            table,
            rows[indices],
            dimension,
            context.workers,
            budget=context.budget,
            dp54=context.opencl_dp54,
        )
        if backend.forced or gpu_wave < cpu_wave:
            gpu += gpu_wave
            remaining[indices] = False
    if np.all(remaining) or (whole_group and np.any(remaining)):
        return np.inf
    return gpu + _parallel_seconds(
        table, rows[remaining], direct[remaining], context.workers
    )


def _gpu_eligible_mask(
    table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> NDArray[np.bool_]:
    names = np.full(len(rows), "dop853")
    plans, _ = direct_plans(table, names, rows, budget=context.budget)
    dimension = context.source.initial_density.shape[0]
    mask = opencl_rows(table, plans.get("dop853"), dimension)
    return np.zeros(len(table.tones), dtype=bool) if mask is None else mask


def gpu_sampled_groups(
    table: SegmentTable,
    context: KernelContext,
    groups: Mapping[int, SampledGroup],
    direct_candidates: NDArray[np.int64],
) -> set[int]:
    """Return groups whose original direct wave prefers OpenCL, before device discovery.

    direct_candidates contains complete row IDs after excluding known
    non-direct neighbors. The same independent-endpoint chain, waves and
    CPU/GPU estimates as Step 9 apply. This does not check or compile a device.
    """
    mask = _gpu_eligible_mask(table, direct_candidates, context)
    selected = set()
    for index, group in groups.items():
        rows = np.asarray(group.rows, dtype=np.int64)
        if np.isfinite(_gpu_seconds(table, rows, np.zeros(len(rows)), mask, context)):
            selected.add(index)
    return selected


def _prefer(
    table: SegmentTable,
    group: SampledGroup,
    context: KernelContext,
    rwa: MultitoneRWAKernel,
    price_common: Callable[[SegmentTable, NDArray[np.int64]], NDArray[np.bool_]] | None,
    gpu_eligible: Callable[[], NDArray[np.bool_]],
) -> bool:
    if (
        price_common is None
        and context.families is not None
        and "dop853" not in context.families
    ):
        return True
    rows = np.asarray(group.rows, dtype=np.int64)
    probe_direct = _direct_seconds(table, rows, context)
    rwa_cost = _rwa_seconds(table, rows, probe_direct, group.spot_rows, context)
    direct = probe_direct * cpu_work_scale(table.duration_ns[rows], context.budget)
    allow_direct = context.families is None or "dop853" in context.families
    if not allow_direct:
        direct = np.full_like(direct, np.inf)
    cpu_cost = _parallel_seconds(table, rows, direct, context.workers)
    backend = context.opencl if allow_direct else None
    forced = backend is not None and backend.forced
    if not forced and rwa_cost >= cpu_cost:
        return False

    if price_common is not None:
        offers = sampled_common_prices(table, rows, context, rwa)
        best_case = _parallel_seconds(
            table, rows, np.minimum(direct, offers), context.workers
        )
        if best_case <= rwa_cost:
            contenders = np.flatnonzero(offers < direct)
            admitted = price_common(table, rows[contenders])
            direct[contenders[admitted]] = offers[contenders[admitted]]
            cpu_cost = _parallel_seconds(table, rows, direct, context.workers)
            if rwa_cost >= cpu_cost:
                return False

    gpu_cost = (
        _gpu_seconds(table, rows, direct, gpu_eligible(), context)
        if backend is not None
        else np.inf
    )
    if forced and np.isfinite(gpu_cost):
        return rwa_cost < gpu_cost
    if gpu_cost <= rwa_cost < cpu_cost and backend is not None:
        # The existing lazy device check may run a self-test. Never compile a
        # point variant to price a candidate that might not run.
        return (
            backend.candidate(
                context.source.initial_density.shape[0], dp54=context.opencl_dp54
            )
            is None
        )
    return rwa_cost < cpu_cost


def confirmed_gpu_sampled_groups(
    table: SegmentTable,
    context: KernelContext,
    groups: Mapping[int, SampledGroup],
    direct_candidates: NDArray[np.int64],
    rwa: MultitoneRWAKernel,
) -> set[int]:
    """Quote groups whose every wave goes to OpenCL under Step 9 prices.

    The caller supplies groups with no admitted common-frame row and a
    superset of all possible suffix direct neighbors. A chain in this
    superset is ambiguous and cannot qualify. No RWA probe or routing
    fact is recorded. Check the device only for a decisive price quote.
    """
    backend = context.opencl
    if backend is None or not groups:
        return set()
    eligible = _gpu_eligible_mask(table, direct_candidates, context)
    selected = set()
    for index, group in groups.items():
        rows = np.asarray(group.rows, dtype=np.int64)
        gpu = _gpu_seconds(
            table, rows, np.zeros(len(rows)), eligible, context, whole_group=True
        )
        if np.isfinite(gpu) and not _prefer(
            table, group, context, rwa, None, lambda: eligible
        ):
            selected.add(index)
    dimension = context.source.initial_density.shape[0]
    if selected and backend.candidate(dimension, dp54=context.opencl_dp54) is not None:
        return selected
    return set()


def prefer_sampled_rwa(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    rwa: MultitoneRWAKernel,
    price_common: Callable[[SegmentTable, NDArray[np.int64]], NDArray[np.bool_]] | None,
    direct_candidates: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Return row-aligned preferences for proxy-eligible complete-table rows.

    Each group pays for its own playback and fresh CPU probes before comparing
    with CPU/GPU DOP853 and admitted common frame. A whitelist without either
    competitor keeps RWA eligible; the caller still runs its numerical guard.
    price_common guards contenders and can hand off coefficients to the caller.
    direct_candidates contains the complete-table rows still able to form
    DOP853 chains, treating unresolved drives as direct without running guards.
    Estimates and preferences expire with context.work_plan.
    """
    admission = sampled_groups(table, context)
    membership = admission.by_row[rows]
    preferred = np.zeros(len(admission.groups), dtype=bool)

    @cache
    def gpu_eligible() -> NDArray[np.bool_]:
        # Share eligibility across groups in this routing round only.
        return _gpu_eligible_mask(table, direct_candidates, context)

    def compare(group: SampledGroup) -> bool:
        wins = _prefer(table, group, context, rwa, price_common, gpu_eligible)
        if context.capture is not None:
            context.capture.sampled_group(
                group, "not_probed" if wins else "cost_skipped"
            )
        return wins

    for index in np.unique(membership):
        group = admission.groups[index]
        preferred[index] = context.work_plan.fact(
            FactKey(
                (
                    "sampled-rwa-preference",
                    context.budget,
                    group.rows[0],
                    context.opencl is not None,
                ),
                bool,
            ),
            lambda group=group: compare(group),
        )
    return preferred[membership]
