"""Evolve-local sampled RWA prices, including serial full-drive spot checks."""

import numpy as np
from numpy.typing import NDArray

from ._common_frame_dispatch import populate_common_frame_plan, preferred
from ._dop853_dispatch import (
    direct_plans,
    opencl_rows,
    opencl_wave_prices,
    two_tone_knot_counts,
)
from .kernel import Kernel, KernelContext, SegmentTable
from .kernels._dop853 import ROW_INITIAL_STEP_NS
from .kernels._multitone_spot import sampled_candidates, spot_rows
from .kernels._rwa_fourier import fourier_coefficient
from .kernels._rwa_numeric import STEP_NS, multitone_retention
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
    totals = np.bincount(table.timeline[rows], weights=costs)
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
    context: KernelContext,
) -> float:
    # Calibration: 3/4-level short and sparse/dense-knot rows plus the 24-level
    # simultaneous-drive scan. The spot is serial, uses its own coherent state,
    # and builds fresh payloads. Ordinary rows share preparation within the
    # current inventory but each pays a per-row allowance. Clock readings and
    # source history do not enter dispatch.
    d = context.source.initial_density.shape[0]
    apply = np.array([_rwa_apply_seconds(table, int(row), context) for row in rows])
    indices = np.searchsorted(rows, spot_rows(table, tuple(rows.tolist()), context))
    probe = float(np.sum(direct[indices] + apply[indices] + 260e-6 + 2.5e-6 * d))
    playback = _parallel_seconds(table, rows, apply + 65e-6 + 1e-6 * d, context.workers)
    return probe + playback


def _common_offers(
    table: SegmentTable,
    rows: NDArray[np.int64],
    context: KernelContext,
    rwa: MultitoneRWAKernel,
) -> NDArray[np.float64]:
    """Borrow common frame's existing complete-group prices and offer policy."""
    plan = populate_common_frame_plan(table, context, price_rwa=True)
    prices = {price.key: price.common for price in plan.prices}
    offered = preferred(table, rows, context, rwa)
    if context.families is not None and "dop853" not in context.families:
        # A whitelisted common frame is still a competitor after a cost skip.
        offered = np.ones_like(offered)
    estimates = np.full(len(rows), np.inf)
    for index in np.flatnonzero(offered):
        estimates[index] = prices[plan.keys[rows[index]].tobytes()]
    return estimates


def _gpu_seconds(
    table: SegmentTable,
    rows: NDArray[np.int64],
    direct: NDArray[np.float64],
    context: KernelContext,
) -> float:
    backend = context.opencl
    if backend is None:
        return np.inf
    dimension = context.source.initial_density.shape[0]
    # Price only endpoints that can still be independent if adjacent drives
    # fall back to direct. Execution will recheck the final route and wave.
    names = np.where(table.tones > 0, "dop853", "")
    plans, _ = direct_plans(table, names)
    mask = opencl_rows(table, plans.get("dop853"), dimension)
    if mask is None:
        return np.inf
    remaining = np.ones(len(rows), dtype=bool)
    eligible = np.flatnonzero(mask[rows])
    timelines = table.timeline[rows[eligible]]
    _, first, counts = np.unique(timelines, return_index=True, return_counts=True)
    ordinal = np.arange(len(eligible)) - np.repeat(first, counts)
    gpu = 0.0
    for wave in np.unique(ordinal):
        indices = eligible[ordinal == wave]
        cpu_wave, gpu_wave = opencl_wave_prices(
            table, rows[indices], dimension, context.workers
        )
        if backend.forced or gpu_wave < cpu_wave:
            gpu += gpu_wave
            remaining[indices] = False
    if np.all(remaining):
        return np.inf
    return gpu + _parallel_seconds(
        table, rows[remaining], direct[remaining], context.workers
    )


def _prefer(
    table: SegmentTable,
    context: KernelContext,
    rwa: MultitoneRWAKernel,
    common: Kernel | None,
) -> bool:
    rows = np.array(sampled_candidates(table, context), dtype=np.int64)
    direct = _direct_seconds(table, rows, context)
    rwa_cost = _rwa_seconds(table, rows, direct, context)
    allow_direct = context.families is None or "dop853" in context.families
    if not allow_direct:
        direct = np.full_like(direct, np.inf)
    cpu_cost = _parallel_seconds(table, rows, direct, context.workers)
    backend = context.opencl if allow_direct else None
    forced = backend is not None and backend.forced
    if not forced and rwa_cost >= cpu_cost:
        return False

    if common is not None:
        offers = _common_offers(table, rows, context, rwa)
        best_case = _parallel_seconds(
            table, rows, np.minimum(direct, offers), context.workers
        )
        if best_case <= rwa_cost:
            contenders = np.flatnonzero(offers < direct)
            with context.work_plan.routing.measure(common.name, "prepare"):
                admitted = common.guard(table, rows[contenders], context)
            direct[contenders[admitted]] = offers[contenders[admitted]]
            cpu_cost = _parallel_seconds(table, rows, direct, context.workers)
            if rwa_cost >= cpu_cost:
                return False

    gpu_cost = (
        _gpu_seconds(table, rows, direct, context) if backend is not None else np.inf
    )
    if forced and np.isfinite(gpu_cost):
        return rwa_cost < gpu_cost
    if gpu_cost <= rwa_cost < cpu_cost and backend is not None:
        # The existing lazy device check may run a self-test. Never compile a
        # point variant to price a candidate that might not run.
        return backend.candidate(context.source.initial_density.shape[0]) is None
    return rwa_cost < cpu_cost


def prefer_sampled_rwa(
    table: SegmentTable,
    context: KernelContext,
    rwa: MultitoneRWAKernel,
    common: Kernel | None,
) -> bool:
    """Price the complete nonempty sampled candidate group before its guard.

    table is the complete geometry, not a prefix or worker slice. The fallback
    is CPU/GPU DOP853 and/or the allowed common frame. An explicit whitelist
    without either competitor skips pricing, but still runs the RWA guard.
    """
    if (
        common is None
        and context.families is not None
        and "dop853" not in context.families
    ):
        return True
    return context.work_plan.fact(
        FactKey(("sampled-rwa-preference", context.budget), bool),
        lambda: _prefer(table, context, rwa, common),
    )
