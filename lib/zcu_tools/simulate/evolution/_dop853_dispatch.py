"""Timeline-wide frame and chain decisions for the direct solvers."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .kernel import KernelContext, RowPlan, SegmentTable
from .kernels._dop853 import DP54_MIN_BUDGET
from .kernels._dop853_opencl import ROWS_PER_BATCH, OpenCLBackend
from .kernels.dop853 import DOP853Plan, cpu_dp54_rows
from .kernels.flux_dop853 import FluxDOP853Plan

_MAX_OPENCL_DIMENSION = 32
_MAX_OPENCL_KNOTS = 960
_MIN_OPENCL_DURATION_NS = 100.0
_MAX_OPENCL_DURATION_NS = 10_000.0
# Arc d=24/d=32 simultaneous-drive medians were 0.68/0.61 of DOP853 wall.
# Use the slower factor for matrix work, not for launch or per-trial floors.
_DP54_GPU_WORK_SCALE = 0.68


def with_opencl(context: KernelContext, backend: OpenCLBackend | None) -> KernelContext:
    """Attach the optional timeline backend and choose its evolve-wide tableau.

    OpenCL admission already caps every eligible row at 10 µs. Only the
    budget distinguishes the tableaus on that backend (ADR-0078).
    """
    return replace(
        context,
        opencl=backend,
        opencl_dp54=backend is not None and context.budget >= DP54_MIN_BUDGET,
    )


def opencl_rows(
    table: SegmentTable, plan: RowPlan | None, dimension: int
) -> NDArray[np.bool_] | None:
    """Mark global DOP853 rows that can run as independent OpenCL endpoints.

    Supply the completed direct plan for this acquire or resolved timeline.
    Return a read-only mask, or None for unsupported models or no DOP853 plan.
    Admission requires at most 32 levels, no explicit collapse operators,
    100–10000 ns of full-support sampled playback and at most 960 distinct
    sample knots across both tones.
    """
    if (
        not isinstance(plan, DOP853Plan)
        or dimension > _MAX_OPENCL_DIMENSION
        or not len(table.complete_pulses)
    ):
        return None
    eligible = (
        (plan.frame == "interaction")
        & ~table.has_collapse_operators
        & table.complete_pulses
        & (table.tones == 2)
        & (table.duration_ns >= _MIN_OPENCL_DURATION_NS)
        & (table.duration_ns <= _MAX_OPENCL_DURATION_NS)
        & np.all(table.level == 0, axis=1)
        & (table.shift_content == b"")
    )
    if table.flux_waveform is not None:
        available = table.flux_waveform_available
        eligible &= False if available is None else ~available[table.timeline]
    rows = np.flatnonzero(eligible)
    if len(rows):
        sampled = np.array([env.kind == "sampled" for env in table.envelopes])
        both_sampled = np.all(sampled[table.tone_envelope[rows, :2]], axis=1)
        eligible[rows] = False
        rows = rows[both_sampled]
        # Each knot in the union can restart the integrator. The runner can
        # append the row end, giving at most one interval per sample knot.
        eligible[rows] = two_tone_knot_counts(table, rows) <= _MAX_OPENCL_KNOTS
    eligible.setflags(write=False)
    return eligible


def two_tone_knot_counts(
    table: SegmentTable, rows: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Count distinct native-grid knots for each two-tone row in rows."""
    pairs, inverse = np.unique(
        table.tone_envelope[rows, :2], axis=0, return_inverse=True
    )
    counts = np.array(
        [
            len(np.union1d(table.envelopes[a].sample_ns, table.envelopes[b].sample_ns))
            for a, b in pairs
        ],
        dtype=np.int64,
    )
    return counts[inverse]


def _batch_gpu_seconds(
    work: NDArray[np.float64], trials: NDArray[np.float64], knots: int, dimension: int
) -> float:
    # One group per row: longer rows stay on device even after peers finish.
    # Arc measured 367 GFLOP/s for 24 d=32 rows and about 22 for a single row.
    # The 20 us trial floor covers smaller d=3 groups (0.33 GFLOP/s at d=3).
    compute = max(
        float(work.sum()) / 330e9, float(work.max()) / 20e9, trials.max() * 20e-6
    )
    size = dimension**2
    h2d = 8 * dimension + 40 * size + len(work) * (8 * size + 68 + 24 * knots)
    d2h = len(work) * 8 * size
    # Arc's large transfers sustained about 6/10 GB/s H2D/D2H. The host
    # allowance covers packing, queue/dispatch, progress reads and handoff.
    return float(compute + 0.004 + h2d / 5e9 + d2h / 9e9)


def opencl_preferred(
    table: SegmentTable,
    rows: NDArray[np.int64],
    dimension: int,
    workers: int,
    *,
    budget: float,
    dp54: bool,
) -> bool:
    """Compare CPU and GPU wall prices for one acquire-wide ready wave.

    rows are already admitted, sorted global IDs. All have the same DC model;
    GPU buffers are grouped by ordered port pair and split at the batch cap.
    For an available device, the price uses the acquire work and worker width,
    not Engine or disk cache warmth or measured wall time. Calibrated on Arc
    for d=3 sampled pulses and d=32 FTF simultaneous drive at 1, 4 and 24 points.
    budget prices the CPU alternative. dp54 is the selected OpenCL tableau.
    """
    cpu, gpu = opencl_wave_prices(
        table, rows, dimension, workers, budget=budget, dp54=dp54
    )
    return gpu < cpu


def opencl_wave_prices(
    table: SegmentTable,
    rows: NDArray[np.int64],
    dimension: int,
    workers: int,
    *,
    budget: float,
    dp54: bool,
) -> tuple[float, float]:
    """Estimate CPU and GPU wall seconds for one wave of admitted OpenCL rows.

    Return (CPU, OpenCL). Budget selects the CPU tableau; dp54 selects the
    OpenCL tableau.
    """
    # One extra knot covers a final held value before the appended row end.
    knots = two_tone_knot_counts(table, rows) + 1
    trials = np.maximum(1.4 * table.duration_ns[rows], 2.0 * (knots - 1))
    work = trials * 12 * (8 * dimension**3 + 40 * dimension**2)
    lanes = min(workers, len(rows))
    # Dense CPU per-thread throughput falls from ~17 to ~9 GFLOP/s from
    # 1 to 7 workers; a 4 us per-trial floor accounts for small matrices.
    cpu_row = work * (1 + 0.15 * (lanes - 1)) / 17e9 + trials * 4e-6
    # This model is in seconds, unlike the relative scheduling units. On the
    # 960-knot FTF rows, measured DP5(4) CPU was 1.02–1.52 times the legacy
    # CPU estimate (which underpredicted actual DOP853 CPU on this host).
    cpu_row *= np.where(cpu_dp54_rows(table, rows, budget), 1.2, 1.0)
    cpu = max(float(cpu_row.sum()) / lanes, float(cpu_row.max()))
    gpu_work = work * _DP54_GPU_WORK_SCALE if dp54 else work
    # Like routing history, replacing the Engine must not change the route.
    # Disk cache amortizes cold compilation per machine, driver, kernel and d;
    # charge only the resident-program overhead, regardless of cache state.
    gpu = 3e-6
    ports, membership = np.unique(
        table.tone_port[rows, :2], axis=0, return_inverse=True
    )
    for group in range(len(ports)):
        members = np.flatnonzero(membership == group)
        for start in range(0, len(members), ROWS_PER_BATCH):
            batch = members[start : start + ROWS_PER_BATCH]
            gpu += _batch_gpu_seconds(
                gpu_work[batch], trials[batch], int(knots[batch].max()), dimension
            )
    return float(cpu), gpu


def direct_plans(
    table: SegmentTable,
    names: NDArray[np.str_],
    rows: NDArray[np.int64] | None = None,
    *,
    budget: float,
) -> tuple[dict[str, RowPlan], NDArray[np.int64]]:
    """Plan direct rows from complete chronological timeline routing.

    names aligns with rows, or the complete table when rows is None. A worker
    supplies its whole timeline, including previously accepted neighbors, not
    just its deferred candidates. Empty names break chains. Returned decisions
    retain global-row indexing; frames outside the selected family are empty.
    A chain ID is its first global row, independent of payload/apply partition.
    budget is the evolve allowance used to select the tableau for CPU endpoints;
    OpenCL tableau selection is separate because admission bounds row duration.
    """
    size = len(table.timeline)
    if rows is None:
        rows = np.arange(size, dtype=np.int64)
    chain = np.full(size, -1, dtype=np.int64)
    plans: dict[str, RowPlan] = {}
    for name in ("dop853", "flux_dop853"):
        selected = names == name
        if not np.any(selected):
            continue
        previous, following = rows[:-1], rows[1:]
        edges = (
            selected[:-1]
            & selected[1:]
            & (following == previous + 1)
            & (table.timeline[previous] == table.timeline[following])
            & (
                np.abs(
                    table.start_ns[following]
                    - (table.start_ns[previous] + table.duration_ns[previous])
                )
                <= 1e-9
            )
        )
        members = np.r_[edges, False] | np.r_[False, edges]
        starts = rows.copy()
        starts[1:][edges] = 0
        ids = np.maximum.accumulate(starts)
        chain[rows[members]] = ids[members]
        if name == "dop853":
            # Zero-filled storage leaves unrelated rows empty without filling a
            # full acquire's Unicode frame vector in every timeline worker.
            frame = np.zeros(size, dtype="U11")
            frame[rows[selected]] = "lab"
            # Removing bare GHz motion avoids an accumulating frequency bias.
            # Even at half flux this is cheaper than phase-budgeted lab control.
            # Chains retain lab endpoints for basis changes across row models.
            frame[rows[selected & ~members]] = "interaction"
            frame.setflags(write=False)
            dp54 = np.zeros(size, dtype=bool)
            selected_rows = rows[selected]
            dp54[selected_rows] = cpu_dp54_rows(table, selected_rows, budget)
            dp54.setflags(write=False)
            plans[name] = DOP853Plan(frame, chain, dp54)
        else:
            plans[name] = FluxDOP853Plan(chain)
    chain.setflags(write=False)
    return plans, chain
