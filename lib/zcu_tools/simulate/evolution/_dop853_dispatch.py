"""Timeline-wide frame and chain decisions for the direct solvers."""

import numpy as np
from numpy.typing import NDArray

from .kernel import RowPlan, SegmentTable
from .kernels.dop853 import DOP853Plan
from .kernels.flux_dop853 import FluxDOP853Plan


def opencl_rows(
    table: SegmentTable, plan: RowPlan | None, dimension: int
) -> NDArray[np.bool_] | None:
    """Mark global DOP853 rows that can run as independent OpenCL endpoints.

    Supply the completed direct plan for this acquire or resolved timeline.
    Return a read-only mask, or None for unsupported models or no DOP853 plan.
    """
    if (
        not isinstance(plan, DOP853Plan)
        or dimension > 32
        or not len(table.complete_pulses)
    ):
        return None
    eligible = (
        (plan.frame == "interaction")
        & table.complete_pulses
        & (table.tones == 2)
        & (table.duration_ns >= 100.0)
        & (table.duration_ns <= 10_000.0)
        & np.all(table.level == 0, axis=1)
        & (table.shift_content == b"")
    )
    if table.flux_waveform is not None:
        available = table.flux_waveform_available
        eligible &= False if available is None else ~available[table.timeline]
    rows = np.flatnonzero(eligible)
    if len(rows):
        # Envelope shape does not enter this guard. With FP64 step origins, the
        # number of trials and knot intervals matters for FP32 drift, not shape.
        sampled = np.array(
            [
                env.kind == "sampled" and len(env.sample_ns) <= 960
                for env in table.envelopes
            ]
        )
        eligible[rows] &= np.all(sampled[table.tone_envelope[rows, :2]], axis=1)
    eligible.setflags(write=False)
    return eligible


def direct_plans(
    table: SegmentTable,
    names: NDArray[np.str_],
    rows: NDArray[np.int64] | None = None,
) -> tuple[dict[str, RowPlan], NDArray[np.int64]]:
    """Plan direct rows from complete chronological timeline routing.

    names aligns with rows, or the complete table when rows is None. A worker
    supplies its whole timeline, including previously accepted neighbors, not
    just its deferred candidates. Empty names break chains. Returned decisions
    retain global-row indexing; frames outside the selected family are empty.
    A chain ID is its first global row, independent of payload/apply partition.
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
            plans[name] = DOP853Plan(frame, chain)
        else:
            plans[name] = FluxDOP853Plan(chain)
    chain.setflags(write=False)
    return plans, chain
