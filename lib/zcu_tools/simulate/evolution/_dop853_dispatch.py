"""Timeline-wide frame and chain decisions for the direct solvers."""

import numpy as np
from numpy.typing import NDArray

from .kernel import RowPlan, SegmentTable
from .kernels.dop853 import DOP853Plan
from .kernels.flux_dop853 import FluxDOP853Plan


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
            if table.tone_amplitude.shape[1]:
                interaction = selected & ~members & (table.tones[rows] == 1)
                interaction &= table.tone_amplitude[rows, 0] != 0
                frame[rows[interaction]] = "interaction"
            frame.setflags(write=False)
            plans[name] = DOP853Plan(frame, chain)
        else:
            plans[name] = FluxDOP853Plan(chain)
    chain.setflags(write=False)
    return plans, chain
