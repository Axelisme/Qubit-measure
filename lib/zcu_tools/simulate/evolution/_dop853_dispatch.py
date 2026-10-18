"""Timeline-wide frame and chain decisions for the direct solvers."""

import numpy as np
from numpy.typing import NDArray

from .kernel import RowPlan, SegmentTable
from .kernels.dop853 import DOP853Plan
from .kernels.flux_dop853 import FluxDOP853Plan


def direct_plans(
    table: SegmentTable, names: NDArray[np.str_]
) -> tuple[dict[str, RowPlan], NDArray[np.int64]]:
    """Plan selected direct rows from complete chronological timeline routing.

    Empty names are unresolved or excluded rows and break chains. A chain ID
    is its first global row, independent of worker, payload and apply partition.
    """
    chain = np.full(len(names), -1, dtype=np.int64)
    plans: dict[str, RowPlan] = {}
    for name in ("dop853", "flux_dop853"):
        selected = names == name
        if not np.any(selected):
            continue
        edges = (
            selected[:-1]
            & selected[1:]
            & (table.timeline[:-1] == table.timeline[1:])
            & (
                np.abs(
                    table.start_ns[1:] - table.start_ns[:-1] - table.duration_ns[:-1]
                )
                <= 1e-9
            )
        )
        members = np.r_[edges, False] | np.r_[False, edges]
        starts = np.arange(len(names), dtype=np.int64)
        starts[1:][edges] = 0
        ids = np.maximum.accumulate(starts)
        chain[members] = ids[members]
        if name == "dop853":
            frame = np.full(len(names), "lab", dtype="U11")
            if table.tone_amplitude.shape[1]:
                interaction = selected & ~members & (table.tones == 1)
                interaction &= table.tone_amplitude[:, 0] != 0
                frame[interaction] = "interaction"
            frame.setflags(write=False)
            plans[name] = DOP853Plan(frame, chain)
        else:
            plans[name] = FluxDOP853Plan(chain)
    chain.setflags(write=False)
    return plans, chain
