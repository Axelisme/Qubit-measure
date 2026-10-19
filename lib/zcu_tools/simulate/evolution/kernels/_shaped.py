"""Model-only bounds shared by shaped propagation guards."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import ModelSource
from ..workplan import WorkPlan
from ._workload_facts import drive_bound


def drive_bounds(
    table: SegmentTable,
    rows: NDArray[np.int64],
    source: ModelSource,
    work_plan: WorkPlan,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return quadrature spectral norm and largest secular rate per selected row.

    Reuse model/level/port scalars across this evolve's guard and build calls.
    Resolve row mapping once per level/port in this call. No density is read.
    """
    bounds: dict[tuple[tuple[int, ...], int], tuple[float, float]] = {}
    strengths, rates = np.empty(len(rows)), np.empty(len(rows))
    for index, row in enumerate(rows):
        level = tuple(int(v) for v in table.level[row])
        port = int(table.tone_port[row, 0])
        key = level, port
        if key not in bounds:
            model = source.model(level)
            bound = drive_bound(model, port, work_plan)
            bounds[key] = bound.strength, bound.rate
        strengths[index], rates[index] = bounds[key]
    return strengths, rates
