"""Model-only bounds shared by shaped propagation guards."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import ModelSource


def drive_bounds(
    table: SegmentTable, rows: NDArray[np.int64], source: ModelSource
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return quadrature spectral norm and largest secular rate per selected row.

    Resolve each level/port once per call. No source initial density is read.
    """
    bounds: dict[tuple[tuple[int, ...], int], tuple[float, float]] = {}
    strengths, rates = np.empty(len(rows)), np.empty(len(rows))
    for index, row in enumerate(rows):
        level = tuple(int(v) for v in table.level[row])
        port = int(table.tone_port[row, 0])
        key = level, port
        if key not in bounds:
            model = source.model(level)
            drive = model.drives[port]
            bounds[key] = (
                float(
                    max(
                        np.linalg.norm(drive.in_phase_ghz, 2),
                        np.linalg.norm(drive.quadrature_ghz, 2),
                    )
                ),
                float(
                    max(
                        model.rates_per_ns.sum(axis=0).max(),
                        model.dephasing_per_ns.max(),
                    )
                ),
            )
        strengths[index], rates[index] = bounds[key]
    return strengths, rates
