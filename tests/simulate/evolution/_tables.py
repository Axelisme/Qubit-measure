"""Segment input builders shared by the engine and static-kernel contracts."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from zcu_tools.simulate.evolution.kernel import SegmentTable


def idle_table(durations: Sequence[float] | NDArray[np.float64]) -> SegmentTable:
    """Return independent zero-drive rows, one duration in ns per timeline."""
    n = len(durations)
    empty = np.empty((n, 0))
    return SegmentTable(
        np.arange(n),
        np.arange(n, dtype=float),
        np.asarray(durations),
        np.zeros((n, 1), dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        empty.astype(np.int64),
        empty,
        empty,
        empty,
        empty.astype(np.int64),
        empty,
        (),
    )
