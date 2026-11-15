"""Retained simultaneous covariant-gate family with no admitted rows."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import KernelContext, SegmentTable
from ._rwa_gate import CovariantGateKernel


class MultitoneGateKernel(CovariantGateKernel):
    """Retain the family name and reject every row.

    Every row is rejected, including pinned families and inventories at or
    above the 20-use cost preference. The existing shared numerical builder
    remains available to single-tone gates; its admission is unchanged.
    """

    name = "multitone_gate"

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return an all-False row mask, including for pinned calls."""
        del table, context
        return np.zeros(len(rows), dtype=np.bool_)
