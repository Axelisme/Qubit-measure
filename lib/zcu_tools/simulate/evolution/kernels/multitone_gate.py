"""Paused simultaneous covariant-gate family; the shared builder is retained."""

import numpy as np
from numpy.typing import NDArray

from ..kernel import KernelContext, SegmentTable
from ._rwa_gate import CovariantGateKernel


class MultitoneGateKernel(CovariantGateKernel):
    """Keep the family name while sampled multitone gates are paused.

    Every row is rejected, including pinned families and inventories at or
    above the 20-use cost preference. The existing shared numerical builder
    remains available to single-tone gates; its admission is unchanged.
    """

    name = "multitone_gate"

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return an all-False row mask while sampled multitone admission is paused."""
        del table, context
        return np.zeros(len(rows), dtype=np.bool_)
