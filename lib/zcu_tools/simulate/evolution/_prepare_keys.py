"""Prepare-key ownership across independent engine timelines."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel, KernelContext, SegmentTable


@dataclass(frozen=True)
class KeyGroups:
    """Model-only keys and their lifetime within one dispatch selection.

    keys: unique opaque prepare keys in NumPy sorted order.
    representative: one global row per key, aligned with keys.
    rows: selected global rows in caller order.
    inverse: key index per selected row.
    counts: number of selected rows per key.
    shared: true only when a key occurs in more than one timeline.
    Arrays are acquire-local and must not be mutated by callers.
    """

    keys: NDArray[np.void]
    representative: NDArray[np.int64]
    rows: NDArray[np.int64]
    inverse: NDArray[np.int64]
    counts: NDArray[np.int64]
    shared: NDArray[np.bool_]

    def selections(self) -> Iterator[tuple[int, NDArray[np.int64]]]:
        """Yield key index and all its global rows without quadratic filtering."""
        if len(self.keys) == 1:
            yield 0, self.rows
            return
        order = np.argsort(self.inverse, kind="stable")
        offsets = np.r_[0, np.cumsum(self.counts)]
        for index in range(len(self.keys)):
            yield index, self.rows[order[offsets[index] : offsets[index + 1]]]


def group_keys(
    kernel: Kernel, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
) -> KeyGroups:
    """Group model-only keys and distinguish cross-timeline sharing from repeats."""
    keys, first, inverse, counts = np.unique(
        kernel.prepare_keys(table, rows, context),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    shared = np.zeros(len(keys), dtype=bool)
    np.logical_or.at(
        shared,
        inverse,
        table.timeline[rows] != table.timeline[rows[first]][inverse],
    )
    return KeyGroups(keys, rows[first], rows, inverse, counts, shared)
