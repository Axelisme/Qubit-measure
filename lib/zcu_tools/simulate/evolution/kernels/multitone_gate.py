"""Acquire-wide reuse admission for simultaneous covariant RWA gates."""

from collections import Counter

import numpy as np
from numpy.typing import NDArray

from ..kernel import KernelContext, SegmentTable
from ..secular import ModelSource
from ..workplan import FactKey, WorkPlan
from ._rwa_gate import GATE_POLICY, CovariantGateKernel, canonical_key

# Above the measured four-U break-even range (3--19 rows).
_MIN_REUSE = 20
_INVENTORY = FactKey(("multitone-gate-inventory", GATE_POLICY), tuple)


def populate_gate_inventory(
    table: SegmentTable, source: ModelSource, plan: WorkPlan
) -> tuple[tuple[bytes, int], ...]:
    """Publish complete-acquire canonical key counts before numerical guards.

    Prefix/worker subsets borrow this immutable inventory; density, numerical
    products and resident cache warmth do not participate.
    """

    def build() -> tuple[tuple[bytes, int], ...]:
        rows = np.flatnonzero(
            (table.tones >= 2)
            & ~table.constant_envelopes(np.arange(len(table.timeline)))
        )
        counts = Counter(canonical_key(table, int(row), source) for row in rows)
        return tuple(sorted(counts.items()))

    return plan.fact(_INVENTORY, build)


class MultitoneGateKernel(CovariantGateKernel):
    """Joint RWA gates with complete-inventory reuse >=20 and shared numerics.

    Requires shared real-envelope RWA admission, pruning bound <=min(budget,1e-3),
    and adaptive n-way secular split proxy <=.01. Row support bounds playback;
    sampled slices, constant companions and final sample hold are supported.
    Rejection retains ordinary row RWA. Products use the Engine resident LRU.
    """

    name = "multitone_gate"

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Combine fixed whole-acquire reuse with row-local numerical admission."""
        counts = dict(populate_gate_inventory(table, context.source, context.work_plan))
        reuse = np.array(
            [
                counts.get(canonical_key(table, int(row), context.source), 0)
                >= _MIN_REUSE
                for row in rows
            ]
        )
        accepted = np.zeros(len(rows), dtype=bool)
        accepted[reuse] = super().guard(table, rows[reuse], context)
        return accepted
