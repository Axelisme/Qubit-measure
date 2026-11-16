"""Constant-tone secular evolution by direct density integration."""

import numpy as np
from numpy.typing import NDArray

from .._t6_dispatch import prepare_keys
from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from . import dop853
from ._workload_facts import packed_segments


class T6Kernel:
    """One constant tone at any strength, with full secular dissipation.

    Prepare keys contain flux level, shift content, port, signed carrier GHz,
    signed gain and duration ns. The engine namespaces keys by the DC model
    source identity. Phase and start time are apply-only.
    Every row uses the shared H0-rotating density solver with the same
    budget-scaled local tolerances as dop853, without a fixed step ceiling.
    These controls do not bound whole-sequence error. Preparations contain
    model-only operators and remain density-independent.
    A positive finite budget is required; integration failure raises
    RuntimeError rather than silently selecting another algorithm.
    """

    name = "t6"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Accept exactly one unity envelope, irrespective of branch priority."""
        source, budget = context.source, context.budget
        del source
        _check_budget(budget)
        accepted = table.tones[rows] == 1
        if table.tone_envelope.shape[1]:
            constant = np.array(
                [envelope.kind == "constant" for envelope in table.envelopes]
            )
            active = np.flatnonzero(accepted)
            accepted[active] &= constant[table.tone_envelope[rows[active], 0]]
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Return phase-independent keys with unchanged physical identity."""
        del context
        return prepare_keys(table, rows)

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return table-only nonnegative chunk-interleaving weights.

        This coarse relative estimate does not determine candidate order.
        """
        return 1 + np.log1p(
            table.duration_ns[rows] * np.abs(table.tone_frequency_ghz[rows, 0])
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> dop853.DOP853Prepared:
        """Pack model-only operators for aligned constant-tone key representatives.

        keys must equal prepare_keys(table, rows, context), with one row per key.
        Invalid keys, unsupported rows or a nonpositive/nonfinite budget raise
        ValueError. Model construction failures propagate.
        """
        source, budget = context.source, context.budget
        _check_budget(budget)
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows, context)
        ):
            raise ValueError("t6: keys and representative rows must align")
        if not self.guard(table, rows, context).all():
            raise ValueError("t6: prepare requires one constant tone")
        return dop853.prepare_models([table.model(source, row) for row in rows])

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline guard preparation handoff; the engine calls guard normally."""
        source, budget = context.source, context.budget
        del table, rows, source, budget

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """Decline input binding; apply uses the model-only prepared payload."""
        budget = context.budget
        del prepared, key_index, table, rows, source_initial, budget

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Integrate aligned densities independently in the H0 frame, in place.

        Return exactly to the model basis with the full secular bath.
        No density repair, RWA, pruning or bath splitting occurs.
        Numerical failure raises RuntimeError without rerouting. Return an
        all-False bool mask aligned with rows; no post-apply accuracy check.
        """
        context.require_no_plan()
        budget = context.budget
        _check_budget(budget)
        if not isinstance(prepared, dop853.DOP853Prepared):
            raise ValueError("t6: incompatible prepared payload")
        plan = dop853.DOP853Plan(
            np.full(len(table.timeline), "interaction"),
            np.full(len(table.timeline), -1, dtype=np.int64),
        )
        dop853.advance_endpoints(
            dop853.EndpointData(
                prepared.models, packed_segments(table, context.work_plan)
            ),
            (rows, key_index),
            states,
            budget,
            plan,
        )
        if not np.isfinite(states).all():
            raise RuntimeError("t6: nonfinite density after application")
        return np.zeros(len(rows), dtype=np.bool_)


def _check_budget(budget: float) -> None:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("t6: budget must be positive and finite")
