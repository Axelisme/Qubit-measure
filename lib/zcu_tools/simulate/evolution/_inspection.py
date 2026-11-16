"""Immutable dispatch observations and their row-coordinate projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._dispatch import Dispatch
    from ._planning import Geometry
    from .engine import TimelineBatch
    from .kernel import SegmentTable


type GuardDecision = Literal[
    "accepted", "rejected", "deferred", "untested_after_short_circuit"
]


@dataclass(frozen=True)
class RowKey:
    """Identify one final split segment in the caller's input.

    batch: zero-based index in the batches sequence, including empty batches.
    point: caller's TimelineBatch.points value, not its array position.
    segment: zero-based chronological split-span index within that point.
        Splits include flux, drive/plateau and explicit segment boundaries.
        Continuations number only the retained spans at/after start_ns.
    """

    batch: int
    point: int
    segment: int


@dataclass(frozen=True)
class DispatchCandidate:
    """One family offered by the shared classifier for a row.

    family: production family name (or internal injected kernel name).
    guard: accepted/rejected for an evaluated numerical guard, deferred when a
        local prepare key must wait for its worker, or
        untested_after_short_circuit when an earlier candidate stopped the round.
    scheduling_cost: nonnegative relative row work for chunk interleaving,
        from Kernel.cost. Evaluated and deferred candidates include this value;
        short-circuited candidates have None. It is not seconds or a guard
        certificate and cannot explain why a family was chosen. The separate
        cold-amortized candidate-ordering estimate is not exposed.
    """

    family: str
    guard: GuardDecision
    scheduling_cost: float | None


@dataclass(frozen=True)
class DispatchRow:
    """Dispatch observation for one split span, with no output density.

    key: batch/point/split-span coordinates defined by RowKey.
    start_ns: absolute span start in ns.
    duration_ns: span length in ns.
    family: accepted family name, or None while unresolved.
    status: selected, deferred to timeline-worker guard, or needs_prefix_input
        when evolve must compute a driven prefix density before classifying.
    candidates: offered families in first-offer order, with their last decision.
        Families skipped structurally or for cost are not listed. Empty when
        prefix input prevents classification. Short-circuited candidates are
        listed only within a round that the classifier actually offered.
    """

    key: RowKey
    start_ns: float
    duration_ns: float
    family: str | None
    status: Literal["selected", "deferred", "needs_prefix_input"]
    candidates: tuple[DispatchCandidate, ...]


@dataclass(frozen=True)
class DispatchInspection:
    """One acquire's dispatch observations without density evolution.

    rows: immutable rows in batch order, input point order, then chronological
        split-span order. Empty work returns (). Only selected rows promise a
        family; unresolved rows explicitly state why no family is available.
    """

    rows: tuple[DispatchRow, ...]


def needs_prefix_input(table: SegmentTable) -> NDArray[np.bool_]:
    """Mark driven certified prefixes that evolve resolves before classification."""
    return np.array(
        [
            group is not None
            and not table.prefix_no_drive[group]
            and group not in table.prefix_inputs
            for group in (table.prefix_group or (None,) * len(table.timeline))
        ],
        dtype=bool,
    )


def report(
    geometry: Geometry,
    batches: Sequence[TimelineBatch],
    dispatch: Dispatch,
    trace: Mapping[int, Mapping[str, DispatchCandidate]],
) -> DispatchInspection:
    """Project shared-classifier decisions into caller batch/point coordinates."""
    table = geometry.table
    batch_indices = [index for index, b in enumerate(batches) for _ in b.points]
    first_rows = np.searchsorted(table.timeline, table.timeline, side="left")
    prefix = needs_prefix_input(table)
    return DispatchInspection(
        tuple(
            DispatchRow(
                RowKey(
                    batch_indices[int(timeline)],
                    int(geometry.points[timeline]),
                    row - int(first_rows[row]),
                ),
                float(table.start_ns[row]),
                float(table.duration_ns[row]),
                str(dispatch.names[row]) or None,
                "needs_prefix_input"
                if prefix[row]
                else "deferred"
                if dispatch.deferred is not None and dispatch.deferred[row]
                else "selected",
                tuple(trace.get(row, {}).values()),
            )
            for row, timeline in enumerate(table.timeline)
        )
    )
