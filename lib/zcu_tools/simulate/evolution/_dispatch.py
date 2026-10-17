"""Cross-family decision graph; numerical guards remain kernel-owned."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from joblib import delayed
from numpy.typing import NDArray

from ._parallel import Executor
from ._planning import Geometry
from ._prepare_keys import group_keys
from ._shaped_dispatch import preference, spectral_candidates
from .engine import PathDiagnostics
from .kernel import GuardPreparation, Kernel, SegmentTable
from .secular import ModelSource


@dataclass(frozen=True)
class ShapedPlan:
    """Acquire-wide shaped costs, fixed before any numerical guard.

    rows: all ordinary shaped global rows, in ascending order.
    order: (len(rows), 3) gate/period/direct index permutations.
    spectral: global rows selected for the low-reuse spectral attempt.
    Workers borrow these arrays and never recalculate reuse from their subset.
    """

    rows: NDArray[np.int64]
    order: NDArray[np.int64]
    spectral: NDArray[np.int64]


@dataclass(frozen=True)
class Selection:
    """Resolve deferred global rows inside their single timeline worker.

    rows: global rows owned by this worker's timeline.
    shaped: acquire-wide cost plan, or None if no selected row is shaped.
    A selection disables further deferral and retains accepted guard products.
    """

    rows: NDArray[np.int64]
    shaped: ShapedPlan | None
    rejected: Mapping[str, NDArray[np.bool_]]


@dataclass(frozen=True)
class Dispatch:
    """Resolved shared decisions and acquire-local worker work.

    names: family per global row; empty deferred slices are filled by their worker.
    diagnostics: counts for decisions resolved so far, refreshed after workers join.
    costs: nonnegative scheduling estimates, including provisional deferred costs.
    preparations: accepted caller guard products, grouped by family.
    deferred: worker-owned row mask, or None when no rows need worker classification.
    shaped: fixed acquire-wide cost plan, or None when no shaped work exists.
    rejected: caller guard decisions to skip in worker-local fallback chains.
    """

    names: NDArray[np.str_]
    diagnostics: PathDiagnostics
    costs: NDArray[np.float64]
    preparations: Mapping[str, tuple[GuardPreparation, ...]] = field(
        default_factory=dict
    )
    deferred: NDArray[np.bool_] | None = None
    shaped: ShapedPlan | None = None
    rejected: Mapping[str, NDArray[np.bool_]] = field(default_factory=dict)


def diagnostics(table: SegmentTable, names: NDArray[np.str_]) -> PathDiagnostics:
    """Count resolved global rows and distinct timeline/family pairs.

    Empty names denote deferred rows and are omitted until workers resolve them.
    This function does not run guards or alter names.
    """
    rows = np.flatnonzero(names != "")
    unique, counts = np.unique(names[rows], return_counts=True)
    pairs = np.empty(len(rows), dtype=[("name", "U16"), ("timeline", "i8")])
    pairs["name"], pairs["timeline"] = names[rows], table.timeline[rows]
    point_names, point_counts = np.unique(np.unique(pairs)["name"], return_counts=True)
    return PathDiagnostics(
        dict(zip(unique.tolist(), counts.tolist(), strict=True)),
        dict(zip(point_names.tolist(), point_counts.tolist(), strict=True)),
    )


def require(
    kernels: Mapping[str, Kernel],
    name: str,
    table: SegmentTable,
    rows: NDArray[np.int64],
) -> Kernel:
    """Resolve a required family or identify the first affected segment."""
    if name not in kernels:
        row = int(rows[0])
        raise RuntimeError(
            f"segment {row} (timeline {table.timeline[row]}, start {table.start_ns[row]} ns) requires kernel {name}"
        )
    return kernels[name]


@dataclass(frozen=True)
class _Guarded:
    rows: NDArray[np.int64]
    rejected: NDArray[np.int64]
    preparations: tuple[GuardPreparation, ...]


def _guard_key(
    kernel: Kernel,
    table: SegmentTable,
    source: ModelSource,
    budget: float,
    rows: NDArray[np.int64],
) -> _Guarded:
    """Keep the exact guard product whenever any row accepts this key."""
    result = kernel.guard_prepared(table, rows, source, budget)
    if result is None:
        raise RuntimeError(
            f"kernel {kernel.name} declares guard_builds without a handoff"
        )
    accepted = result.accepted
    selected = rows[accepted]
    preparations = result.preparations if len(selected) else ()
    return _Guarded(selected, rows[~accepted], preparations)


class _Classifier:
    def __init__(
        self,
        geometry: Geometry,
        kernels: Mapping[str, Kernel],
        source: ModelSource,
        budget: float,
        pool: Executor | None = None,
        selection: Selection | None = None,
    ):
        self.pool, self.selection = pool, selection
        self.geometry, self.kernels, self.source, self.budget = (
            geometry,
            kernels,
            source,
            budget,
        )
        self.table = geometry.table
        self.names = np.full(len(self.table.timeline), "", dtype="U16")
        self.costs = np.zeros(len(self.names))
        self.deferred = np.zeros(len(self.names), dtype=bool)
        self.scope = np.ones(len(self.names), dtype=bool)
        if selection is not None:
            self.scope[:] = False
            self.scope[selection.rows] = True
        self.shaped = selection.shaped if selection is not None else None
        self.preparations: dict[str, list[GuardPreparation]] = {}
        self.rejected: dict[str, NDArray[np.bool_]] = {}

    def guard_keys(
        self, kernel: Kernel, rows: NDArray[np.int64]
    ) -> Iterable[_Guarded | None]:
        """Defer local builders; guard shared keys with the existing pool."""
        groups = group_keys(kernel, self.table, rows)
        if self.selection is None:
            local = rows[~groups.shared[groups.inverse]]
            self.deferred[local] = True
            if len(local):
                self.costs[local] = kernel.cost(self.table, local)
        selections = (
            selected
            for index, selected in groups.selections()
            if self.selection is not None or groups.shared[index]
        )
        if self.pool is not None and np.count_nonzero(groups.shared) >= 8:
            return self.pool(
                delayed(_guard_key)(
                    kernel, self.table, self.source, self.budget, selected
                )
                for selected in selections
            )
        return (
            _guard_key(kernel, self.table, self.source, self.budget, selected)
            for selected in selections
        )

    def try_kernel(self, kernel: Kernel, rows: NDArray[np.int64]) -> NDArray[np.int64]:
        """Batch cheap guards in the caller and never retry its rejected rows."""
        eligible = rows
        if self.selection is not None:
            previous = self.selection.rejected.get(kernel.name)
            if previous is not None:
                eligible = rows[~previous[rows]]
        if not len(eligible):
            return rows
        if kernel.guard_builds:
            results = self.guard_keys(kernel, eligible)
        else:
            accepted = kernel.guard(self.table, eligible, self.source, self.budget)
            results = (_Guarded(eligible[accepted], eligible[~accepted], ()),)
        for result in results:
            if result is None:
                raise RuntimeError("guard job returned no decision")
            self.names[result.rows] = kernel.name
            if len(result.rejected):
                self.rejected.setdefault(
                    kernel.name, np.zeros(len(self.names), dtype=bool)
                )[result.rejected] = True
            if result.preparations:
                self.preparations.setdefault(kernel.name, []).extend(
                    result.preparations
                )
        return rows[(self.names[rows] == "") & ~self.deferred[rows]]

    def accept(self, rows: NDArray[np.int64], candidates: tuple[str, ...]) -> None:
        """Try numerical guards in fixed cost order, or defer the remaining chain."""
        for name in candidates:
            if not len(rows):
                return
            rows = self.try_kernel(require(self.kernels, name, self.table, rows), rows)
        if len(rows):
            raise RuntimeError(f"no valid kernel for segment {rows[0]}")

    def classify_shaped(self, rows: NDArray[np.int64]) -> None:
        """Share acquire-wide costs with workers, independent of spectral outcomes."""
        if self.shaped is None:
            self.shaped = ShapedPlan(
                rows,
                preference(self.table, rows, self.kernels),
                spectral_candidates(self.table, rows, self.kernels),
            )
        plan = self.shaped
        spectral = self.kernels.get("shaped_sambe")
        if spectral is not None:
            selected = plan.spectral[self.scope[plan.spectral]]
            if len(selected):
                self.try_kernel(spectral, selected)
        pending = (
            self.scope[plan.rows]
            & (self.names[plan.rows] == "")
            & ~self.deferred[plan.rows]
        )
        families = ("gate_cache", "period_table", "dop853")
        pending_rows, order = plan.rows[pending], plan.order[pending]
        for permutation in np.unique(order, axis=0):
            selected = pending_rows[np.all(order == permutation, axis=1)]
            self.accept(selected, tuple(families[index] for index in permutation))

    def classify(self) -> Dispatch:
        table = self.table
        rows = np.flatnonzero(self.scope)
        self.accept(rows[table.tones[rows] == 0], ("t4",))
        flux = self.geometry.flux_drive & (table.tones > 0)
        self.accept(rows[flux[rows]], ("flux_dop853",))
        ordinary = ~flux
        kinds = np.array([envelope.kind == "constant" for envelope in table.envelopes])
        constant = np.ones(len(table.timeline), dtype=bool)
        for slot in range(table.tone_port.shape[1]):
            playing = table.tones > slot
            constant[playing] &= kinds[table.tone_envelope[playing, slot]]
        single = ordinary & (table.tones == 1)
        self.accept(rows[(single & constant)[rows]], ("dyson", "floquet_period", "t6"))
        shaped = np.flatnonzero(single & ~constant)
        if np.any(self.scope[shaped]):
            self.classify_shaped(shaped)
        self.accept(
            rows[(ordinary & (table.tones == 2) & constant)[rows]],
            ("floquet", "dop853"),
        )
        self.accept(
            rows[
                (ordinary & (((table.tones == 2) & ~constant) | (table.tones >= 3)))[
                    rows
                ]
            ],
            ("dop853",),
        )
        return self._result()

    def _result(self) -> Dispatch:
        for name in np.unique(self.names[self.names != ""]):
            rows = np.flatnonzero(self.names == name)
            self.costs[rows] = self.kernels[str(name)].cost(self.table, rows)
        if not np.isfinite(self.costs).all() or np.any(self.costs < 0):
            raise ValueError("kernel costs must be finite and nonnegative")
        return Dispatch(
            self.names,
            diagnostics(self.table, self.names),
            self.costs,
            {name: tuple(items) for name, items in self.preparations.items()},
            self.deferred,
            self.shaped,
            self.rejected,
        )


def classify(
    geometry: Geometry,
    kernels: Mapping[str, Kernel],
    source: ModelSource,
    budget: float,
    pool: Executor | None = None,
    selection: Selection | None = None,
) -> Dispatch:
    """Guard shared candidates now and resolve local candidates in their worker.

    Eight or more shared prepare keys may use a borrowed Engine threading pool.
    Smaller shared selections stay serial. Selection resolves one timeline's
    deferred rows without a nested pool, retaining guard products for apply.
    Both modes use the same acquire-wide cost plan and numerical guards.
    """
    return _Classifier(geometry, kernels, source, budget, pool, selection).classify()
