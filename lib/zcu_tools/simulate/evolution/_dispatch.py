"""Cross-family decision graph; numerical guards remain kernel-owned."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ._planning import Geometry
from ._shaped_dispatch import preference
from .engine import PathDiagnostics
from .kernel import GuardPreparation, Kernel, SegmentTable
from .secular import ModelSource


@dataclass(frozen=True)
class Dispatch:
    names: NDArray[np.str_]
    diagnostics: PathDiagnostics
    costs: NDArray[np.float64]
    preparations: Mapping[str, tuple[GuardPreparation, ...]] = field(
        default_factory=dict
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


class _Classifier:
    def __init__(
        self,
        geometry: Geometry,
        kernels: Mapping[str, Kernel],
        source: ModelSource,
        budget: float,
    ):
        self.geometry, self.kernels, self.source, self.budget = (
            geometry,
            kernels,
            source,
            budget,
        )
        self.table = geometry.table
        self.names = np.full(len(self.table.timeline), "", dtype="U16")
        self.preparations: dict[str, tuple[GuardPreparation, ...]] = {}

    def accept(self, rows: NDArray[np.int64], candidates: tuple[str, ...]) -> None:
        """Try families in order, checking each numerical guard before fallback."""
        for name in candidates:
            if not len(rows):
                return
            kernel = require(self.kernels, name, self.table, rows)
            result = kernel.guard_prepared(self.table, rows, self.source, self.budget)
            if result is None:
                accepted = kernel.guard(self.table, rows, self.source, self.budget)
            else:
                accepted = result.accepted
                self.preparations[name] = (
                    *self.preparations.get(name, ()),
                    *result.preparations,
                )
            self.names[rows[accepted]] = name
            rows = rows[~accepted]
        if len(rows):
            raise RuntimeError(f"no valid kernel for segment {rows[0]}")

    def classify(self) -> Dispatch:
        table = self.table
        rows = np.arange(len(table.timeline), dtype=np.int64)
        self.accept(rows[table.tones == 0], ("t4",))
        flux = self.geometry.flux_drive & (table.tones > 0)
        self.accept(rows[flux], ("flux_dop853",))
        ordinary = ~flux
        kinds = np.array([envelope.kind == "constant" for envelope in table.envelopes])
        constant = np.ones(len(rows), dtype=bool)
        for slot in range(table.tone_port.shape[1]):
            playing = table.tones > slot
            constant[playing] &= kinds[table.tone_envelope[playing, slot]]
        single = ordinary & (table.tones == 1)
        self.accept(rows[single & constant], ("dyson", "rwa", "floquet_period", "t6"))
        shaped = rows[single & ~constant]
        if len(shaped):
            families = ("gate_cache", "period_table", "dop853")
            order = preference(table, shaped, self.kernels)
            for permutation in np.unique(order, axis=0):
                selected = shaped[np.all(order == permutation, axis=1)]
                self.accept(selected, tuple(families[index] for index in permutation))
        self.accept(
            rows[ordinary & (table.tones == 2) & constant], ("floquet", "dop853")
        )
        self.accept(
            rows[ordinary & (((table.tones == 2) & ~constant) | (table.tones >= 3))],
            ("dop853",),
        )
        return self._result()

    def _result(self) -> Dispatch:
        unique, counts = np.unique(self.names, return_counts=True)
        segment_counts = dict(zip(unique.tolist(), counts.tolist(), strict=True))
        pairs = np.empty(len(self.names), dtype=[("name", "U16"), ("timeline", "i8")])
        pairs["name"], pairs["timeline"] = self.names, self.table.timeline
        point_names, point_counts = np.unique(
            np.unique(pairs)["name"], return_counts=True
        )
        point_classes = dict(
            zip(point_names.tolist(), point_counts.tolist(), strict=True)
        )
        costs = np.zeros(len(self.names))
        for name in unique:
            rows = np.flatnonzero(self.names == name)
            costs[rows] = self.kernels[str(name)].cost(self.table, rows)
        if not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError("kernel costs must be finite and nonnegative")
        return Dispatch(
            self.names,
            PathDiagnostics(segment_counts, point_classes),
            costs,
            self.preparations,
        )


def classify(
    geometry: Geometry,
    kernels: Mapping[str, Kernel],
    source: ModelSource,
    budget: float,
) -> Dispatch:
    """Apply branch order, amortized shaped costs and numerical fallbacks."""
    return _Classifier(geometry, kernels, source, budget).classify()
