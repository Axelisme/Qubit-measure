"""Cross-family decision graph; numerical guards remain kernel-owned."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ._planning import Geometry
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


def _shape_counts(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """Count phase-independent shaped gate contents before selecting a family."""
    if not len(rows):
        return np.empty(0, dtype=np.int64)
    keys = np.empty(
        len(rows),
        dtype=[
            ("level", "i8", table.level.shape[1]),
            ("envelope", "S64"),
            ("port", "i8"),
            ("frequency", "f8"),
            ("amplitude", "f8"),
            ("duration", "f8"),
            ("offset", "f8"),
        ],
    )
    identities = np.array(
        [envelope.identity for envelope in table.envelopes], dtype="S64"
    )
    keys["level"] = table.level[rows]
    keys["envelope"] = identities[table.tone_envelope[rows, 0]]
    for name, column in [
        ("port", table.tone_port),
        ("frequency", table.tone_frequency_ghz),
        ("amplitude", table.tone_amplitude),
        ("offset", table.tone_envelope_offset_ns),
    ]:
        keys[name] = column[rows, 0]
    keys["duration"] = table.duration_ns[rows]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return counts[inverse]


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
        self.accept(rows[single & constant], ("dyson", "floquet_period", "t6"))
        shaped = rows[single & ~constant]
        gate = self.kernels.get("gate_cache")
        if gate is None:
            # The canonical gate identity still identifies a required missing
            # registry entry without consulting another numerical family.
            counts = _shape_counts(table, shaped)
        else:
            _, inverse, usage = np.unique(
                gate.prepare_keys(table, shaped),
                return_inverse=True,
                return_counts=True,
            )
            counts = usage[inverse]
        self.accept(shaped[counts >= 10], ("gate_cache",))
        self.accept(shaped[counts < 10], ("period_table",))
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
    """Apply branch order, gate repetition threshold and numerical fallbacks."""
    return _Classifier(geometry, kernels, source, budget).classify()
