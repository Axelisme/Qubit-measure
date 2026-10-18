"""Synthetic engine batches and recording C5 collaborators shared by tests."""

from dataclasses import dataclass, replace
from threading import Event

import numpy as np
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    TimelineBatch,
)
from zcu_tools.simulate.evolution.kernel import (
    ApplyContext,
    GuardResult,
    Prepared,
    SegmentTable,
)

from .contract.kernel_suite import synthetic_model


def source() -> ModelSource:
    model = synthetic_model()
    return ModelSource(
        np.diag([0.2, 0.3, 0.5]).astype(complex),
        lambda level: replace(model, level=level),
    )


def batch(n: int = 12, *, duration: float = 3.0) -> TimelineBatch:
    return TimelineBatch(
        np.arange(n),
        (),
        FluxSchedule(np.arange(n + 1), np.zeros(n), np.zeros((n, 1), dtype=np.int64)),
        np.full(n, duration),
    )


def track(
    n: int, envelope: EnvelopeSpec, *, start: float = 0, duration: float = 3
) -> DriveTrack:
    return DriveTrack(
        "drive",
        np.full(n, start),
        np.full(n, duration),
        np.full(n, 0.1),
        np.zeros(n),
        np.ones(n),
        (envelope,),
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=bool),
    )


@dataclass
class Payload:
    count: int

    @property
    def nbytes(self) -> int:
        return self.count * 16


class RecordingKernel:
    """Identity dynamics with public observations at the kernel boundary."""

    guard_builds = False

    def __init__(self, name: str, *, accepts: bool = True, cancel: Event | None = None):
        self.name = name
        self.accepts = accepts
        self.cancel = cancel
        self.preparations = 0
        self.calls: list[
            tuple[SegmentTable, NDArray[np.int64], NDArray[np.complex128]]
        ] = []

    def guard(self, table, rows, source, budget):
        return np.full(len(rows), self.accepts)

    def prepare_keys(self, table, rows):
        keys = np.empty(
            len(rows), dtype=[("level", "i8", table.level.shape[1]), ("duration", "f8")]
        )
        keys["level"] = table.level[rows]
        keys["duration"] = table.duration_ns[rows]
        return keys

    def cost(self, table, rows):
        return np.ones(len(rows))

    def prepare(self, keys, table, rows, source, budget):
        self.preparations += len(keys)
        return Payload(len(keys))

    def guard_prepared(self, table, rows, source, budget) -> GuardResult | None:
        """Keep the ordinary guard path for this test collaborator."""
        del table, rows, source, budget

    def bind_input(
        self, prepared, key_index, table, rows, source_initial, budget
    ) -> Prepared | None:
        return None

    def apply(
        self, prepared: Prepared, key_index, table, rows, states, context: ApplyContext
    ):
        self.calls.append((table, rows.copy(), states.copy()))
        if self.cancel is not None:
            self.cancel.set()
        return np.zeros(len(rows), dtype=np.bool_)


def run(engine: Engine, src: ModelSource, batches: tuple[TimelineBatch, ...]):
    try:
        return engine.evolve(src, batches, cancel=Event())
    finally:
        engine.close()
