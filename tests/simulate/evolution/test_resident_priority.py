"""Overlapping guard domains preserve stale failed-product lookup order."""

from contextlib import closing
from dataclasses import dataclass, replace
from threading import Event

import numpy as np
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec
from zcu_tools.simulate.evolution.kernel import GuardPreparation, GuardResult

from ._engine import RecordingKernel, batch, source, track


@dataclass(frozen=True)
class Domain:
    durations: tuple[float, ...]
    nbytes: int = 16


class PriorityKernel(RecordingKernel):
    guard_builds = True

    def __init__(self):
        super().__init__("dyson")
        self.admitted: list[float] = []

    def prepare_keys(self, table, rows, context):
        keys = np.empty(len(rows), dtype=[("duration", "f8")])
        keys["duration"] = table.duration_ns[rows]
        return keys

    def guard_prepared(self, table, rows, context):
        duration = float(table.duration_ns[rows[0]])
        domain = (duration, 3.0) if duration in (1.0, 2.0) else (duration,)
        keys = np.array([(value,) for value in domain], dtype=[("duration", "f8")])
        return GuardResult(
            table.duration_ns[rows] != 2.0,
            (GuardPreparation(keys, Domain(domain)),),
        )

    def admit_prepared(self, prepared, table, rows, context):
        self.admitted.append(prepared.durations[0])
        return GuardResult(table.duration_ns[rows] != 2.0)


def test_stale_rejected_domain_precedes_overlapping_ordinary_domain():
    kernel = PriorityKernel()
    src = source()
    with closing(
        Engine(1, 100000, kernels=(kernel, RecordingKernel("floquet_period")))
    ) as engine:
        for duration in (1.0, 2.0, 9.0, 3.0):
            timeline = replace(
                batch(2, duration=duration),
                tracks=(track(2, EnvelopeSpec("constant"), duration=duration),),
            )
            engine.evolve(src, (timeline,), cancel=Event())
        assert kernel.admitted == [2.0]
