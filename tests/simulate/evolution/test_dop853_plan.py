"""Timeline-owned direct-solver frames survive worker and apply partitions."""

from dataclasses import replace
from threading import Event
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, FluxSchedule
from zcu_tools.simulate.evolution.kernel import ApplyContext
from zcu_tools.simulate.evolution.kernels.dop853 import DOP853Kernel
from zcu_tools.simulate.evolution.kernels.t4 import T4

from ._engine import batch, source, track


@runtime_checkable
class RecordedDirectPlan(Protocol):
    """The frame/chain decisions observed at the injected kernel boundary."""

    frame: NDArray[np.str_]
    chain: NDArray[np.int64]


class ObservedDOP853(DOP853Kernel):
    """Record dispatch decisions at the ordinary injected kernel seam."""

    def __init__(self):
        self.decisions = []

    def apply(self, prepared, key_index, table, rows, states, context: ApplyContext):
        plan = context.plan
        assert plan is not None, "dispatch must supply timeline-owned DOP853 decisions"
        if not isinstance(plan, RecordedDirectPlan):
            raise TypeError("direct solver plan must expose frame and chain decisions")
        self.decisions.extend(
            (
                int(table.timeline[row]),
                float(table.start_ns[row]),
                str(plan.frame[row]),
                int(plan.chain[row]),
            )
            for row in rows
        )
        return super().apply(prepared, key_index, table, rows, states, context)


def test_mixed_frames_follow_timeline_not_workers_or_chunks():
    n = 4
    shaped = EnvelopeSpec(
        "sampled",
        np.array([0.2 + 0.1j, 1.0, 0.4 - 0.2j]),
        np.array([0.0, 0.375, 0.75]),
    )
    # The first pair touches across a flux-level boundary. The next pulse is
    # isolated, then the final two overlapping tracks require the lab frame.
    # Shared and timeline-local level keys split the first pair's preparation.
    flux = FluxSchedule(
        np.arange(0, 2 * n + 1, 2),
        np.tile([0.0, 0.75], n),
        np.array([[0], [1], [0], [1], [0], [2], [0], [3]]),
    )
    timeline = replace(
        batch(n, duration=4.0),
        flux=flux,
        tracks=(
            track(n, shaped, duration=0.75),
            track(n, shaped, start=0.75, duration=0.75),
            track(n, shaped, start=2.0, duration=0.75),
            track(n, shaped, start=3.25, duration=0.75),
            track(n, shaped, start=3.25, duration=0.75),
        ),
    )
    outputs = []
    for workers, chunks in ((1, 1), (1, 3), (2, 1), (2, 3)):
        direct = ObservedDOP853()
        # No optional shaped families: every driven row uses the same terminal
        # numerical algorithm, so this isolates frame/chain scheduling.
        engine = Engine(workers, 0, kernels=(direct, T4()), chunks_per_worker=chunks)
        try:
            outputs.append(
                engine.evolve(source(), (timeline,), cancel=Event()).densities
            )
        finally:
            engine.close()
        for point in range(n):
            decisions = sorted(
                item[1:] for item in direct.decisions if item[0] == point
            )
            assert [item[:2] for item in decisions] == [
                (0.0, "lab"),
                (0.75, "lab"),
                (2.0, "interaction"),
                (3.25, "lab"),
            ]
            chain = [item[2] for item in decisions]
            assert chain[0] == chain[1] and chain[0] >= 0
            assert chain[2:] == [-1, -1]
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])
