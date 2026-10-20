"""Cold-cost selection of retained multitone graphs through Engine."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    DriveTrack,
    Engine,
    EnvelopeSpec,
    FluxSchedule,
    ModelSource,
    SecularDrive,
    SecularModel,
    TimelineBatch,
)


def retained_graph(*, dense: bool) -> tuple[ModelSource, TimelineBatch]:
    size = 8
    positive = np.zeros((size, size), dtype=np.complex128)
    for i in range(size):
        for j in range(i + 1, size):
            if dense or (i % 2 == 0 and j == i + 1):
                positive[i, j] = 0.025
    drive = SecularDrive("a", positive + positive.T, 1j * (positive - positive.T))
    model = SecularModel(
        (0,),
        "dense-retention" if dense else "small-retention",
        np.arange(size, dtype=np.float64) * 0.3,
        np.eye(size, dtype=np.complex128),
        (drive, replace(drive, name="b")),
        np.zeros((size, size)),
        np.zeros((size, size)),
    )
    initial = np.zeros((size, size), dtype=np.complex128)
    initial[0, 0] = 1
    source = ModelSource(initial, lambda level: replace(model, level=level))
    times = np.linspace(0, 128, 65)
    envelope = EnvelopeSpec(
        "sampled", np.exp(-0.5 * ((times - 64) / 32) ** 2).astype(complex), times
    )
    tracks = tuple(
        DriveTrack(
            port,
            np.array([0.0]),
            np.array([128.0]),
            np.array([frequency]),
            np.array([phase]),
            np.array([1.0]),
            (envelope,),
            np.array([0]),
            np.array([True]),
        )
        for port, frequency, phase in (("a", 0.3, 0.7), ("b", 0.301437, -1.2))
    )
    batch = TimelineBatch(
        np.array([0]),
        tracks,
        FluxSchedule(np.array([0, 1]), np.array([0.0]), np.array([[0]])),
        np.array([128.0]),
    )
    return source, batch


@pytest.mark.parametrize("dense, family", [(True, "dop853"), (False, "multitone_rwa")])
def test_retained_graph_cost_selects_cheaper_route_independent_of_cache(dense, family):
    histories = []
    for cache_bytes in (0, 1_000_000):
        source, batch = retained_graph(dense=dense)
        with closing(Engine(1, cache_bytes)) as engine:
            results = [
                engine.evolve(source, (batch,), cancel=Event()) for _ in range(3)
            ]
        routes = [dict(result.paths.segments_by_kernel) for result in results]
        assert routes == [{family: 1}] * 3
        histories.append(routes)
    assert histories[0] == histories[1]
