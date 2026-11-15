"""Dispatch queries observe Engine routes without advancing quantum states."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, EnvelopeSpec, ModelSource

from ._engine import batch, source, track
from .contract.synthetic_models import synthetic_model


def test_query_matches_evolve_and_maps_split_rows_to_batches_and_points():
    first = replace(
        batch(2, duration=3.0),
        points=np.array([9, 4]),
        segment_boundaries=((1.0,), (2.0,)),
    )
    second = replace(batch(1, duration=2.0), points=np.array([17]))
    with closing(source()) as src, closing(Engine(1)) as engine:
        query = engine.inspect_dispatch(src, (first, second))
        result = engine.evolve(src, (first, second), telemetry=True)
    assert [
        (row.key.batch, row.key.point, row.key.segment, row.start_ns, row.duration_ns)
        for row in query.rows
    ] == [
        (0, 9, 0, 0.0, 1.0),
        (0, 9, 1, 1.0, 2.0),
        (0, 4, 0, 0.0, 2.0),
        (0, 4, 1, 2.0, 1.0),
        (1, 17, 0, 0.0, 2.0),
    ]
    assert {row.family for row in query.rows} == set(result.paths.segments_by_kernel)
    assert len(query.rows) == result.paths.segments_by_kernel["t4"]
    for row in query.rows:
        assert row.status == "selected"
        assert [(c.family, c.guard) for c in row.candidates] == [("t4", "accepted")]
        assert row.candidates[0].price is not None
        assert row.candidates[0].price >= 0


def test_later_short_circuit_keeps_an_earlier_guard_rejection():
    pulse = track(
        20,
        EnvelopeSpec("sampled", np.array([0.2, 1.0]), np.array([0.0, 20.0])),
        duration=20.0,
    )
    timeline = replace(
        batch(20, duration=20.0),
        tracks=(pulse, replace(pulse, frequency_ghz=np.full(20, 0.3))),
    )
    with (
        closing(source()) as src,
        closing(
            Engine(1, families=("multitone_gate", "multitone_rwa", "dop853"))
        ) as engine,
    ):
        query = engine.inspect_dispatch(src, (timeline,))
    assert len(query.rows) == 20
    for row in query.rows:
        assert row.family == "dop853"
        assert {c.family: c.guard for c in row.candidates} == {
            "multitone_gate": "rejected",
            "multitone_rwa": "rejected",
            "dop853": "accepted",
        }


@pytest.mark.parametrize("count", [1, 2])
def test_floquet_query_runs_shared_guard_but_defers_local_frame(count):
    model = synthetic_model()
    model = replace(
        model,
        rates_per_ns=model.rates_per_ns * 0.001,
        dephasing_per_ns=model.dephasing_per_ns * 0.001,
        drives=tuple(
            replace(
                d,
                in_phase_ghz=d.in_phase_ghz * 0.1,
                quadrature_ghz=d.quadrature_ghz * 0.1,
            )
            for d in model.drives
        ),
    )
    strong = replace(
        track(count, EnvelopeSpec("constant"), duration=1.1),
        frequency_ghz=np.full(count, 3.7),
        amplitude=np.full(count, 0.1),
    )
    probe = replace(
        strong, frequency_ghz=np.full(count, 0.11), amplitude=np.full(count, 0.002)
    )
    timeline = replace(batch(count, duration=1.1), tracks=(strong, probe))
    with (
        closing(
            ModelSource(np.eye(3, dtype=np.complex128) / 3, lambda level: model)
        ) as src,
        closing(Engine(1, families=("floquet",))) as engine,
    ):
        query = engine.inspect_dispatch(src, (timeline,))
        if count == 2:
            evolved = engine.evolve(src, (timeline,), telemetry=True)
            assert evolved.paths.segments_by_kernel == {"floquet": count}
    assert len(query.rows) == count
    for row in query.rows:
        (candidate,) = row.candidates
        assert candidate.family == "floquet"
        assert candidate.price is not None
        if count == 1:
            assert row.status == "deferred" and row.family is None
            assert candidate.guard == "deferred"
        else:
            assert row.status == "selected" and row.family == "floquet"
            assert candidate.guard == "accepted"


def test_query_marks_candidates_after_an_accepted_guard_as_untested():
    pulse = replace(track(2, EnvelopeSpec("constant")), amplitude=np.full(2, 1e-6))
    timeline = replace(batch(2), tracks=(pulse,))
    with closing(source()) as src, closing(Engine(1)) as engine:
        query = engine.inspect_dispatch(src, (timeline,))
    for row in query.rows:
        assert row.family == "dyson"
        assert row.candidates[0].guard == "accepted"
        assert row.candidates[1:]
        assert all(
            c.guard == "untested_after_short_circuit" for c in row.candidates[1:]
        )
        assert all(c.price is None for c in row.candidates[1:])
