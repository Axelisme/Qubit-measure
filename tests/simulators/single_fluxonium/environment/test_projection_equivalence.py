"""Rank-1 history readouts agree with an independent per-outcome reference."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from zcu_tools.experiment.readout_plan import ReadoutContents
from zcu_tools.simulate.adc import ProjectionReadout, normalize_populations
from zcu_tools.simulate.evolution import (
    Basis,
    Engine,
    FluxSchedule,
    ModelSource,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulators.single_fluxonium.environment.projection import (
    ProjectionRunner,
)
from zcu_tools.simulators.single_fluxonium.environment.template import (
    SingleFluxoniumPlan,
)

pytestmark = pytest.mark.sim


def projection_inputs(levels: int, points: int, events: int = 4):
    """Small dissipative DC model with unequal point-local read times."""
    initial = np.diag(np.arange(1, levels + 1, dtype=float))
    initial = (initial / np.trace(initial)).astype(complex)
    rates = np.full((levels, levels), 0.013)
    np.fill_diagonal(rates, 0)
    rates[0, 1:] = 0.07
    model = SecularModel(
        (0,),
        "projection-reference",
        np.arange(levels, dtype=np.float64) * 0.1,
        np.eye(levels, dtype=complex),
        (),
        rates,
        np.zeros((levels, levels)),
    )
    source = ModelSource(initial, lambda level: replace(model, level=level))
    basis = Basis(
        "dc",
        tuple(str(state) for state in range(levels)),
        np.eye(levels, dtype=complex),
    )
    reads = tuple(
        ReadoutContents(
            np.arange(1, points + 1, dtype=float) * (event + 1),
            ((),) * points,
            ((),) * points,
            np.arange(points, dtype=np.int64),
            ((),) * points,
        )
        for event in range(events)
    )
    batch = TimelineBatch(
        np.arange(points, dtype=np.int64),
        (),
        FluxSchedule(
            np.arange(points + 1, dtype=np.int64),
            np.zeros(points),
            np.zeros((points, 1), dtype=np.int64),
        ),
        reads[-1].end_ns,
        segment_boundaries=tuple(
            tuple(read.end_ns[p] for read in reads) for p in range(points)
        ),
    )
    plan = SingleFluxoniumPlan(batch, reads[-1], ((),) * points, reads, ((),) * points)
    return source, basis, plan


def naive_readout(
    engine: Engine, source: ModelSource, plan: SingleFluxoniumPlan, floor: float
):
    """Evolve and expand every history separately; never share projected states."""
    points = len(plan.batch.points)
    levels = source.initial_density.shape[0]
    first = replace(
        plan.batch,
        end_ns=plan.read_events[0].end_ns,
        segment_boundaries=tuple((float(t),) for t in plan.read_events[0].end_ns),
    )
    initial = engine.evolve(source, (first,)).densities
    outcomes = [(np.ones(points), (), initial)]
    counts, losses = [], []
    survival = np.ones(points)
    for event in range(len(plan.read_events) - 1):
        candidates = []
        for weight, history, density in outcomes:
            probabilities = normalize_populations(
                np.diagonal(density, axis1=-2, axis2=-1).real
            )
            for state in range(levels):
                projected = np.zeros_like(density)
                projected[:, state, state] = 1
                candidates.append(
                    (weight * probabilities[:, state], (*history, state), projected)
                )
        weights = np.stack([value[0] for value in candidates], axis=-1)
        keep = weights >= floor
        dropped = np.where(keep, 0, weights).sum(axis=-1)
        retained = np.where(keep, weights, 0)
        normalized = retained / retained.sum(axis=-1)[:, None]
        outcomes = []
        for index, (_, history, projected) in enumerate(candidates):
            if not np.any(keep[:, index]):
                continue
            end = plan.read_events[event + 1].end_ns
            batch = replace(
                plan.batch,
                start_ns=plan.read_events[event].end_ns,
                end_ns=end,
                initial_densities=projected,
                segment_boundaries=tuple(
                    tuple(t for t in row if t <= stop)
                    for row, stop in zip(
                        plan.batch.segment_boundaries, end, strict=True
                    )
                ),
            )
            density = engine.evolve(source, (batch,)).densities
            outcomes.append((normalized[:, index], history, density))
        counts.append(
            np.full(points, len(outcomes))
            if floor == 0
            else np.count_nonzero([v[0] for v in outcomes], axis=0)
        )
        losses.append(survival * dropped)
        survival *= 1 - dropped
    populations = [np.empty((points, len(outcomes), levels)) for _ in plan.read_events]
    for index, (_, history, density) in enumerate(outcomes):
        for event, state in enumerate(history):
            populations[event][:, index] = np.eye(levels)[state]
        populations[-1][:, index] = normalize_populations(
            np.diagonal(density, axis1=-2, axis2=-1).real
        )
    return (
        ProjectionReadout(
            np.stack([v[0] for v in outcomes], axis=-1), tuple(populations)
        ),
        np.array(counts).T,
        np.array(losses).T,
    )


@pytest.mark.parametrize("levels", [3, 4, 5])
@pytest.mark.parametrize("points", [1, 3])
@pytest.mark.parametrize("floor", [0.0, 0.015])
def test_projection_readout_matches_per_history_reference(
    levels: int, points: int, floor: float
):
    source, basis, plan = projection_inputs(levels, points)
    try:
        with closing(Engine(1)) as reference_engine, closing(Engine(1)) as engine:
            expected, counts, losses = naive_readout(
                reference_engine, source, plan, floor
            )
            actual = ProjectionRunner(engine, source, basis, floor, None).prepare(
                (plan,), (points,), b"oracle"
            )
        # Same DC propagation and reduction order permit bitwise equality. Event
        # one-hot rows also identify the retained history set and its ordering.
        np.testing.assert_array_equal(actual.readout.weights, expected.weights)
        for observed, reference in zip(
            actual.readout.populations, expected.populations, strict=True
        ):
            np.testing.assert_array_equal(observed, reference)
        np.testing.assert_array_equal(actual.diagnostics.retained_outcomes, counts)
        np.testing.assert_array_equal(actual.diagnostics.pruned_weight, losses)
    finally:
        source.close()


def test_projection_budget_rejects_exponential_readout_before_evolution():
    source, basis, plan = projection_inputs(3, 1, events=24)
    cancelled = Event()
    cancelled.set()
    try:
        with (
            closing(Engine(1)) as engine,
            pytest.raises(
                ValueError,
                match=r"projection memory budget.*estimated_bytes=\d+.*budget_bytes=\d+.*levels=3.*points=1.*outcomes=\d+.*projection_prune_floor.*batch",
            ),
        ):
            ProjectionRunner(engine, source, basis, 0.0, cancelled).prepare(
                (plan,), (1,), b"budget"
            )
    finally:
        source.close()
