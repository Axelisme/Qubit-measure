"""Projection chains agree with independently enumerated dissipative histories."""

from contextlib import closing
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.experiment.readout_plan import ReadoutContents
from zcu_tools.simulate.adc import ProjectionChain
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


def projection_inputs(levels: int, points: int, events: int = 4, *, idle: bool = False):
    """Small dissipative DC model with unequal point-local read times."""
    initial = np.diag(np.arange(1, levels + 1, dtype=float))
    initial = (initial / np.trace(initial)).astype(complex)
    rates = np.full((levels, levels), 0.013)
    np.fill_diagonal(rates, 0)
    rates[0, 1:] = 0.07
    if idle:
        rates[:] = 0
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


def chain_paths(chain: ProjectionChain):
    """Enumerate all read states, including terminal, for a small test chain."""
    levels = chain.initial.shape[-1]
    probabilities = []
    for history in product(range(levels), repeat=len(chain.transitions) + 1):
        weight = chain.initial[..., history[0]].copy()
        for transition, before, after in zip(
            chain.transitions, history[:-1], history[1:], strict=True
        ):
            weight *= transition[..., before, after]
        probabilities.append(weight)
    return np.stack(probabilities, axis=-1)


def history_oracle(source: ModelSource, plan: SingleFluxoniumPlan):
    """Independently enumerate histories with the analytic DC master equation.

    This fixture has diagonal Hamiltonian and density, so expm of its population
    generator is the exact Lindblad solution. No producer/ADC helper is used.
    """
    rates = source.model((0,)).rates_per_ns
    generator = rates - np.diag(rates.sum(axis=0))
    levels = len(generator)
    rows = []
    for point in range(len(plan.batch.points)):
        times = [0.0, *(read.end_ns[point] for read in plan.read_events)]
        row = []
        for history in product(range(levels), repeat=len(plan.read_events)):
            density = source.initial_density.copy()
            probability = 1.0
            for state, start, stop in zip(history, times[:-1], times[1:], strict=True):
                populations = expm(generator * (stop - start)) @ np.diag(density).real
                probability *= populations[state]
                density = np.zeros_like(density)
                density[state, state] = 1
            row.append(probability)
        rows.append(row)
    return np.array(rows)


@pytest.mark.parametrize("levels", [2, 3])
@pytest.mark.parametrize("points", [1, 3])
def test_projection_chain_matches_per_history_reference(levels: int, points: int):
    source, basis, plan = projection_inputs(levels, points, events=3)
    try:
        expected = history_oracle(source, plan)
        with closing(Engine(1)) as engine:
            actual = ProjectionRunner(engine, source, basis, 0.0, None).prepare(
                (plan,), (points,), b"oracle"
            )
        assert isinstance(actual.readout, ProjectionChain)
        # The analytic DC oracle and the engine solve the same master equation;
        # only floating-point exponentiation/reduction differs at this scale.
        np.testing.assert_allclose(
            chain_paths(actual.readout), expected, rtol=1e-12, atol=1e-14
        )
    finally:
        source.close()


def test_floor_adopts_solver_rows_per_point_and_bounds_path_error():
    source, basis, plan = projection_inputs(2, 3, events=3)
    floor = 0.4
    try:
        exact_paths = history_oracle(source, plan).reshape(3, 2, 2, 2)
        first = exact_paths.sum(axis=(2, 3))
        # At the first read, state 0 is below floor at point 0 but above it at
        # points 1 and 2. The same batched solve must not change point 0's row.
        np.testing.assert_array_equal(first[:, 0] >= floor, [False, True, True])
        with closing(Engine(1)) as engine:
            actual = ProjectionRunner(engine, source, basis, floor, None).prepare(
                (plan,), (3,), b"point-local-floor"
            )
        chain = actual.readout
        np.testing.assert_allclose(chain.initial, first, atol=1e-14)
        generator = source.model((0,)).rates_per_ns
        generator = generator - np.diag(generator.sum(axis=0))
        expected = np.zeros_like(exact_paths)
        counts = np.zeros((3, 2), dtype=np.int64)
        losses = np.zeros((3, 2))
        for point in range(3):
            weights = {(state,): first[point, state] for state in range(2)}
            for event in range(2):
                marginal = np.array(
                    [
                        sum(
                            weight
                            for history, weight in weights.items()
                            if history[-1] == state
                        )
                        for state in range(2)
                    ]
                )
                adopted = marginal >= floor
                counts[point, event] = np.count_nonzero(adopted)
                losses[point, event] = marginal[~adopted].sum()
                duration = (
                    plan.read_events[event + 1].end_ns[point]
                    - plan.read_events[event].end_ns[point]
                )
                exact = expm(generator * duration).T
                for state in range(2):
                    np.testing.assert_allclose(
                        chain.transitions[event][point, state],
                        exact[state] if adopted[state] else np.eye(2)[state],
                        atol=1e-14,
                    )
                weights = {
                    (*history, target): weight
                    * (
                        exact[history[-1], target]
                        if adopted[history[-1]]
                        else float(history[-1] == target)
                    )
                    for history, weight in weights.items()
                    for target in range(2)
                }
            for history, weight in weights.items():
                expected[(point, *history)] = weight
        np.testing.assert_allclose(
            chain_paths(chain), expected.reshape(3, -1), atol=1e-14
        )
        np.testing.assert_array_equal(actual.diagnostics.continued_states, counts)
        np.testing.assert_allclose(
            actual.diagnostics.uncontinued_probability, losses, atol=1e-14
        )
        tv = 0.5 * np.abs(expected - exact_paths).sum(axis=(1, 2, 3))
        assert tv[0] > 0
        assert np.all(tv <= losses.sum(axis=-1) + 1e-14)
    finally:
        source.close()


def test_solver_identity_rows_still_count_as_continued_states():
    source, basis, plan = projection_inputs(2, 1, events=2, idle=True)
    try:
        with closing(Engine(1)) as engine:
            actual = ProjectionRunner(engine, source, basis, 0.4, None).prepare(
                (plan,), (1,), b"solver-identity"
            )
        np.testing.assert_array_equal(actual.readout.transitions[0], [np.eye(2)])
        assert actual.diagnostics.continued_states == ((1,),)
        np.testing.assert_allclose(
            actual.diagnostics.uncontinued_probability, [[1 / 3]]
        )
    finally:
        source.close()
