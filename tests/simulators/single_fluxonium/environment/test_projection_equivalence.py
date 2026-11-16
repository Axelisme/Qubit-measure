"""Projection chains agree with histories of prescribed Engine answers.

Environment composition is tested at ProjectionRunner.prepare (ADR-0078).
No numeric propagator or production history/ADC helper supplies the oracle.
"""

from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.core.simulation import CancellationQuery
from zcu_tools.experiment.readout_plan import ReadoutContents
from zcu_tools.simulate.adc import ProjectionChain
from zcu_tools.simulate.evolution import (
    Basis,
    EvolutionResult,
    FluxSchedule,
    ModelSource,
    PathDiagnostics,
    SecularModel,
    TimelineBatch,
)
from zcu_tools.simulators.single_fluxonium.environment.projection import (
    ProjectionRunner,
)
from zcu_tools.simulators.single_fluxonium.environment.template import (
    SingleFluxoniumPlan,
)

from tests.simulators._engine import EngineDouble

pytestmark = pytest.mark.sim


class AnswerEngine(EngineDouble):
    """Look up supplied densities by public span endpoint and input projector.

    The tables are answers, not a transition kernel applied to input densities.
    EngineDouble records each public handoff and handles cancellation/errors.
    """

    def __init__(
        self, prefix: NDArray[np.float64], transitions: NDArray[np.float64]
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.transitions = transitions

    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
    ) -> EvolutionResult:
        """Supply the answer for each point, deliberately in reverse point order."""
        (batch,) = batches
        event = int(batch.end_ns[0]) - 1
        if batch.initial_densities is None:
            assert event == 0
            populations = self.prefix[batch.points]
        else:
            states = np.argmax(
                np.diagonal(batch.initial_densities, axis1=-2, axis2=-1).real,
                axis=-1,
            )
            projectors = np.eye(self.prefix.shape[-1], dtype=complex)[states]
            np.testing.assert_array_equal(
                batch.initial_densities,
                np.array([np.diag(row) for row in projectors]),
            )
            populations = self.transitions[event - 1, batch.points, states]
        densities = np.array([np.diag(row) for row in populations], dtype=complex)
        self.result = EvolutionResult(
            batch.points[::-1], densities[::-1], PathDiagnostics()
        )
        return super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
        )


def given_answers(levels: int, points: int):
    """Nontrivial, point/span-dependent answers, including an unreachable state."""
    if levels == 2:
        prefix = np.array([[0.25, 0.75], [0.6, 0.4], [0, 1]])
        transitions = np.array(
            [
                [
                    [[0.8, 0.2], [0.3, 0.7]],
                    [[0.6, 0.4], [0.2, 0.8]],
                    [[0.4, 0.6], [0.7, 0.3]],
                ],
                [
                    [[0.1, 0.9], [0.5, 0.5]],
                    [[0.9, 0.1], [0.4, 0.6]],
                    [[0.2, 0.8], [0.6, 0.4]],
                ],
            ]
        )
    else:
        prefix = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0, 1, 0]])
        transitions = np.array(
            [
                [
                    [[0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],
                    [[0.2, 0.5, 0.3], [0.7, 0.1, 0.2], [0.1, 0.3, 0.6]],
                    [[0.4, 0.4, 0.2], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]],
                ],
                [
                    [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.1, 0.7, 0.2]],
                    [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
                    [[0.1, 0.8, 0.1], [0.4, 0.2, 0.4], [0.7, 0.1, 0.2]],
                ],
            ]
        )
    return prefix[:points], transitions[:, :points]


def projection_inputs(levels: int, points: int, events: int = 3):
    """Minimal DC-basis source and public plan; no model evolution is needed."""
    initial = (np.eye(levels) / levels).astype(np.complex128)
    model = SecularModel(
        (0,),
        "projection-answers",
        np.arange(levels, dtype=np.float64) * 0.1,
        np.eye(levels, dtype=complex),
        (),
        np.zeros((levels, levels)),
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
            np.full(points, event + 1, dtype=float),
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


def expect_chain(readout) -> ProjectionChain:
    """Require the projective seam's promised chain representation."""
    if not isinstance(readout, ProjectionChain):
        raise AssertionError("projective preparation must return a chain")
    return readout


def chain_paths(chain: ProjectionChain):
    """Read each complete path probability from the public chain output."""
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


def history_oracle(prefix, transitions, floor=0.0):
    """Branch explicit histories and sum their weights, never propagate a marginal.

    The given answer tables are independent fixture data. Adoption is decided
    from accumulated history mass, not the producer's matrix contraction. This
    also computes counts/losses without examining the returned chain or helpers.
    """
    points, levels = prefix.shape
    paths, counts, losses, masks = [], [], [], []
    for point in range(points):
        histories = {(state,): prefix[point, state] for state in range(levels)}
        point_counts, point_losses, point_masks = [], [], []
        for answers in transitions:
            mass = [
                sum(w for history, w in histories.items() if history[-1] == state)
                for state in range(levels)
            ]
            adopted = [weight >= floor for weight in mass]
            point_masks.append(adopted)
            point_counts.append(sum(adopted))
            point_losses.append(
                sum(w for w, keep in zip(mass, adopted, strict=True) if not keep)
            )
            histories = {
                (*history, target): weight
                * (
                    answers[point, history[-1], target]
                    if adopted[history[-1]]
                    else float(history[-1] == target)
                )
                for history, weight in histories.items()
                for target in range(levels)
            }
        paths.append(list(histories.values()))
        counts.append(point_counts)
        losses.append(point_losses)
        masks.append(point_masks)
    return np.array(paths), np.array(counts), np.array(losses), np.array(masks)


@pytest.mark.parametrize("levels", [2, 3])
@pytest.mark.parametrize("points", [1, 3])
def test_projection_chain_matches_per_history_reference(levels: int, points: int):
    prefix, transitions = given_answers(levels, points)
    source, basis, plan = projection_inputs(levels, points)
    with closing(source), closing(AnswerEngine(prefix, transitions)) as engine:
        expected, counts, losses, _ = history_oracle(prefix, transitions)
        actual = ProjectionRunner(engine, source, basis, 0.0, None).prepare(
            (plan,), (points,), b"oracle"
        )
        chain = expect_chain(actual.readout)
        np.testing.assert_array_equal(chain.initial, prefix)
        # Population normalization can round a supplied row by about one ULP.
        np.testing.assert_allclose(
            np.array(chain.transitions), transitions, rtol=0, atol=1e-15
        )
        # Only history products/reduction rounding differs; no solver tolerance.
        np.testing.assert_allclose(chain_paths(chain), expected, rtol=1e-14, atol=1e-16)
        np.testing.assert_array_equal(actual.diagnostics.continued_states, counts)
        np.testing.assert_array_equal(
            actual.diagnostics.uncontinued_probability, losses
        )
        # floor=0 adopts even the nonidentity rows of zero-probability states.
        assert np.all(counts == levels)


def test_floor_adopts_solver_rows_per_point_and_bounds_path_error():
    prefix, transitions = given_answers(2, 3)
    source, basis, plan = projection_inputs(2, 3)
    floor = 0.4
    with closing(source), closing(AnswerEngine(prefix, transitions)) as engine:
        exact_paths, _, _, _ = history_oracle(prefix, transitions)
        expected, counts, losses, masks = history_oracle(prefix, transitions, floor)
        actual = ProjectionRunner(engine, source, basis, floor, None).prepare(
            (plan,), (3,), b"point-local-floor"
        )
        chain = expect_chain(actual.readout)
        np.testing.assert_array_equal(chain.initial, prefix)
        # The same source row is solved for the plan but unused at point 0.
        np.testing.assert_array_equal(masks[:, 0, 0], [False, True, False])
        for point, event, state in product(range(3), range(2), range(2)):
            np.testing.assert_array_equal(
                chain.transitions[event][point, state],
                transitions[event, point, state]
                if masks[point, event, state]
                else np.eye(2)[state],
            )
        np.testing.assert_allclose(chain_paths(chain), expected, rtol=1e-14, atol=1e-16)
        np.testing.assert_array_equal(counts, [[1, 2], [2, 2], [1, 1]])
        np.testing.assert_allclose(losses, [[0.25, 0], [0, 0], [0, 0.3]], atol=1e-16)
        np.testing.assert_array_equal(actual.diagnostics.continued_states, counts)
        np.testing.assert_allclose(
            actual.diagnostics.uncontinued_probability, losses, atol=1e-16
        )
        tv = 0.5 * np.abs(expected - exact_paths).sum(axis=-1)
        assert tv[0] > 0
        assert tv[1] == pytest.approx(0, abs=1e-16)
        assert tv[2] > 0
        assert np.all(tv <= losses.sum(axis=-1) + 1e-14)


def test_solver_identity_rows_still_count_as_continued_states():
    prefix, _ = given_answers(2, 1)
    transitions = np.eye(2)[None, None]
    source, basis, plan = projection_inputs(2, 1, events=2)
    with closing(source), closing(AnswerEngine(prefix, transitions)) as engine:
        actual = ProjectionRunner(engine, source, basis, 0.4, None).prepare(
            (plan,), (1,), b"solver-identity"
        )
        chain = expect_chain(actual.readout)
        np.testing.assert_array_equal(chain.transitions[0], [np.eye(2)])
        assert actual.diagnostics.continued_states == ((1,),)
        np.testing.assert_array_equal(
            actual.diagnostics.uncontinued_probability, [[0.25]]
        )
