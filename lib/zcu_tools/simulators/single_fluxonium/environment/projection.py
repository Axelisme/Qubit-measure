"""DC rank-1 outcome expansion and continuation through the evolution seam."""

from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import CancellationQuery
from zcu_tools.simulate.adc import ProjectionReadout, normalize_populations
from zcu_tools.simulate.evolution import (
    Basis,
    Engine,
    ModelSource,
    PathDiagnostics,
    batch_populations,
    merge_path_diagnostics,
)
from zcu_tools.simulators.single_fluxonium.environment.template import (
    SingleFluxoniumPlan,
)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Author-only projection cost and discarded probability, not measured IQ.

    points: flat request point IDs, in the same order as the following rows.
    retained_outcomes: per-point counts after each chronological projection.
        Counts exclude padding and include complete outcome histories, including
        zero-weight histories when floor is zero.
    pruned_weight: per-point probability discarded at each projection, measured
        against the original unpruned probability mass. Sum a row for cumulative
        discarded mass; earlier pruning is not counted again at later events.
    evolution_calls: number of Engine.evolve calls, including initial prefixes.
    """

    points: tuple[int, ...]
    retained_outcomes: tuple[tuple[int, ...], ...]
    pruned_weight: tuple[tuple[float, ...], ...]
    evolution_calls: int


@dataclass(frozen=True)
class ProjectionEvolution:
    """Noiseless projection readout and evolution/reuse diagnostics.

    readout: outcome weights and chronological event populations in point layout.
    paths: merged engine diagnostics from all prefixes and continuations.
    reuse_keys: exact span digests, including initial densities and time bounds;
        validate every key with Engine.reuse before reusing this preparation.
    diagnostics: per-point projection pruning and work counts.
    """

    readout: ProjectionReadout
    paths: PathDiagnostics
    reuse_keys: tuple[bytes, ...]
    diagnostics: ProjectionDiagnostics


@dataclass(frozen=True)
class _Outcome:
    weights: NDArray[np.float64]
    history: tuple[int, ...]
    densities: NDArray[np.complex128]


class ProjectionRunner:
    """Expand DC-energy outcomes while keeping the engine's routing policy intact.

    engine/source are the source-owned evolution resources. basis is the same
    retained DC energy basis as source densities and ADC labels. floor is the
    nonnegative outcome probability threshold; cancel is forwarded unchanged.
    One runner serves one preparation, retaining only its span diagnostics/keys.
    ADC CTMC jumps do not feed back into these density continuations.
    """

    def __init__(
        self,
        engine: Engine,
        source: ModelSource,
        basis: Basis,
        floor: float,
        cancel: CancellationQuery | None,
    ) -> None:
        self.engine, self.source, self.basis = engine, source, basis
        self.floor, self.cancel = floor, cancel
        self.paths: list[PathDiagnostics] = []
        self.keys: list[bytes] = []

    def _advance(
        self,
        plan: SingleFluxoniumPlan,
        end: NDArray[np.float64],
        outcome: _Outcome | None,
        key: bytes,
    ) -> NDArray[np.complex128]:
        start = None
        densities = None
        if outcome is not None:
            start = plan.read_events[len(outcome.history) - 1].end_ns
            densities = outcome.densities
        batch = replace(
            plan.batch,
            end_ns=end,
            segment_boundaries=tuple(
                tuple(t for t in boundaries if t <= stop)
                for boundaries, stop in zip(
                    plan.batch.segment_boundaries, end, strict=True
                )
            ),
            start_ns=start,
            initial_densities=densities,
        )
        digest = sha256(key)
        for array in (batch.points, end, start, densities):
            if array is not None:
                digest.update(repr((array.dtype.str, array.shape)).encode())
                digest.update(array.tobytes())
        reuse_key = digest.digest()
        result = self.engine.evolve(
            self.source, (batch,), cancel=self.cancel, reuse_key=reuse_key
        )
        self.keys.append(reuse_key)
        self.paths.append(result.paths)
        # Engine returns rows in its own point order, not necessarily batch order.
        indices = {int(point): index for index, point in enumerate(result.points)}
        order = [indices[int(point)] for point in plan.batch.points]
        return result.densities[order]

    def _split(
        self,
        outcomes: list[_Outcome],
        points: NDArray[np.int64],
    ) -> tuple[list[_Outcome], NDArray[np.float64]]:
        candidates = []
        for outcome in outcomes:
            probabilities = normalize_populations(
                batch_populations(outcome.densities, self.basis, self.basis)
            )
            for state in range(probabilities.shape[-1]):
                weights = outcome.weights * probabilities[:, state]
                densities = np.zeros_like(outcome.densities)
                densities[:, state, state] = 1
                candidates.append(
                    _Outcome(weights, (*outcome.history, state), densities)
                )
        weights = np.stack([outcome.weights for outcome in candidates], axis=-1)
        keep = weights >= self.floor
        dropped = np.where(keep, 0, weights).sum(axis=-1)
        retained = np.where(keep, weights, 0)
        totals = retained.sum(axis=-1)
        if np.any(totals == 0):
            point = int(points[np.flatnonzero(totals == 0)[0]])
            raise ValueError(
                f"point {point}: projection pruning removed all outcomes at floor {self.floor}"
            )
        normalized = retained / totals[:, None]
        return [
            replace(outcome, weights=normalized[:, index])
            for index, outcome in enumerate(candidates)
            if np.any(keep[:, index])
        ], dropped

    def _plan(
        self,
        plan: SingleFluxoniumPlan,
        key: bytes,
    ) -> tuple[
        list[_Outcome], tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...]
    ]:
        points = len(plan.batch.points)
        density = self._advance(plan, plan.read_events[0].end_ns, None, key)
        outcomes = [_Outcome(np.ones(points), (), density)]
        survival = np.ones(points)
        counts, pruned = [], []
        for event in range(len(plan.read_events) - 1):
            outcomes, dropped = self._split(outcomes, plan.batch.points)
            counts.append(
                np.full(points, len(outcomes))
                if self.floor == 0
                else np.count_nonzero([outcome.weights for outcome in outcomes], axis=0)
            )
            pruned.append(survival * dropped)
            survival *= 1 - dropped
            outcomes = [
                replace(
                    outcome,
                    densities=self._advance(
                        plan, plan.read_events[event + 1].end_ns, outcome, key
                    ),
                )
                for outcome in outcomes
            ]
        return (
            outcomes,
            tuple(
                tuple(int(count[point]) for count in counts) for point in range(points)
            ),
            tuple(
                tuple(float(weight[point]) for weight in pruned)
                for point in range(points)
            ),
        )

    def prepare(
        self,
        plans: tuple[SingleFluxoniumPlan, ...],
        point_dims: tuple[int, ...],
        key: bytes,
    ) -> ProjectionEvolution:
        """Evolve plans to terminal and return padded outcome histories by point.

        Plans cover each request point once and have a common event count.
        point_dims is the C-order request layout; key hashes all caller inputs.
        Low-weight histories are discarded per point after each projection and
        survivors renormalized. Zero-weight padding carries valid ground-state
        distributions but cannot be sampled. Errors propagate without a result.
        """
        results = [self._plan(plan, key) for plan in plans]
        states = len(self.basis.labels)
        outcomes_count = max(len(result[0]) for result in results)
        size = int(np.prod(point_dims))
        weights = np.zeros((size, outcomes_count))
        populations = [
            np.zeros((size, outcomes_count, states)) for _ in plans[0].read_events
        ]
        for values in populations:
            values[..., 0] = 1
        retained: list[tuple[int, ...]] = [()] * size
        pruned: list[tuple[float, ...]] = [()] * size
        for plan, (outcomes, counts, dropped) in zip(plans, results, strict=True):
            for point, count, loss in zip(
                plan.batch.points, counts, dropped, strict=True
            ):
                retained[point], pruned[point] = count, loss
            for index, outcome in enumerate(outcomes):
                weights[plan.batch.points, index] = outcome.weights
                for event, state in enumerate(outcome.history):
                    populations[event][plan.batch.points, index] = np.eye(states)[state]
                populations[-1][plan.batch.points, index] = normalize_populations(
                    batch_populations(outcome.densities, self.basis, self.basis)
                )
        return ProjectionEvolution(
            ProjectionReadout(
                weights.reshape((*point_dims, outcomes_count)),
                tuple(
                    values.reshape((*point_dims, outcomes_count, states))
                    for values in populations
                ),
            ),
            merge_path_diagnostics(self.paths),
            tuple(self.keys),
            ProjectionDiagnostics(
                tuple(range(size)), tuple(retained), tuple(pruned), len(self.keys)
            ),
        )
