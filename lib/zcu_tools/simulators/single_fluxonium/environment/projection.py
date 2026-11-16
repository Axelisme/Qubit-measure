"""DC rank-1 histories with last-state density sharing.

PROJECTION_MEMORY_BUDGET_BYTES is a deterministic 2 GiB allowance for projection
arrays, including split temporaries, histories, retained plans and padded event
populations. Checks precede their allocation and raise ValueError. The allowance
leaves 6 GiB of an 8 GiB process cap for the engine, ADC, imports and allocator;
it is not a bound on those other owners or on total process RSS.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
from math import prod

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import ProjectionReadout, normalize_populations
from zcu_tools.simulate.boundary import CancellationQuery
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

PROJECTION_MEMORY_BUDGET_BYTES = 2 * 1024**3


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
class _Outcomes:
    # History-major order is unchanged: parent first, then ascending new state.
    weights: NDArray[np.float64]  # (points, histories)
    history: NDArray[np.int64]  # (histories, projections)

    @property
    def nbytes(self) -> int:
        return self.weights.nbytes + self.history.nbytes


@dataclass(frozen=True)
class _PlanResult:
    outcomes: _Outcomes
    terminal: dict[int, NDArray[np.float64]]
    counts: NDArray[np.int64]
    pruned: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        return (
            self.outcomes.nbytes
            + self.counts.nbytes
            + self.pruned.nbytes
            + sum(values.nbytes for values in self.terminal.values())
        )


class ProjectionRunner:
    """Expand DC-energy histories with at most one density per last state/span.

    engine/source are the source-owned evolution resources. basis is the same
    retained DC energy basis as source densities and ADC labels. floor is the
    nonnegative outcome probability threshold; cancel is forwarded unchanged.
    One runner serves one preparation. Densities never cross a plan or span;
    ADC CTMC jumps do not feed back into these density continuations.
    Projection arrays exceeding PROJECTION_MEMORY_BUDGET_BYTES raise ValueError
    before allocation, with dimensions and suggested workload reductions.
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
        self._retained_bytes = 0
        self._plan_bytes = 0

    def _check_budget(
        self, points: int, outcomes: int, events: int, arrays: int
    ) -> None:
        states = len(self.basis.labels)
        # Old/new densities, TimelineBatch's owned copy and engine return/reorder
        # can overlap. Probability conversion also owns temporary dense arrays.
        shared = points * states**3 * 16 * 6
        estimated = self._retained_bytes + self._plan_bytes + shared + arrays
        if estimated > PROJECTION_MEMORY_BUDGET_BYTES:
            raise ValueError(
                "projection memory budget exceeded: "
                f"estimated_bytes={estimated}, budget_bytes={PROJECTION_MEMORY_BUDGET_BYTES}, "
                f"levels={states}, points={points}, outcomes={outcomes}, events={events}; "
                "reduce levels, increase projection_prune_floor, or reduce points/batch size"
            )

    def _advance(
        self,
        plan: SingleFluxoniumPlan,
        event: int,
        densities: NDArray[np.complex128] | None,
        key: bytes,
    ) -> NDArray[np.complex128]:
        end = plan.read_events[event].end_ns
        start = None if densities is None else plan.read_events[event - 1].end_ns
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
        indices = {int(point): index for index, point in enumerate(result.points)}
        order = [indices[int(point)] for point in plan.batch.points]
        return result.densities[order]

    def _continue(
        self, plan: SingleFluxoniumPlan, event: int, outcomes: _Outcomes, key: bytes
    ) -> dict[int, NDArray[np.complex128]]:
        # Solve each distinct last state once, in its first history occurrence
        # order. There is no cached solve to validate within this span. The caller
        # still validates all reuse keys before reusing a complete preparation
        # (ADR-0078); every new span goes through Engine.evolve's routing policy.
        densities = {}
        shape = (len(plan.batch.points), len(self.basis.labels), len(self.basis.labels))
        for state in dict.fromkeys(int(s) for s in outcomes.history[:, -1]):
            projected = np.zeros(shape, dtype=np.complex128)
            projected[:, state, state] = 1
            densities[state] = self._advance(plan, event, projected, key)
        return densities

    def _probabilities(
        self, densities: dict[int, NDArray[np.complex128]]
    ) -> dict[int, NDArray[np.float64]]:
        return {
            state: normalize_populations(
                batch_populations(density, self.basis, self.basis)
            )
            for state, density in densities.items()
        }

    def _split(
        self,
        outcomes: _Outcomes,
        probabilities: dict[int, NDArray[np.float64]],
        points: NDArray[np.int64],
    ) -> tuple[_Outcomes, NDArray[np.float64]]:
        states = len(self.basis.labels)
        count = outcomes.weights.shape[1] * states
        events = outcomes.history.shape[1] + 1
        # Account for weights, retained, normalized, where/reduction scratch and
        # boolean masks, all in original parent/state order. No candidate density
        # or Python object is created for a history that pruning will discard.
        workspace = outcomes.nbytes + len(points) * count * 48
        self._check_budget(len(points), count, events, workspace)
        weights = np.empty((len(points), count))
        for parent in range(outcomes.weights.shape[1]):
            state = int(outcomes.history[parent, -1]) if events > 1 else -1
            weights[:, parent * states : (parent + 1) * states] = (
                outcomes.weights[:, parent, None] * probabilities[state]
            )
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
        selected = np.flatnonzero(np.any(keep, axis=0))
        survivor_bytes = len(selected) * (len(points) * 8 + events * 16 + 32)
        self._check_budget(
            len(points), len(selected), events, workspace + survivor_bytes
        )
        history = np.empty((len(selected), events), dtype=np.int64)
        history[:, :-1] = outcomes.history[selected // states]
        history[:, -1] = selected % states
        return _Outcomes(normalized[:, selected], history), dropped

    def _plan(self, plan: SingleFluxoniumPlan, key: bytes) -> _PlanResult:
        points, events = len(plan.batch.points), len(plan.read_events)
        self._plan_bytes = points * events * 32
        self._check_budget(points, 1, events, 0)
        density = self._advance(plan, 0, None, key)
        probabilities = self._probabilities({-1: density})
        del density
        outcomes = _Outcomes(np.ones((points, 1)), np.empty((1, 0), dtype=np.int64))
        survival = np.ones(points)
        counts = np.empty((points, events - 1), dtype=np.int64)
        pruned = np.empty((points, events - 1))
        for event in range(events - 1):
            outcomes, dropped = self._split(outcomes, probabilities, plan.batch.points)
            counts[:, event] = (
                outcomes.weights.shape[1]
                if self.floor == 0
                else np.count_nonzero(outcomes.weights, axis=-1)
            )
            pruned[:, event] = survival * dropped
            survival *= 1 - dropped
            densities = self._continue(plan, event + 1, outcomes, key)
            probabilities = self._probabilities(densities)
            del densities
        self._plan_bytes = 0
        return _PlanResult(outcomes, probabilities, counts, pruned)

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
        distributions but cannot be sampled. Projection arrays exceeding the
        module's fixed 2 GiB budget raise ValueError before allocation. Other
        numeric errors propagate without a result.
        """
        states, size = len(self.basis.labels), prod(point_dims)
        events = len(plans[0].read_events)
        # With pruning disabled the exact outcome count is known without solving.
        if self.floor == 0:
            count = states ** (events - 1)
            self._check_budget(
                size, count, events, size * count * (1 + events * states) * 16
            )
        results = []
        for plan in plans:
            result = self._plan(plan, key)
            results.append(result)
            self._retained_bytes += result.nbytes
        outcomes_count = max(result.outcomes.weights.shape[1] for result in results)
        output_bytes = size * outcomes_count * (1 + events * states) * 8
        # ProjectionReadout owns a second full copy. Include one event's
        # indexing/validation scratch and the diagnostic Python tuple payloads.
        self._check_budget(
            size,
            outcomes_count,
            events,
            2 * output_bytes
            + size * outcomes_count * (states * 8 + 32)
            + size * events * 128,
        )
        weights = np.zeros((size, outcomes_count))
        populations = [np.zeros((size, outcomes_count, states)) for _ in range(events)]
        for values in populations:
            values[..., 0] = 1
        retained: list[tuple[int, ...]] = [()] * size
        pruned: list[tuple[float, ...]] = [()] * size
        one_hot = np.eye(states)
        for plan, result in zip(plans, results, strict=True):
            outcomes = result.outcomes
            count = outcomes.weights.shape[1]
            for point, counts, dropped in zip(
                plan.batch.points, result.counts, result.pruned, strict=True
            ):
                retained[point] = tuple(int(value) for value in counts)
                pruned[point] = tuple(float(value) for value in dropped)
            weights[plan.batch.points, :count] = outcomes.weights
            for event in range(events - 1):
                populations[event][plan.batch.points, :count] = one_hot[
                    outcomes.history[:, event]
                ]
            for index in range(count):
                state = int(outcomes.history[index, -1]) if events > 1 else -1
                populations[-1][plan.batch.points, index] = result.terminal[state]
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
