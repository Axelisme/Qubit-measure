"""DC rank-1 projection chains with one continuation solve per source state."""

from dataclasses import dataclass, replace
from hashlib import sha256
from math import prod
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.adc import ProjectionChain, normalize_populations
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


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Author-only projection approximation and work, not measured IQ.

    points: flat request point IDs, in the same order as the following rows.
    continued_states: per-point counts of adopted solver rows for each transition.
        Counts use the adoption mask, even when the solver row is identity.
    uncontinued_probability: per-point source marginal mass on the complement
        of that mask at each transition. Its row sum bounds path total variation.
    Both rows have reads-1 entries, excluding prefix and anything after terminal.
    evolution_calls: actual Engine.evolve calls, including prefixes. Adoption
        counts cannot determine calls: a solved row may be unused at some points.
    """

    points: tuple[int, ...]
    continued_states: tuple[tuple[int, ...], ...]
    uncontinued_probability: tuple[tuple[float, ...], ...]
    evolution_calls: int


@dataclass(frozen=True)
class ProjectionEvolution:
    """Noiseless projection chain and evolution/reuse diagnostics.

    readout: first-event probabilities and subsequent [source,target] rows.
    paths: merged engine diagnostics from all prefixes and continuations.
    reuse_keys: exact span digests, including initial densities and time bounds;
        validate every key with Engine.reuse before reusing this preparation.
    diagnostics: per-point transition approximation and actual work counts.
    """

    readout: ProjectionChain
    paths: PathDiagnostics
    reuse_keys: tuple[bytes, ...]
    diagnostics: ProjectionDiagnostics


class ProjectionRunner:
    """Prepare DC-energy Markov chains without enumerating outcome histories.

    engine/source are the source-owned evolution resources. basis is the same
    retained DC energy basis as source densities and ADC labels. floor is the
    finite threshold in [0,1): each point adopts solver rows only for source
    states whose approximate marginal is at least floor; other rows are identity.
    cancel is forwarded unchanged. One runner serves one preparation.
    Densities never cross a plan or span; ADC CTMC jumps do not feed back.
    paths retains completed span diagnostics even when prepare raises.
    evolve_wall_s sums every Engine call's wall seconds, including a failed call.
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
        self.evolve_wall_s = 0.0
        self.keys: list[bytes] = []

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
        started = monotonic()
        try:
            result = self.engine.evolve(
                self.source,
                (batch,),
                cancel=self.cancel,
                reuse_key=reuse_key,
                telemetry=True,
            )
        finally:
            self.evolve_wall_s += monotonic() - started
        self.keys.append(reuse_key)
        self.paths.append(result.paths)
        indices = {int(point): index for index, point in enumerate(result.points)}
        order = [indices[int(point)] for point in plan.batch.points]
        return result.densities[order]

    def _populations(self, density: NDArray[np.complex128]) -> NDArray[np.float64]:
        return normalize_populations(batch_populations(density, self.basis, self.basis))

    def prepare(
        self,
        plans: tuple[SingleFluxoniumPlan, ...],
        point_dims: tuple[int, ...],
        key: bytes,
    ) -> ProjectionEvolution:
        """Evolve plans into one immutable chain in C-order point_dims layout.

        Plans cover every request point once and share a chronological event
        count. key hashes all caller inputs. Prefix evolution gives the first
        event's populations. Subsequent transitions evolve each required rank-1
        source state once per plan, for all its points. Adoption is point-local:
        rows below floor remain identity even if solved for another point.
        Zero floor is exact, including zero-probability states. Numerical and
        cancellation errors propagate without a result.
        """
        states, size = len(self.basis.labels), prod(point_dims)
        events = len(plans[0].read_events)
        initial = np.empty((size, states))
        transitions = [np.empty((size, states, states)) for _ in range(events - 1)]
        continued: list[tuple[int, ...]] = [()] * size
        uncontinued: list[tuple[float, ...]] = [()] * size
        for plan in plans:
            points = plan.batch.points
            initial[points] = self._populations(self._advance(plan, 0, None, key))
            marginal = initial[points]
            counts = np.empty((len(points), events - 1), dtype=np.int64)
            losses = np.empty((len(points), events - 1))
            for event, transition in enumerate(transitions, start=1):
                adopted = marginal >= self.floor
                counts[:, event - 1] = np.count_nonzero(adopted, axis=-1)
                losses[:, event - 1] = np.where(adopted, 0, marginal).sum(axis=-1)
                rows = np.broadcast_to(
                    np.eye(states), (len(points), states, states)
                ).copy()
                # Each new span still goes through Engine.evolve's routing
                # policy; complete reuse validates every span key (ADR-0078).
                for state in np.flatnonzero(np.any(adopted, axis=0)):
                    projected = np.zeros(
                        (len(points), states, states), dtype=np.complex128
                    )
                    projected[:, state, state] = 1
                    solved = self._populations(
                        self._advance(plan, event, projected, key)
                    )
                    rows[adopted[:, state], state] = solved[adopted[:, state]]
                transition[points] = rows
                marginal = np.einsum("ps,pst->pt", marginal, rows)
            for index, point in enumerate(points):
                continued[point] = tuple(int(value) for value in counts[index])
                uncontinued[point] = tuple(float(value) for value in losses[index])
        return ProjectionEvolution(
            ProjectionChain(
                initial.reshape((*point_dims, states)),
                tuple(
                    values.reshape((*point_dims, states, states))
                    for values in transitions
                ),
            ),
            merge_path_diagnostics(self.paths),
            tuple(self.keys),
            ProjectionDiagnostics(
                tuple(range(size)), tuple(continued), tuple(uncontinued), len(self.keys)
            ),
        )
