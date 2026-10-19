"""Shared joint-RWA phase-charge facts and explicit weak-cycle admission."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..secular import SecularModel
from ..workplan import FactKey, WorkPlan
from .multitone_rwa import RWA_POLICY, RWARetention, retained_edges

ZERO_GHZ = 1e-12
PRUNE_LIMIT = 1e-3
RWA_GRAPH_POLICY = ("joint-phase-graph-pruned-v1", RWA_POLICY, ZERO_GHZ, PRUNE_LIMIT)


@dataclass(frozen=True)
class PhaseGraph:
    """Model-only phase reduction with charged omissions, never an admission.

    charges: (M,tones) integer diagonal rotation coordinates in dressed ordering.
    positive: (tones,M,M) compatible retained Fourier coefficients, GHz/gain.
    omitted: (tones,) spectral norm of all removed positive edges, including
        numerical zeros. Multiplying by gain, peak and duration gives the bound.
    """

    charges: NDArray[np.int64]
    positive: NDArray[np.complex128]
    omitted: NDArray[np.float64]


def phase_graph(
    model: SecularModel,
    ports: tuple[int, ...],
    frequencies: tuple[float, ...],
    plan: WorkPlan,
    retention: RWARetention | None = None,
) -> PhaseGraph:
    """Publish deterministic dressed charges and explicit RWA edge omissions.

    ports/frequencies name each ordered carrier; negative frequencies work too.
    retention supplies resolved Fourier edges instead of the initial window;
    its final edge coordinates enter the fact identity. For a fixed retained set
    this model-only fact is independent of gains, shape, duration and phase.
    It uses the shared RWA retained operator, then strongest-first integer
    constraints q[i]-q[j]=e[k]. Incompatible cycles are omitted, not silently
    declared covariant. Consumers MUST apply pruning_admitted before use.
    Empty or misaligned carriers raise ValueError. Outputs are read-only copies.
    """
    if not ports or len(ports) != len(frequencies):
        raise ValueError("phase graph requires aligned nonempty carriers")
    count = len(ports)
    identity = (
        "joint-rwa-phase-graph",
        RWA_GRAPH_POLICY,
        model.identity,
        model.level,
        ports,
        frequencies,
        None if retention is None else retention.edges,
    )

    def build() -> PhaseGraph:
        original = np.array(
            [
                retained_edges(
                    model,
                    ports[k],
                    frequencies[k],
                    retention,
                    k,
                )
                for k in range(count)
            ]
        )
        positive = np.zeros_like(original)
        size = len(model.energies_ghz)
        charges = np.zeros((size, count), dtype=np.int64)
        components = np.arange(size)
        # Deterministic strongest-edge spanning constraints. Weak inconsistent
        # edges and numerical zeros are explicit omissions, charged below.
        edges = [
            (abs(original[k, i, j]), k, i, j)
            for k, i, j in np.argwhere(np.abs(original) > ZERO_GHZ)
        ]
        for _, k, i, j in sorted(edges, key=lambda edge: (-edge[0], *edge[1:])):
            desired = np.eye(count, dtype=np.int64)[k]
            delta = desired - (charges[i] - charges[j])
            if components[i] == components[j]:
                if np.any(delta):
                    continue
            else:
                group = components == components[i]
                charges[group] += delta
                components[group] = components[j]
            positive[k, i, j] = original[k, i, j]
        omitted = np.array(
            [np.linalg.norm(a - b, 2) for a, b in zip(original, positive, strict=True)],
            dtype=np.float64,
        )
        return PhaseGraph(charges, positive, omitted)

    return plan.fact(FactKey(identity, PhaseGraph), build)


def pruning_bound(
    graph: PhaseGraph, peak_amplitudes: NDArray[np.float64], duration_ns: float
) -> float:
    """Return a state-independent per-row population omission bound.

    peak_amplitudes is nonnegative abs(gain)*envelope peak, one per carrier.
    Hamiltonian removal has norm <=2 sum(amplitude*omitted), in GHz.
    Duhamel and trace-distance contraction bound any population difference by
    4*pi*duration*sum(amplitude*omitted). This is NOT a sequence or RWA-vs-lab
    certificate. Invalid dimensions, signs or nonfinite values raise ValueError.
    """
    if (
        peak_amplitudes.shape != graph.omitted.shape
        or np.any(peak_amplitudes < 0)
        or not np.isfinite(peak_amplitudes).all()
        or not np.isfinite(duration_ns)
        or duration_ns < 0
    ):
        raise ValueError("invalid phase-graph pruning amplitudes or duration")
    return float(4 * np.pi * duration_ns * (peak_amplitudes @ graph.omitted))


def pruning_admitted(
    graph: PhaseGraph,
    peak_amplitudes: NDArray[np.float64],
    duration_ns: float,
    budget: float,
) -> bool:
    """Admit named weak-cycle pruning only below min(budget,1e-3) per row.

    Exact graphs have zero bound. Larger or incompatible strong cycles retain
    ordinary row RWA. Consumers still own RWA and algorithm-specific guards.
    Nonpositive/nonfinite budgets raise ValueError; never reads density.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("pruning budget must be finite and positive")
    return pruning_bound(graph, peak_amplitudes, duration_ns) <= min(
        budget, PRUNE_LIMIT
    )
