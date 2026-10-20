"""Constant rotating-frame Lindbladians split by the shared retained graph."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ._rwa_graph import RWA_GRAPH_POLICY, phase_graph, pruning_admitted
from ._workload_facts import rwa_topology, secular_bath
from .multitone_rwa import (
    ADAPTIVE_RWA_POLICY,
    RWACoefficients,
    RWARetention,
    admit_constant_rows,
    constant_retentions,
    interaction_hamiltonian,
    rwa_coefficients,
)

_MAX_BLOCK_DIMENSION = 64
_POLICY = (
    "constant-rwa-blocks-v2",
    RWA_GRAPH_POLICY,
    ADAPTIVE_RWA_POLICY,
    _MAX_BLOCK_DIMENSION,
)


@dataclass(frozen=True)
class _Block:
    # Canonical Hamiltonian/decay, active Hermitian coordinates, graph components.
    h: NDArray[np.complex128]
    decay: NDArray[np.float64]
    generator: NDArray[np.float64]
    a: NDArray[np.int64]
    b: NDArray[np.int64]
    kind: NDArray[np.int64]
    components: tuple[NDArray[np.int64], ...]
    labels: NDArray[np.int64]
    charges: NDArray[np.int64]

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.h,
                self.decay,
                self.generator,
                self.a,
                self.b,
                self.kind,
                self.labels,
                self.charges,
                *self.components,
            )
        )


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Block, ...]

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self.entries)


@njit(nogil=True, cache=True)
def _generator(h, population, decay, a, b, kind):
    # Real matrix in a Hermitian basis, applied complex-linearly. This preserves
    # non-Hermitian residuals too; it is NOT a projection onto physical states.
    dimension = len(a)
    result = np.empty((dimension, dimension))
    for r in range(dimension):
        i, j = a[r], b[r]
        for c in range(dimension):
            x, y = a[c], b[c]
            v = 1j if kind[c] == 2 else 1 + 0j
            value = (
                -2j
                * np.pi
                * v
                * ((h[i, x] if j == y else 0j) - (h[y, j] if i == x else 0j))
            )
            if kind[c] != 0:
                value += (
                    -2j
                    * np.pi
                    * v.conjugate()
                    * ((h[i, y] if j == x else 0j) - (h[x, j] if i == y else 0j))
                )
            if i == j:
                if kind[c] == 0:
                    value += population[i, x]
            elif x == i and y == j:
                value -= decay[i, j] * v
            result[r, c] = value.imag if kind[r] == 2 else value.real
    return result


def _build(
    table: SegmentTable,
    row: int,
    context: KernelContext,
    retention: RWARetention | None,
) -> _Block:
    if retention is None:
        raise ValueError("constant RWA requires a single fixed-flux constant carrier")
    model = table.model(context.source, row)
    port, frequency = (
        int(table.tone_port[row, 0]),
        float(table.tone_frequency_ghz[row, 0]),
    )
    graph = phase_graph(model, (port,), (frequency,), context.work_plan, retention)
    original = rwa_coefficients(
        model, (port,), (frequency,), retention, work_plan=context.work_plan
    )
    coefficients = RWACoefficients(
        graph.positive,
        graph.positive.conj().transpose(0, 2, 1).copy(),
        original.diagonal,
    )
    energies = model.energies_ghz + graph.charges[:, 0] * frequency
    energies = energies - energies[0]
    h = np.diag(energies).astype(np.complex128) + interaction_hamiltonian(
        0.0, energies, coefficients, np.array([complex(table.tone_amplitude[row, 0])])
    )
    topology = rwa_topology(graph, context.work_plan)
    # Resident payloads own and charge their arrays even when facts are shared.
    components = tuple(component.copy() for component in topology.components)
    a, b, kind = (v.copy() for v in (topology.a, topology.b, topology.kind))
    bath = secular_bath(model, context.work_plan)
    decay = bath.decay.copy()
    generator = _generator(h, bath.population, decay, a, b, kind)
    charges = graph.charges[:, 0].copy()
    labels = topology.labels.copy()
    for array in (h, decay, generator, a, b, kind, labels, charges, *components):
        array.setflags(write=False)
    return _Block(h, decay, generator, a, b, kind, components, labels, charges)


@njit(nogil=True, cache=True)
def _initial_coordinates(operators, coordinates, topology, rho, theta, duration):
    h, decay = operators
    a, b, kind = coordinates
    charges, labels = topology
    size = len(charges)
    phase = np.exp(1j * charges * theta)
    rotated = np.empty_like(rho)
    result = np.empty_like(rho)
    active = np.zeros((size, size), dtype=np.bool_)
    for i in range(size):
        for j in range(size):
            value = rho[i, j] * phase[i].conjugate() * phase[j]
            rotated[i, j] = value
            result[i, j] = value * np.exp(
                (-2j * np.pi * (h[i, i] - h[j, j]) - decay[i, j]) * duration
            )
            if labels[i] != labels[j] and value != 0:
                # Either direction activates the pair, including non-Hermitian
                # residuals. Only an exact-zero RHS may skip its exponential.
                active[min(labels[i], labels[j]), max(labels[i], labels[j])] = True
    state = np.empty(len(a), dtype=np.complex128)
    for k in range(len(a)):
        i, j = a[k], b[k]
        if kind[k] == 0:
            state[k] = rotated[i, j]
        elif kind[k] == 1:
            state[k] = (rotated[i, j] + rotated[j, i]) / 2
        else:
            state[k] = (rotated[i, j] - rotated[j, i]) / (2j)
    return rotated, result, state, active


@njit(nogil=True, cache=True)
def _finish_coordinates(result, out, a, b, kind):
    # Preparation emits each real/imaginary coordinate pair adjacently.
    # Both coordinates can be complex: this is not Hermitian projection.
    for k in range(len(a)):
        i, j = a[k], b[k]
        if kind[k] == 0:
            result[i, j] = out[k]
        elif kind[k] == 1:
            result[i, j] = out[k] + 1j * out[k + 1]
            result[j, i] = out[k] - 1j * out[k + 1]


@njit(nogil=True, cache=True)
def _finish_frame(result, charges, theta, frequency, duration):
    end = np.exp(1j * charges * (theta + 2 * np.pi * frequency * duration))
    for i in range(len(charges)):
        for j in range(len(charges)):
            result[i, j] *= end[i] * end[j].conjugate()
    return result


def _play(
    entry: _Block,
    rho: NDArray[np.complex128],
    frequency: float,
    theta: float,
    duration: float,
) -> NDArray[np.complex128]:
    rho, result, state, active = _initial_coordinates(
        (entry.h, entry.decay),
        (entry.a, entry.b, entry.kind),
        (entry.charges, entry.labels),
        rho,
        theta,
        duration,
    )
    out = expm(entry.generator * duration) @ state
    _finish_coordinates(result, out, entry.a, entry.b, entry.kind)
    for index, ca in enumerate(entry.components):
        for other in range(index + 1, len(entry.components)):
            cb = entry.components[other]
            if len(ca) == len(cb) == 1 or not active[index, other]:
                continue  # Scalar closed form, or an exact-zero RHS.
            forward, reverse = np.ix_(ca, cb), np.ix_(cb, ca)
            ha, hb = entry.h[np.ix_(ca, ca)], entry.h[np.ix_(cb, cb)]
            generator = (
                -2j
                * np.pi
                * (np.kron(ha, np.eye(len(cb))) - np.kron(np.eye(len(ca)), hb.T))
            )
            generator[np.diag_indices(len(generator))] -= entry.decay[forward].ravel()
            center = np.trace(generator) / len(generator)
            channel = np.exp(center * duration) * expm(
                (generator - center * np.eye(len(generator))) * duration
            )
            result[forward] = (channel @ rho[forward].ravel()).reshape(len(ca), len(cb))
            result[reverse] = (
                (channel @ rho[reverse].conj().T.ravel())
                .reshape(len(ca), len(cb))
                .conj()
                .T
            )
    return _finish_frame(result, entry.charges, theta, frequency, duration)


class SingleToneRWAKernel:
    """Single constant-tone C5 family with arbitrary complex density playback.

    Uses the shared RWA omission, strength and 0.06 rad phase admission, budget
    >=1e-3 and fixed flux. The retained graph must also pass the explicit
    min(budget,1e-3) pruning bound. Preparation never sees density. Each key
    stores a constant rotating-frame Hamiltonian and full secular bath, with
    all populations and within-component coherences in one real generator
    of dimension at most 64 (larger graphs retain the original fallback).
    A 168-point synthetic sweep of chains and two disjoint eight-level blocks
    covered real-generator dimensions through 128, detunings 51/70/110 MHz,
    gains 0.5/1, durations 30/200 ns and phases 0/0.7. Maximum population
    difference against Floquet period playback was 0.010484 (<0.1); halving
    the largest verified dimension gives the 64 cap (2x safety factor).
    The 50 MHz window is only the initial edge set. Each iteration restores
    every over-threshold one-photon edge and the strongest missing first-order
    path per over-threshold two-photon channel and signed time order. The complete
    omission rule is recomputed each round, with no amplitude cutoff.
    The shared omission estimate, Fourier terms and virtual diagonal all use
    that same final set; incompatible phase cycles still require pruning.
    Other component pairs use small complex generators; exact-zero input blocks
    skip their exponential only at apply. No clipping, normalization, input
    reduction, dense full-space channel or Strang splitting is used.
    This is a calibrated RWA approximation, not an error certificate.
    """

    name = "single_tone_rwa"
    revision = _POLICY
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return row-local single constant RWA and retained-graph admissions."""
        accepted = admit_constant_rows(table, rows, context)
        if not np.any(accepted):
            return accepted
        retentions = constant_retentions(table, context)
        for index, row in enumerate(rows):
            if not accepted[index]:
                continue
            model = table.model(context.source, row)
            port = int(table.tone_port[row, 0])
            frequency = float(table.tone_frequency_ghz[row, 0])
            retention = retentions.rows[row]
            graph = phase_graph(
                model, (port,), (frequency,), context.work_plan, retention
            )
            accepted[index] = pruning_admitted(
                graph,
                np.abs(table.tone_amplitude[row, :1]),
                float(table.duration_ns[row]),
                context.budget,
            )
            # Keep the advertised small-block route out of dense full-space
            # exponentials on strongly connected large joint models.
            topology = rwa_topology(graph, context.work_plan)
            accepted[index] &= len(topology.a) <= _MAX_BLOCK_DIMENSION
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash model, policy, final retained set, budget, port, carrier and gain.

        Duration affects the retained set, not identity within a resolved set.
        Start/phase affect playback only; no input density enters this model-only
        key. Rows without a retention receive inventory-only keys, disjoint
        from retained-set keys. Only guard-accepted rows may prepare or apply.
        """
        keys = np.empty(len(rows), dtype="V32")
        retentions = constant_retentions(table, context)
        for index, row in enumerate(rows):
            model = table.model(context.source, row)
            retention = retentions.rows[row]
            identity = (
                _POLICY,
                retention.edges if retention is not None else None,
                model.identity,
                model.level,
                context.budget,
                int(table.tone_port[row, 0]),
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_amplitude[row, 0]),
            )
            keys[index] = np.void(sha256(repr(identity).encode()).digest())
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate one small constant-generator exponential per row."""
        del table
        return np.full(len(rows), 4.0, dtype=np.float64)

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build immutable block generators aligned with representative keys.

        Raises ValueError for misaligned keys/rows or rows without a retention.
        Arrays are fully charged to nbytes; no density-dependent data enters
        resident preparation.
        """
        if len(keys) != len(rows):
            raise ValueError("constant RWA keys and rows must align")
        retentions = constant_retentions(table, context)
        return _Prepared(
            tuple(
                _build(table, int(row), context, retentions.rows[row]) for row in rows
            )
        )

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """No guard preparation handoff; ordinary admission does not build."""
        del prepared, table, rows, context

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Use the ordinary guard without preparing any numeric generator."""
        del table, rows, context

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """No input binding; full complex density playback remains linear."""
        del prepared, key_index, table, rows, source_initial, context

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Evolve in place, restoring lab endpoints; return all-False exceedances.

        states is (rows,M,M) in the model eigenbasis. Reject invalid payloads,
        row plans, indices and shapes with ValueError. Numerical failures
        propagate, without retry or rerouting. Arbitrary complex residuals and
        cross-component coherences are retained, not symmetrized or discarded.
        """
        if context.plan is not None or not isinstance(prepared, _Prepared):
            raise ValueError("constant RWA requires its own payload and no row plan")
        if (
            key_index.shape != rows.shape
            or states.ndim != 3
            or len(states) != len(rows)
        ):
            raise ValueError("constant RWA state/index shape mismatch")
        for index, row in enumerate(rows):
            key = int(key_index[index])
            if not 0 <= key < len(prepared.entries):
                raise ValueError("constant RWA prepared index out of range")
            entry = prepared.entries[key]
            if states.shape[1:] != entry.h.shape:
                raise ValueError("constant RWA state dimension mismatch")
            states[index] = _play(
                entry,
                states[index],
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_theta_rad[row, 0]),
                float(table.duration_ns[row]),
            )
        return np.zeros(len(rows), dtype=np.bool_)
