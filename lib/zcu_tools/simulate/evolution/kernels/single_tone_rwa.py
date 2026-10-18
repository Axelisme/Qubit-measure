"""Constant rotating-frame Lindbladians split by the shared retained graph."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ._rwa_graph import RWA_GRAPH_POLICY, phase_graph, pruning_admitted
from ._secular import secular_operators
from .multitone_rwa import (
    CONSTANT_RWA_POLICY,
    RWACoefficients,
    admit_constant_rows,
    interaction_hamiltonian,
    rwa_coefficients,
)

_POLICY = ("constant-rwa-blocks-v1", RWA_GRAPH_POLICY, CONSTANT_RWA_POLICY, 64)


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


def _build(table: SegmentTable, row: int, context: KernelContext) -> _Block:
    model = context.source.model(tuple(int(v) for v in table.level[row]))
    port, frequency = (
        int(table.tone_port[row, 0]),
        float(table.tone_frequency_ghz[row, 0]),
    )
    graph = phase_graph(model, (port,), (frequency,), context.work_plan)
    original = rwa_coefficients(model, (port,), (frequency,))
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
    _, labels = connected_components(
        np.abs(graph.positive[0]) + np.abs(graph.positive[0].T), directed=False
    )
    components = tuple(np.flatnonzero(labels == label) for label in sorted(set(labels)))
    # All populations and within-component coherences are closed under the
    # full secular population bath. Cross-component coherences close separately.
    coords = [(i, i, 0) for i in range(len(energies))]
    for i in range(len(energies)):
        for j in range(i + 1, len(energies)):
            if labels[i] == labels[j]:
                coords.extend(((i, j, 1), (i, j, 2)))
    a, b, kind = np.array(coords, dtype=np.int64).T.copy()
    population, decay = secular_operators(model)
    generator = _generator(h, population, decay, a, b, kind)
    charges = graph.charges[:, 0].copy()
    for array in (h, decay, generator, a, b, kind, charges, *components):
        array.setflags(write=False)
    return _Block(h, decay, generator, a, b, kind, components, charges)


def _play(
    entry: _Block,
    rho: NDArray[np.complex128],
    frequency: float,
    theta: float,
    duration: float,
) -> NDArray[np.complex128]:
    start = np.exp(1j * entry.charges * theta)
    rho = rho * start.conj()[:, None] * start[None, :]
    diagonal_h = np.diag(entry.h)
    result = rho * np.exp(
        (-2j * np.pi * (diagonal_h[:, None] - diagonal_h[None, :]) - entry.decay)
        * duration
    )
    a, b, kind = entry.a, entry.b, entry.kind
    state = np.where(
        kind == 0,
        rho[a, b],
        np.where(
            kind == 1, (rho[a, b] + rho[b, a]) / 2, (rho[a, b] - rho[b, a]) / (2j)
        ),
    )
    out = expm(entry.generator * duration) @ state
    diagonal, real, imag = kind == 0, kind == 1, kind == 2
    result[a[diagonal], b[diagonal]] = out[diagonal]
    result[a[real], b[real]] = out[real] + 1j * out[imag]
    result[b[real], a[real]] = out[real] - 1j * out[imag]
    for index, ca in enumerate(entry.components):
        for cb in entry.components[index + 1 :]:
            if len(ca) == len(cb) == 1:
                continue  # Scalar closed form already computed above.
            forward, reverse = np.ix_(ca, cb), np.ix_(cb, ca)
            if not np.any(rho[forward] != 0) and not np.any(rho[reverse] != 0):
                continue  # Exact zero RHS remains zero; no tolerance truncation.
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
    end = np.exp(1j * entry.charges * (theta + 2 * np.pi * frequency * duration))
    return result * end[:, None] * end.conj()[None, :]


class SingleToneRWAKernel:
    """Single constant-tone C5 family with arbitrary complex density playback.

    Uses the shared RWA omission, strength and 0.06 rad phase admission, budget
    >=1e-3 and fixed flux. The retained graph must also pass the explicit
    min(budget,1e-3) pruning bound. Preparation never sees density. Each key
    stores a constant rotating-frame Hamiltonian and full secular bath, with
    all populations and within-component coherences in one real generator
    of dimension at most 64 (larger graphs retain the original fallback).
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
        for index, row in enumerate(rows):
            if not accepted[index]:
                continue
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            graph = phase_graph(
                model,
                (int(table.tone_port[row, 0]),),
                (float(table.tone_frequency_ghz[row, 0]),),
                context.work_plan,
            )
            accepted[index] = pruning_admitted(
                graph,
                np.abs(table.tone_amplitude[row, :1]),
                float(table.duration_ns[row]),
                context.budget,
            )
            # Keep the advertised small-block route out of dense full-space
            # exponentials on strongly connected large joint models.
            _, labels = connected_components(
                np.abs(graph.positive[0]) + np.abs(graph.positive[0].T), directed=False
            )
            sizes = np.bincount(labels)
            accepted[index] &= int(np.sum(sizes**2)) <= 64
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash model, policy, budget, port, frequency and signed gain.

        Duration, start and phase affect playback only; no input density enters
        this model-only key. Rows must already satisfy guard before prepare.
        """
        keys = np.empty(len(rows), dtype="V32")
        for index, row in enumerate(rows):
            model = context.source.model(tuple(int(v) for v in table.level[row]))
            identity = (
                _POLICY,
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

        Raises ValueError for misaligned keys/rows. Arrays are fully charged to
        nbytes; no density-dependent data enters resident preparation.
        """
        if len(keys) != len(rows):
            raise ValueError("constant RWA keys and rows must align")
        return _Prepared(tuple(_build(table, int(row), context) for row in rows))

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
