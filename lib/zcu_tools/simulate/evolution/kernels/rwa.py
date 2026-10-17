"""Reduced rotating-wave Hamiltonian with the full secular dissipator."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularModel
from ._weak import (
    constant_mask,
    drive_ratios,
    freeze,
    generator,
    model_groups,
    model_keys,
    parameters,
)


@njit(nogil=True, cache=True)
def _photons(plus: NDArray[np.complex128]) -> NDArray[np.int64]:
    # A maximum-strength spanning forest assigns a signed photon to each edge.
    # Unlike a global round(E/f), disconnected near-degenerate ladders do not
    # acquire arbitrary relative phases. Off-resonant edges retain detuning.
    size = len(plus)
    numbers = np.zeros(size, dtype=np.int64)
    component = np.arange(size)
    strengths = np.abs(plus) + np.abs(plus.T)
    order = np.argsort(strengths.ravel())[::-1]
    for edge in order:
        i, j = edge // size, edge % size
        if i >= j or strengths[i, j] == 0 or component[i] == component[j]:
            continue
        old = component[j]
        direction = 1 if abs(plus[i, j]) >= abs(plus[j, i]) else -1
        delta = numbers[i] + direction - numbers[j]
        for k in range(size):
            if component[k] == old:
                numbers[k] += delta
                component[k] = component[i]
    return numbers


@njit(nogil=True, cache=True)
def _exponential(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    # Scaling/squaring Taylor on ||A||_inf <= 1/2. This also handles defective
    # rate generators; diagonalizing the driven Liouvillian would not.
    norm = np.max(np.sum(np.abs(matrix), axis=1))
    squarings = max(0, int(np.ceil(np.log2(max(1.0, 2 * norm)))))
    reduced = matrix / 2.0**squarings
    result = np.eye(len(matrix), dtype=np.complex128)
    term = result.copy()
    for order in range(1, 20):
        term = term @ reduced / order
        result += term
    for _ in range(squarings):
        result = result @ result
    return result


@njit(nogil=True, cache=True)
def _rwa_batch(
    params: NDArray[np.float64],
    free: NDArray[np.complex128],
    plus: NDArray[np.complex128],
    states: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    size = len(plus)
    numbers = _photons(plus)
    output = np.empty_like(states)
    for job in range(len(params)):
        frequency, gain, duration, phase = params[job]
        hamiltonian = np.zeros((size, size), dtype=np.complex128)
        for i in range(size):
            hamiltonian[i, i] = -numbers[i] * frequency
            for j in range(size):
                if numbers[j] - numbers[i] == 1:
                    hamiltonian[i, j] = gain * plus[i, j] * np.exp(1j * phase)
                elif numbers[j] - numbers[i] == -1:
                    hamiltonian[i, j] = (
                        gain * plus[j, i].conjugate() * np.exp(-1j * phase)
                    )
        liouvillian = free.copy()
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    liouvillian[i * size + j, k * size + j] -= (
                        2j * np.pi * hamiltonian[i, k]
                    )
                    liouvillian[i * size + j, i * size + k] += (
                        2j * np.pi * hamiltonian[k, j]
                    )
        # Connected components of the reduced Liouvillian include every
        # population (spectator relaxation is never dropped) and every input
        # coherence. Solve independent blocks rather than a full expm.
        evolved = _block_action(liouvillian * duration, states[job].ravel())
        for i in range(size):
            for j in range(size):
                rotation = np.exp(
                    -2j * np.pi * (numbers[i] - numbers[j]) * frequency * duration
                )
                output[job, i, j] = evolved[i * size + j] * rotation
    return output


@njit(nogil=True, cache=True)
def _block_action(
    matrix: NDArray[np.complex128], state: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    dimension = len(state)
    labels = np.arange(dimension)
    for i in range(dimension):
        for j in range(i):
            if matrix[i, j] != 0 or matrix[j, i] != 0:
                old, new = labels[i], labels[j]
                for k in range(dimension):
                    if labels[k] == old:
                        labels[k] = new
    output = np.empty_like(state)
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        block = matrix[indices][:, indices].copy()
        output[indices] = _exponential(block) @ state[indices]
    return output


@dataclass(frozen=True)
class _Entry:
    free: NDArray[np.complex128]
    plus: tuple[NDArray[np.complex128], ...]

    @property
    def nbytes(self) -> int:
        return self.free.nbytes + sum(pair.nbytes for pair in self.plus)


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Entry, ...]

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)


def _prepare_model(model: SecularModel) -> _Entry:
    return _Entry(
        freeze(generator(model)),
        tuple(
            freeze((drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2)
            for drive in model.drives
        ),
    )


class RWAKernel:
    """C5 one-tone constant reduced RWA for r<=0.2.

    This guard also accepts r<=0.1: Dyson precedence and its transfer/r2
    failures belong to engine branch order, not this kernel. The Hamiltonian
    uses a maximum-strength photon forest; all populations and input
    coherences retain their secular bath. No state-dependent truncation occurs.
    A state-independent trace-norm bound on omitted sidebands must also fit
    the budget; otherwise the engine can use its full-carrier T6 fallback.
    """

    name = "rwa"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("RWA requires positive budget")
        accepted = constant_mask(table, rows)
        active = np.flatnonzero(accepted)
        for positions, model in model_groups(table, rows[active], source):
            original = active[positions]
            for port in np.unique(table.tone_port[rows[original], 0]):
                selected = original[table.tone_port[rows[original], 0] == port]
                params = parameters(table, rows[selected])
                one, _ = drive_ratios(model, int(port), params)
                drive = model.drives[int(port)]
                plus = (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
                numbers = _photons(plus)
                omitted = np.where(numbers[None, :] - numbers[:, None] == 1, 0, plus)
                # Duhamel and trace-norm contractivity of both Lindblad
                # channels give ||rho_full-rho_RWA||_1 <= integral
                # 2*||delta H/hbar||_op dt. The omitted Hamiltonian is
                # gain*(B exp(i wt)+B^dagger exp(-i wt)), hence 8*pi
                # *|gain|*tau*||B||_op. Every density entry is bounded
                # by trace norm, so this covers populations and low-four
                # for any input density, phase, carrier and bath. We do
                # not assume rapid oscillations average to zero.
                error = (
                    8
                    * np.pi
                    * abs(params[:, 1])
                    * params[:, 2]
                    * np.linalg.norm(omitted, ord=2)
                )
                accepted[selected] = (one <= 0.2) & (error <= budget)
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Model lookup levels; the engine supplies the DC-model cache namespace."""
        return model_keys(table, rows)

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Relative solve cost; scaling/squaring grows logarithmically with duration."""
        return 1 + np.log1p(table.duration_ns[rows])

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> _Prepared:
        """Copy state-independent generators and drive sidebands per model key."""
        if len(keys) != len(rows) or not np.isfinite(budget) or budget <= 0:
            raise ValueError(
                "RWA requires aligned representative rows and positive budget"
            )
        return _Prepared(
            tuple(
                _prepare_model(source.model(tuple(int(v) for v in table.level[row])))
                for row in rows
            )
        )

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Decline input binding; apply uses the model-only prepared payload."""
        del prepared, key_index, table, rows, source_initial, budget

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> NDArray[np.bool_]:
        """Advance arbitrary input densities in place in the lab eigenbasis.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("RWA requires positive budget")
        if not isinstance(prepared, _Prepared):
            raise ValueError("RWA requires its own prepared payload")
        for key in np.unique(key_index):
            positions = np.flatnonzero(key_index == key)
            entry = prepared.entries[key]
            for port in np.unique(table.tone_port[rows[positions], 0]):
                selected = positions[table.tone_port[rows[positions], 0] == port]
                result = _rwa_batch(
                    parameters(table, rows[selected]),
                    entry.free,
                    entry.plus[port],
                    states[selected],
                )
                if not np.isfinite(result).all():
                    raise RuntimeError(
                        f"RWA nonfinite density in segment rows {rows[selected]}"
                    )
                states[selected] = result
        return np.zeros(len(rows), dtype=np.bool_)
