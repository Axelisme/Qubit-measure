"""Full-carrier second-order modal Dyson evolution, with model-only preparation."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularModel
from ._weak import (
    commutator,
    constant_mask,
    drive_ratios,
    freeze,
    generator,
    model_groups,
    model_keys,
    parameters,
)


@njit(nogil=True, cache=True)
def _first(x: complex, y: complex) -> complex:
    # exp[x,y], ordered so expm1 never overflows from a positive real gap.
    if y.real > x.real:
        x, y = y, x
    delta = y - x
    if abs(delta) < 1e-5:
        return np.exp(x) * (
            1 + delta / 2 + delta**2 / 6 + delta**3 / 24 + delta**4 / 120
        )
    return np.exp(x) * np.expm1(delta) / delta


@njit(nogil=True, cache=True)
def _second(x: complex, y: complex, z: complex) -> complex:
    # Put the farthest pair at the endpoints to avoid a small denominator.
    if abs(x - y) > max(abs(x - z), abs(y - z)):
        y, z = z, y
    elif abs(y - z) > abs(x - z):
        x, y = y, x
    if abs(x - z) >= 0.5:
        return (_first(x, y) - _first(y, z)) / (x - z)
    # Confluent divided difference: exp(mean)*sum h_k(delta)/(k+2)!.
    # This includes repeated eigenvalues, zero duration and exact resonance.
    mean = (x + y + z) / 3
    h = np.zeros(22, dtype=np.complex128)
    h[0] = 1
    for delta in (x - mean, y - mean, z - mean):
        for k in range(1, 22):
            h[k] += delta * h[k - 1]
    value = 0j
    factorial = 2.0
    for k in range(22):
        value += h[k] / factorial
        factorial *= k + 3
    return np.exp(mean) * value


@njit(nogil=True, cache=True)
def evaluate_dyson(
    params: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128],
    initial_modes: NDArray[np.complex128],
    couplings: NDArray[np.complex128],
    pruning: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Array-only backend: return modal states for a batch of one-port jobs.

    params (jobs,4): frequency GHz, signed gain, duration ns, phase rad.
    eigenvalues (D,): free Liouvillian modes, real parts nonpositive.
    initial_modes (jobs,D): input densities transformed by W inverse.
    couplings (2,D,D): W inverse L(A±) W, signs -1,+1.
    pruning (D+1,): max(abs(W[:,mode])) followed by positive error budget.
    Inputs are not changed. Output is complex128 (jobs,D). Nonfinite output
    is checked by the Kernel adapter. No model, table or Python object enters
    this function, allowing a later array backend to replace it.
    """
    jobs, dimension = initial_modes.shape
    result = np.empty_like(initial_modes)
    budget = pruning[-1]
    # Each discarded path gets an equal share of budget/8. Physical states
    # have Frobenius norm <=1; here the bound uses the actual modal input.
    allowance = budget / (8 * (2 * dimension**2 + 4 * dimension**3))
    for job in range(jobs):
        frequency, gain, duration, phase = params[job]
        omega = 2 * np.pi * frequency
        scaled = eigenvalues * duration
        growth = np.exp(max(0.0, np.max(eigenvalues.real)) * duration)
        for a in range(dimension):
            value = np.exp(scaled[a]) * initial_modes[job, a]
            for outer in range(2):
                r = 2 * outer - 1
                factor = gain * np.exp(1j * r * phase)
                for b in range(dimension):
                    coefficient = couplings[outer, a, b]
                    if coefficient == 0:
                        continue
                    first = coefficient * initial_modes[job, b]
                    bound = abs(first * gain) * duration * growth * pruning[a]
                    if abs(first) > 1e-16 or bound > allowance:
                        value += (
                            factor
                            * first
                            * duration
                            * _first(scaled[a], scaled[b] + 1j * r * omega * duration)
                        )
                    for inner in range(2):
                        s = 2 * inner - 1
                        factor2 = gain**2 * np.exp(1j * (r + s) * phase)
                        for c in range(dimension):
                            second = (
                                coefficient
                                * couplings[inner, b, c]
                                * initial_modes[job, c]
                            )
                            if second == 0:
                                continue
                            bound = (
                                abs(second * gain**2)
                                * duration**2
                                / 2
                                * growth
                                * pruning[a]
                            )
                            if abs(second) <= 1e-12 and bound <= allowance:
                                continue
                            value += (
                                factor2
                                * second
                                * duration**2
                                * _second(
                                    scaled[a],
                                    scaled[b] + 1j * r * omega * duration,
                                    scaled[c] + 1j * (r + s) * omega * duration,
                                )
                            )
            result[job, a] = value
    return result


@dataclass(frozen=True)
class _Entry:
    values: NDArray[np.complex128]
    vectors: NDArray[np.complex128]
    inverse: NDArray[np.complex128]
    couplings: tuple[NDArray[np.complex128], ...]

    @property
    def nbytes(self) -> int:
        return sum(
            v.nbytes for v in (self.values, self.vectors, self.inverse, *self.couplings)
        )


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Entry, ...]

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)


def _prepare_model(model: SecularModel) -> _Entry:
    # Use the exact secular block structure instead of a dense eigensolve.
    # Coherences already are eigenmodes; only the population Q needs factoring.
    size = len(model.energies_ghz)
    free = generator(model)
    populations = np.arange(size) * (size + 1)
    values = np.diag(free).copy()
    vectors = np.eye(size**2, dtype=np.complex128)
    try:
        q, rotation = np.linalg.eig(free[np.ix_(populations, populations)])
        values[populations] = q
        vectors[np.ix_(populations, populations)] = rotation
        if np.linalg.cond(vectors) > 1e10:
            raise RuntimeError("Dyson population modes are ill-conditioned")
        inverse = np.linalg.inv(vectors).astype(np.complex128)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Dyson modal decomposition failed") from error
    couplings = []
    for drive in model.drives:
        pair = np.stack(
            [
                inverse
                @ commutator(
                    (drive.in_phase_ghz - sign * 1j * drive.quadrature_ghz) / 2
                )
                @ vectors
                for sign in (-1, 1)
            ]
        )
        couplings.append(freeze(pair))
    return _Entry(freeze(values), freeze(vectors), freeze(inverse), tuple(couplings))


def _advance(
    entry: _Entry,
    port: int,
    params: NDArray[np.float64],
    states: NDArray[np.complex128],
    budget: float,
) -> NDArray[np.complex128]:
    modes = states.reshape(len(states), -1) @ entry.inverse.T
    pruning = np.r_[np.max(abs(entry.vectors), axis=0), budget]
    evolved = evaluate_dyson(
        params, entry.values, modes, entry.couplings[port], pruning
    )
    return (evolved @ entry.vectors.T).reshape(states.shape)


class DysonKernel:
    """C5 weak one-tone constant kernel; engine owns known-state eligibility.

    Guards use r<=0.1, r2<=0.05 and the initial-density second-order population
    change <=0.02. Prepared modes are shared across frequency, gain, duration,
    phase and port. Preparation never reads a density. apply accepts arbitrary
    densities, not only diagonal or stationary states.
    """

    name = "dyson"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Dyson requires positive budget")
        accepted = constant_mask(table, rows)
        active = np.flatnonzero(accepted)
        for positions, model in model_groups(table, rows[active], source):
            original = active[positions]
            entry = _prepare_model(model)
            for port in np.unique(table.tone_port[rows[original], 0]):
                selected = original[table.tone_port[rows[original], 0] == port]
                params = parameters(table, rows[selected])
                one, two = drive_ratios(model, int(port), params)
                dc_model = source.model((0,) * len(model.level))
                overlap = model.eigenvectors.conj().T @ dc_model.eigenvectors
                initial = overlap @ source.initial_density @ overlap.conj().T
                states = np.repeat(initial[None], len(selected), axis=0)
                driven = _advance(entry, int(port), params, states, budget)
                zero = params.copy()
                zero[:, 1] = 0
                free = _advance(entry, int(port), zero, states, budget)
                transfer = np.max(
                    abs(np.diagonal(driven - free, axis1=1, axis2=2)), axis=1
                )
                accepted[selected] = (one <= 0.1) & (two <= 0.05) & (transfer <= 0.02)
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Model-level lookup keys, namespaced by the engine's DC model identity."""
        return model_keys(table, rows)

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Closed-form work does not scale with pulse length or carrier frequency."""
        return np.ones_like(table.duration_ns[rows])

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> _Prepared:
        """Factor one secular free generator per representative model row."""
        if len(keys) != len(rows) or not np.isfinite(budget) or budget <= 0:
            raise ValueError(
                "Dyson requires aligned representative rows and positive budget"
            )
        return _Prepared(
            tuple(
                _prepare_model(source.model(tuple(int(v) for v in table.level[row])))
                for row in rows
            )
        )

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Advance states in place; rows/key_index may be non-contiguous or reversed."""
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Dyson requires positive budget")
        if not isinstance(prepared, _Prepared):
            raise ValueError("Dyson requires its own prepared payload")
        for key in np.unique(key_index):
            positions = np.flatnonzero(key_index == key)
            for port in np.unique(table.tone_port[rows[positions], 0]):
                selected = positions[table.tone_port[rows[positions], 0] == port]
                result = _advance(
                    prepared.entries[key],
                    int(port),
                    parameters(table, rows[selected]),
                    states[selected],
                    budget,
                )
                if not np.isfinite(result).all():
                    raise RuntimeError(
                        f"Dyson nonfinite density in segment rows {rows[selected]}"
                    )
                states[selected] = result
