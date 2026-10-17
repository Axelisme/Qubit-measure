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
def _first(x: complex, y: complex, exp_x: complex, exp_y: complex) -> complex:
    # exp[x,y], ordered so expm1 never overflows from a positive real gap.
    if y.real > x.real:
        x, y = y, x
        exp_x, exp_y = exp_y, exp_x
    delta = y - x
    if abs(delta) < 1e-5:
        return exp_x * (1 + delta / 2 + delta**2 / 6 + delta**3 / 24 + delta**4 / 120)
    if abs(delta) > 0.5:
        return (exp_y - exp_x) / delta
    return exp_x * np.expm1(delta) / delta


@njit(nogil=True, cache=True)
def _second(
    x: complex,
    y: complex,
    z: complex,
    exp_x: complex,
    exp_y: complex,
    exp_z: complex,
) -> complex:
    # Put the farthest pair at the endpoints to avoid a small denominator.
    if abs(x - y) > max(abs(x - z), abs(y - z)):
        y, z = z, y
        exp_y, exp_z = exp_z, exp_y
    elif abs(y - z) > abs(x - z):
        x, y = y, x
        exp_x, exp_y = exp_y, exp_x
    if abs(x - z) >= 0.5:
        return (_first(x, y, exp_x, exp_y) - _first(y, z, exp_y, exp_z)) / (x - z)
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
def _build_paths(
    params: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128],
    modes: NDArray[np.complex128],
    couplings: NDArray[np.complex128],
    pruning: NDArray[np.float64],
    outputs: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Contract one input into paths; retain a superset for all batch parameters."""
    dimension = len(modes)
    # Each discarded path gets an equal share of budget/8. The actual modal
    # input and max gain/duration bound every row; thresholds remain unchanged.
    allowance = pruning[-1] / (8 * (2 * dimension**2 + 4 * dimension**3))
    gain = np.max(np.abs(params[:, 1]))
    duration = np.max(params[:, 2])
    growth = np.exp(max(0.0, np.max(eigenvalues.real)) * duration)
    indices = []
    coefficients = []
    for a in outputs:
        for outer in range(2):
            r = 2 * outer - 1
            for b in range(dimension):
                coefficient = couplings[outer, a, b]
                if coefficient == 0:
                    continue
                first = coefficient * modes[b]
                bound = abs(first * gain) * duration * growth * pruning[a]
                if abs(first) > 1e-16 or bound > allowance:
                    # c=-1 marks first order; all other records are second order.
                    indices.append((a, b, -1, r, 0))
                    coefficients.append(first)
                for inner in range(2):
                    s = 2 * inner - 1
                    for c in range(dimension):
                        second = coefficient * couplings[inner, b, c] * modes[c]
                        bound = (
                            abs(second * gain**2)
                            * duration**2
                            / 2
                            * growth
                            * pruning[a]
                        )
                        if abs(second) > 1e-12 or bound > allowance:
                            indices.append((a, b, c, r, s))
                            coefficients.append(second)
    return (
        np.array(indices, dtype=np.int64).reshape((-1, 5)),
        np.array(coefficients, dtype=np.complex128),
    )


@njit(nogil=True, cache=True)
def _group_first_factors(
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Index unique (a,b,r) factors, including both edges of second-order paths."""
    lookup = {}
    pairs = []
    path_pairs = np.full((len(indices), 2), -1, dtype=np.int64)
    for path in range(len(indices)):
        a, b, c, r, s = indices[path]
        key = (a, b, r)
        if key not in lookup:
            lookup[key] = len(pairs)
            pairs.append(key)
        path_pairs[path, 0] = lookup[key]
        if c != -1:
            key = (b, c, s)
            if key not in lookup:
                lookup[key] = len(pairs)
                pairs.append(key)
            path_pairs[path, 1] = lookup[key]
    return np.array(pairs, dtype=np.int64).reshape((-1, 3)), path_pairs


@njit(nogil=True, cache=True)
def _evaluate_first_factors(
    scaled: NDArray[np.complex128],
    exponentials: NDArray[np.complex128],
    carriers: NDArray[np.complex128],
    omega_time: float,
    pairs: NDArray[np.int64],
) -> NDArray[np.complex128]:
    """Return exp[scaled[a], scaled[b]+i*r*omega_time] once per unique pair."""
    factors = np.empty(len(pairs), dtype=np.complex128)
    for pair in range(len(pairs)):
        a = int(pairs[pair, 0])
        b = int(pairs[pair, 1])
        r = int(pairs[pair, 2])
        factors[pair] = _first(
            scaled[a],
            scaled[b] + 1j * r * omega_time,
            exponentials[a],
            exponentials[b] * carriers[r + 2],
        )
    return factors


@njit(nogil=True, cache=True)
def _evaluate_paths(
    params: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128],
    modes: NDArray[np.complex128],
    pruning: NDArray[np.float64],
    indices: NDArray[np.int64],
    coefficients: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Evaluate contracted paths, retaining the original per-row pruning bound."""
    dimension = len(modes)
    allowance = pruning[-1] / (8 * (2 * dimension**2 + 4 * dimension**3))
    result = np.empty((len(params), dimension), dtype=np.complex128)
    pairs, path_pairs = _group_first_factors(indices)
    # With equal |gain| and duration, the table already applied each row's exact
    # bound. Mixed batches still recheck it below, preserving row splitting.
    uniform_bound = np.all(np.abs(params[:, 1]) == abs(params[0, 1])) and np.all(
        params[:, 2] == params[0, 2]
    )
    for job in range(len(params)):
        frequency, gain, duration, phase = params[job]
        scaled = eigenvalues * duration
        exponentials = np.exp(scaled)
        result[job] = exponentials * modes
        if gain == 0 or duration == 0:
            continue
        omega_time = 2 * np.pi * frequency * duration
        carriers = np.exp(1j * np.arange(-2, 3) * omega_time)
        divided_firsts = _evaluate_first_factors(
            scaled, exponentials, carriers, omega_time, pairs
        )
        growth = np.exp(max(0.0, np.max(eigenvalues.real)) * duration)
        first_factors = gain * np.exp(1j * np.array([-1, 1]) * phase)
        second_factors = gain**2 * np.exp(1j * np.array([-2, 0, 2]) * phase)
        for path in range(len(indices)):
            a, b, c, r, s = indices[path]
            coefficient = coefficients[path]
            if c == -1:
                if (
                    uniform_bound
                    or abs(coefficient) > 1e-16
                    or abs(coefficient * gain) * duration * growth * pruning[a]
                    > allowance
                ):
                    result[job, a] += (
                        first_factors[(r + 1) // 2]
                        * coefficient
                        * duration
                        * divided_firsts[path_pairs[path, 0]]
                    )
            else:
                if (
                    uniform_bound
                    or abs(coefficient) > 1e-12
                    or abs(coefficient * gain**2)
                    * duration**2
                    / 2
                    * growth
                    * pruning[a]
                    > allowance
                ):
                    endpoint_gap = scaled[a] - scaled[c] - 1j * (r + s) * omega_time
                    if abs(endpoint_gap) >= 0.5:
                        second = (
                            divided_firsts[path_pairs[path, 0]]
                            - carriers[r + 2] * divided_firsts[path_pairs[path, 1]]
                        ) / endpoint_gap
                    else:
                        # Reorder close endpoints; confluent triples keep the
                        # stable series rather than subtracting nearly equal F.
                        second = _second(
                            scaled[a],
                            scaled[b] + 1j * r * omega_time,
                            scaled[c] + 1j * (r + s) * omega_time,
                            exponentials[a],
                            exponentials[b] * carriers[r + 2],
                            exponentials[c] * carriers[r + s + 2],
                        )
                    result[job, a] += (
                        second_factors[(r + s + 2) // 2]
                        * coefficient
                        * duration**2
                        * second
                    )
    return result


@njit(nogil=True, cache=True)
def _evaluate_batch(
    params: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128],
    initial_modes: NDArray[np.complex128],
    couplings: NDArray[np.complex128],
    pruning: NDArray[np.float64],
    outputs: NDArray[np.int64],
) -> NDArray[np.complex128]:
    """Share contracted paths for equal inputs; handle distinct states rowwise."""
    if len(params) == 0:
        return initial_modes.copy()
    if np.all(params[:, 1] == 0) or np.all(params[:, 2] == 0):
        return np.exp(params[:, 2:3] * eigenvalues) * initial_modes
    if np.all(initial_modes == initial_modes[0]):
        indices, coefficients = _build_paths(
            params, eigenvalues, initial_modes[0], couplings, pruning, outputs
        )
        return _evaluate_paths(
            params, eigenvalues, initial_modes[0], pruning, indices, coefficients
        )
    result = np.empty_like(initial_modes)
    for job in range(len(params)):
        indices, coefficients = _build_paths(
            params[job : job + 1],
            eigenvalues,
            initial_modes[job],
            couplings,
            pruning,
            outputs,
        )
        result[job] = _evaluate_paths(
            params[job : job + 1],
            eigenvalues,
            initial_modes[job],
            pruning,
            indices,
            coefficients,
        )[0]
    return result


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
    this function. Equal inputs share a batch-local contracted path table;
    distinct inputs use the same pruning and evaluation row by row.
    """
    return _evaluate_batch(
        params,
        eigenvalues,
        initial_modes,
        couplings,
        pruning,
        np.arange(len(eigenvalues), dtype=np.int64),
    )


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


def _population_transfer(
    entry: _Entry,
    port: int,
    params: NDArray[np.float64],
    initial: NDArray[np.complex128],
    budget: float,
) -> NDArray[np.float64]:
    """Compute driven-minus-free populations, including coherent linear terms."""
    size = len(initial)
    populations = np.arange(size, dtype=np.int64) * (size + 1)
    modes = initial.ravel() @ entry.inverse.T
    pruning = np.r_[np.max(abs(entry.vectors), axis=0), budget]
    evolved = _evaluate_batch(
        params,
        entry.values,
        np.repeat(modes[None], len(params), axis=0),
        entry.couplings[port],
        pruning,
        populations,
    )
    free = np.exp(params[:, 2:3] * entry.values) * modes
    # Secular population eigenvectors have support only on population indices.
    # No coherence-output paths can contribute to this projection.
    transfer = (evolved - free) @ entry.vectors[populations].T
    return np.max(abs(transfer), axis=1)


class DysonKernel:
    """C5 weak one-tone constant kernel; engine owns known-state eligibility.

    Guards use r<=min(0.1, 0.25*budget**(1/3)), r2<=0.05 and the initial-density
    second-order population change <=min(0.02, (1.5*budget)**(2/3)). The r limit
    follows far-detuned third-order coherence error proportional to r**3; its
    coefficient is calibrated on a synthetic scan with margin (r<=0.025 at
    budget=1e-3). The population limit follows the two-level resonant third-order
    coherence error approximately (2/3)*abs(delta_p)**(3/2) <= budget (0.0131 at
    budget=1e-3). Neither threshold is a rigorous multilevel or coherent-input
    remainder bound. Guard failures follow the engine's reduced RWA/T6 path.
    Prepared modes are shared across frequency, gain, duration, phase and port.
    Preparation never reads a density. apply accepts arbitrary densities, not
    only diagonal or stationary states.
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
            entry = None
            for port in np.unique(table.tone_port[rows[original], 0]):
                selected = original[table.tone_port[rows[original], 0] == port]
                params = parameters(table, rows[selected])
                one, two = drive_ratios(model, int(port), params)
                eligible = (one <= min(0.1, 0.25 * budget ** (1 / 3))) & (two <= 0.05)
                accepted[selected] = False
                selected, params = selected[eligible], params[eligible]
                if not len(selected):
                    continue
                if entry is None:
                    entry = _prepare_model(model)
                dc_model = source.model((0,) * len(model.level))
                initial = source.initial_density
                # Equal bases are exactly the same coordinates. Multiplying a
                # near-identity overlap would introduce spurious coherence.
                if not np.array_equal(model.eigenvectors, dc_model.eigenvectors):
                    overlap = model.eigenvectors.conj().T @ dc_model.eigenvectors
                    initial = overlap @ initial @ overlap.conj().T
                transfer = _population_transfer(
                    entry, int(port), params, initial, budget
                )
                accepted[selected] = transfer <= min(0.02, (1.5 * budget) ** (2 / 3))
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
