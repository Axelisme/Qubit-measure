"""Full-carrier second-order modal Dyson evolution, with model-only preparation."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ._dyson_bound import accepts_population_bound
from ._dyson_model import (
    commutator,
    constant_mask,
    drive_ratios,
    freeze,
    generator,
    model_groups,
    model_keys,
    parameters,
)
from ._dyson_paths import build_shared_paths, contractible_modes


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
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Contract ordinary per-batch paths with coefficient and bound cutoffs."""
    first_cutoff, second_cutoff = 1e-16, 1e-12
    dimension = len(modes)
    # Each discarded path gets an equal share of budget/8. The actual modal
    # input and max gain/duration bound every row; thresholds remain unchanged.
    allowance = pruning[-1] / (8 * (2 * dimension**2 + 4 * dimension**3))
    gain = np.max(np.abs(params[:, 1]))
    duration = np.max(params[:, 2])
    growth = np.exp(max(0.0, np.max(eigenvalues.real)) * duration)
    indices = []
    coefficients = []
    for a in range(dimension):
        for outer in range(2):
            r = 2 * outer - 1
            for b in range(dimension):
                coefficient = couplings[outer, a, b]
                if coefficient == 0:
                    continue
                first = coefficient * modes[b]
                bound = abs(first * gain) * duration * growth * pruning[a]
                if abs(first) > first_cutoff or bound > allowance:
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
                        if abs(second) > second_cutoff or bound > allowance:
                            indices.append((a, b, c, r, s))
                            coefficients.append(second)
    # Numba cannot reshape an empty list of tuples to a known width.
    path_indices = np.empty((len(indices), 5), dtype=np.int64)
    for path in range(len(indices)):
        path_indices[path] = indices[path]
    return path_indices, np.array(coefficients, dtype=np.complex128)


@dataclass(frozen=True)
class DriveExtent:
    """Conservative shared-row drive bounds; invalid values raise ValueError.

    max_abs_gain: finite nonnegative maximum absolute signed gain (unitless).
    max_duration_ns: finite nonnegative maximum duration, in ns. The two maxima
        may come from different rows.
    """

    max_abs_gain: float
    max_duration_ns: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.max_abs_gain)
            or self.max_abs_gain < 0
            or not np.isfinite(self.max_duration_ns)
            or self.max_duration_ns < 0
        ):
            raise ValueError("Dyson drive extent requires finite nonnegative bounds")


class PrunedDysonPaths(NamedTuple):
    """Budgeted contracted paths in their original accumulation order."""

    indices: NDArray[np.int64]
    """Retained (paths,5) records (a,b,c,r,s); c=-1 denotes first order."""

    coefficients: NDArray[np.complex128]
    """Retained (paths,) coefficients, already contracted with input modes."""

    discarded_bound: float
    """Sum of omitted path bounds on each reconstructed density entry."""


@njit(inline="always")
def _path_key(a: int, b: int, c: int, r: int, s: int, dimension: int) -> int:
    """Encode a valid path, including first-order sentinels, in mixed radix."""
    term = 0 if c == -1 else 1 + ((s + 1) // 2) * dimension + c
    return ((a * 2 + (r + 1) // 2) * dimension + b) * (2 * dimension + 1) + term


@njit(nogil=True, cache=True)
def _paired_path_groups(
    indices: NDArray[np.int64], conjugates: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Return disjoint conjugate pairs; unmatched paths must remain retained.

    Shared tables can contain hundreds of thousands of records even for seven
    rows. Use integer keys and a half-full, linearly probed array lookup instead
    of constructing Python tuples and dict entries. Keep input order so stable
    bound sorting and pruning stay unchanged.
    """
    dimension = len(conjugates)
    keys = np.empty(len(indices), dtype=np.int64)
    capacity = 2 * len(indices) + 1
    slots = np.full(capacity, -1, dtype=np.int64)
    for path in range(len(indices)):
        a, b, c, r, s = indices[path]
        key = _path_key(a, b, c, r, s, dimension)
        keys[path] = key
        slot = key % capacity
        while slots[slot] != -1:
            if keys[slots[slot]] == key:
                raise ValueError("Dyson path records must be unique")
            slot = (slot + 1) % capacity
        slots[slot] = path
    groups = np.empty((len(indices), 2), dtype=np.int64)
    count = 0
    for path in range(len(indices)):
        a, b, c, r, s = indices[path]
        key = _path_key(
            int(conjugates[a]),
            int(conjugates[b]),
            -1 if c == -1 else int(conjugates[c]),
            -r,
            -s,
            dimension,
        )
        slot = key % capacity
        while slots[slot] != -1 and keys[slots[slot]] != key:
            slot = (slot + 1) % capacity
        partner = slots[slot]
        if partner >= path:
            groups[count, 0] = path
            groups[count, 1] = partner
            count += 1
    return groups[:count]


@njit(nogil=True, cache=True)
def _pruned_path_mask(
    groups: NDArray[np.int64],
    group_bounds: NDArray[np.float64],
    order: NDArray[np.int64],
    path_count: int,
    allowance: float,
) -> tuple[NDArray[np.bool_], float]:
    """Spend the allowance in stable bound order with scalar sequential sums."""
    keep = np.ones(path_count, dtype=np.bool_)
    discarded = 0.0
    for group in order:
        total = discarded + float(group_bounds[group])
        if total > allowance:
            break
        discarded = total
        keep[groups[group, 0]] = False
        keep[groups[group, 1]] = False
    return keep, discarded


def _validate_path_records(indices: NDArray[np.int64], dimension: int) -> None:
    """Validate (a,b,c,r,s) fields without changing the path records.

    The caller guarantees an integer array of shape (N,5) and a positive modal
    dimension. a and b must index modes; c is -1 for first order or a mode index
    for second order. r is +/-1; s is 0 for first order and +/-1 for second order.
    Return None when all records are valid, otherwise raise ValueError.
    """
    a, b, c, r, s = indices.T
    if (
        np.any(a < 0)
        or np.any(a >= dimension)
        or np.any(b < 0)
        or np.any(b >= dimension)
        or np.any(c < -1)
        or np.any(c >= dimension)
        or np.any(abs(r) != 1)
        or np.any(np.where(c == -1, s != 0, abs(s) != 1))
    ):
        raise ValueError("Dyson pruning received an invalid path record")


def prune_dyson_paths(
    indices: NDArray[np.int64],
    coefficients: NDArray[np.complex128],
    eigenvalues: NDArray[np.complex128],
    conjugates: NDArray[np.int64],
    pruning: NDArray[np.float64],
    extent: DriveExtent,
) -> PrunedDysonPaths:
    """Greedily omit complete conjugate groups within an entrywise budget/8.

    indices (N,5) contains unique (a,b,c,r,s) modal path records. a,b,c index
    eigenvalues (D,); c=-1 marks first order with s=0. Carrier signs r and
    second-order s are -1 or +1. coefficients (N,) include the actual input
    modes. Carrier shifts must be purely imaginary. conjugates (D,) is an
    involution mapping each mode to its density-adjoint partner.

    pruning (D+1,) contains max(abs(W[:,a])) then the positive segment budget.
    extent holds the maximum absolute gain and duration over all shared rows.
    The maxima may come from different rows. Positive real eigenvalues are
    allowed: every bound includes
    exp(max(0,max(Re(eigenvalues)))*duration).

    First-order bounds are |coefficient|*G*T*growth*pruning[a], with G and T
    from extent. Second-order bounds replace G*T with G**2*T**2/2.
    Groups are ordered by their summed bound (stable on ties), and omitted
    until the next group would exceed budget/8. Missing partners stay retained;
    a self-conjugate record is counted once. Retained records keep input order.
    No per-row pruning may follow this operation. Inputs are not changed.

    Raises ValueError for invalid shapes, nonfinite inputs or bounds, negative
    duration/weights, nonpositive budget, invalid records or conjugate mapping.
    """
    dimension = len(eigenvalues)
    if (
        eigenvalues.ndim != 1
        or dimension == 0
        or indices.ndim != 2
        or indices.shape[1] != 5
        or coefficients.shape != (len(indices),)
        or conjugates.shape != (dimension,)
        or pruning.shape != (dimension + 1,)
    ):
        raise ValueError("Dyson pruning requires aligned path and modal shapes")
    if (
        not all(np.isfinite(v).all() for v in (coefficients, eigenvalues, pruning))
        or np.any(pruning[:-1] < 0)
        or pruning[-1] <= 0
    ):
        raise ValueError("Dyson pruning requires finite bounds and positive budget")
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or not np.issubdtype(conjugates.dtype, np.integer)
        or np.any(conjugates < 0)
        or np.any(conjugates >= dimension)
    ):
        raise ValueError("Dyson pruning requires integer paths and conjugate modes")
    if not np.array_equal(conjugates[conjugates], np.arange(dimension)):
        raise ValueError("Dyson conjugate modes must form an involution")
    _validate_path_records(indices, dimension)
    a, c = indices[:, 0], indices[:, 2]
    strength = np.float64(extent.max_abs_gain) * extent.max_duration_ns
    with np.errstate(over="ignore", invalid="ignore"):
        growth = np.exp(
            max(0.0, float(np.max(eigenvalues.real))) * extent.max_duration_ns
        )
        bounds = (
            abs(coefficients)
            * np.where(c == -1, strength, strength**2 / 2)
            * growth
            * pruning[a]
        )
    if not np.isfinite(bounds).all():
        raise ValueError("Dyson pruning bounds must be finite")
    groups = _paired_path_groups(indices, conjugates)
    group_bounds = bounds[groups[:, 0]] + np.where(
        groups[:, 0] == groups[:, 1], 0.0, bounds[groups[:, 1]]
    )
    keep, discarded = _pruned_path_mask(
        groups,
        group_bounds,
        np.argsort(group_bounds, kind="stable"),
        len(indices),
        pruning[-1] / 8,
    )
    return PrunedDysonPaths(indices[keep], coefficients[keep], discarded)


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
    factor_pairs = np.empty((len(pairs), 3), dtype=np.int64)
    for pair in range(len(pairs)):
        factor_pairs[pair] = pairs[pair]
    return factor_pairs, path_pairs


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


class _Paths(NamedTuple):
    """Contracted terms, shared first factors and per-row pruning policy."""

    indices: NDArray[np.int64]
    coefficients: NDArray[np.complex128]
    pairs: NDArray[np.int64]
    path_pairs: NDArray[np.int64]
    shared: bool


@njit(nogil=True, cache=True)
def _evaluate_paths(
    params: NDArray[np.float64],
    eigenvalues: NDArray[np.complex128],
    modes: NDArray[np.complex128],
    pruning: NDArray[np.float64],
    paths: _Paths,
) -> NDArray[np.complex128]:
    """Evaluate paths; shared tables already spent their pruning budget."""
    indices, coefficients, pairs, path_pairs, shared = paths
    dimension = len(modes)
    allowance = pruning[-1] / (8 * (2 * dimension**2 + 4 * dimension**3))
    result = np.empty((len(params), dimension), dtype=np.complex128)
    # Shared tables keep every retained pair on every row: another cutoff would
    # spend beyond budget/8 and could break Hermiticity. Ordinary batches retain
    # their original per-row policy, including the uniform-batch shortcut.
    uniform_bound = shared or (
        len(params) > 0
        and np.all(np.abs(params[:, 1]) == abs(params[0, 1]))
        and np.all(params[:, 2] == params[0, 2])
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
) -> NDArray[np.complex128]:
    """Share contracted paths for equal inputs; handle distinct states rowwise."""
    if len(params) == 0:
        return initial_modes.copy()
    if np.all(params[:, 1] == 0) or np.all(params[:, 2] == 0):
        return np.exp(params[:, 2:3] * eigenvalues) * initial_modes
    if np.all(initial_modes == initial_modes[0]):
        indices, coefficients = _build_paths(
            params, eigenvalues, initial_modes[0], couplings, pruning
        )
        pairs, path_pairs = _group_first_factors(indices)
        return _evaluate_paths(
            params,
            eigenvalues,
            initial_modes[0],
            pruning,
            _Paths(indices, coefficients, pairs, path_pairs, shared=False),
        )
    result = np.empty_like(initial_modes)
    for job in range(len(params)):
        indices, coefficients = _build_paths(
            params[job : job + 1],
            eigenvalues,
            initial_modes[job],
            couplings,
            pruning,
        )
        pairs, path_pairs = _group_first_factors(indices)
        result[job] = _evaluate_paths(
            params[job : job + 1],
            eigenvalues,
            initial_modes[job],
            pruning,
            _Paths(indices, coefficients, pairs, path_pairs, shared=False),
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
    )


@dataclass(frozen=True)
class _Entry:
    values: NDArray[np.complex128]
    vectors: NDArray[np.complex128]
    inverse: NDArray[np.complex128]
    couplings: tuple[NDArray[np.complex128], ...]
    conjugates: NDArray[np.int64]

    @property
    def nbytes(self) -> int:
        return sum(
            v.nbytes
            for v in (
                self.values,
                self.vectors,
                self.inverse,
                *self.couplings,
                self.conjugates,
            )
        )


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Entry, ...]

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)


@dataclass(frozen=True)
class _BoundInput:
    """One model/port and source input, owned only by the current evolve."""

    entries: tuple[_Entry, ...]
    port: int
    modes: NDArray[np.complex128]
    pruning: NDArray[np.float64]
    indices: NDArray[np.int64]
    coefficients: NDArray[np.complex128]
    pairs: NDArray[np.int64]
    path_pairs: NDArray[np.int64]

    @property
    def nbytes(self) -> int:
        return self.entries[0].nbytes + sum(
            array.nbytes
            for array in (
                self.modes,
                self.pruning,
                self.indices,
                self.coefficients,
                self.pairs,
                self.path_pairs,
            )
        )


@njit(nogil=True, cache=True)
def _transform_modes(
    modes: NDArray[np.complex128],
    transform: NDArray[np.complex128],
    size: int,
) -> NDArray[np.complex128]:
    """Transform populations rowwise; secular coherence coordinates are unchanged."""
    stride = size + 1
    population_block = np.ascontiguousarray(transform[::stride, ::stride])
    result = modes.copy()
    for row in range(len(modes)):
        populations = np.ascontiguousarray(modes[row, ::stride])
        # Fixed-size products keep rounding independent of worker batch size.
        result[row, ::stride] = population_block @ populations
    return result


def _advance_bound(
    bound: _BoundInput,
    params: NDArray[np.float64],
    shape: tuple[int, ...],
) -> NDArray[np.complex128]:
    entry = bound.entries[0]
    modes = _evaluate_paths(
        params,
        entry.values,
        bound.modes,
        bound.pruning,
        _Paths(
            bound.indices,
            bound.coefficients,
            bound.pairs,
            bound.path_pairs,
            shared=True,
        ),
    )
    return _transform_modes(modes, entry.vectors, shape[-1]).reshape(shape)


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
        # Normalized eigenvector columns imply sigma_min <= 1 <= sigma_max,
        # so the coherence identity blocks cannot change this condition number.
        if np.linalg.cond(rotation) > 1e10:
            raise RuntimeError("Dyson population modes are ill-conditioned")
        inverse = np.eye(size**2, dtype=np.complex128)
        inverse[np.ix_(populations, populations)] = np.linalg.inv(
            rotation.astype(np.complex128)
        )
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
    # Adjoint each column of W, then express it in modal coordinates. Unlike
    # nearest-eigenvalue matching, this handles repeated coherence roots and
    # complex-conjugate population eigenmodes of a real rate generator.
    adjoints = vectors.reshape(size, size, -1).conj().swapaxes(0, 1)
    conjugates = np.argmax(abs(inverse @ adjoints.reshape(size**2, -1)), axis=0)
    conjugates.setflags(write=False)
    return _Entry(
        freeze(values), freeze(vectors), freeze(inverse), tuple(couplings), conjugates
    )


def _advance(
    entry: _Entry,
    port: int,
    params: NDArray[np.float64],
    states: NDArray[np.complex128],
    budget: float,
) -> NDArray[np.complex128]:
    modes = _transform_modes(
        states.reshape(len(states), -1), entry.inverse, states.shape[-1]
    )
    pruning = np.r_[np.max(abs(entry.vectors), axis=0), budget]
    evolved = evaluate_dyson(
        params, entry.values, modes, entry.couplings[port], pruning
    )
    return _transform_modes(evolved, entry.vectors, states.shape[-1]).reshape(
        states.shape
    )


def _free_populations(
    entry: _Entry,
    durations: NDArray[np.float64],
    states: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Propagate only the secular population block, without coherence modes."""
    size = states.shape[-1]
    populations = np.arange(size) * (size + 1)
    block = np.ix_(populations, populations)
    modes = np.diagonal(states, axis1=1, axis2=2) @ entry.inverse[block].T
    free = np.exp(durations[:, None] * entry.values[populations]) * modes
    return free @ entry.vectors[block].T


class DysonKernel:
    """C5 weak one-tone constant kernel; engine owns known-state eligibility.

    Guards use r<=min(0.1, 0.25*budget**(1/3)), r2<=0.05 and
    B2+budget/8<=min(0.02, (1.5*budget)**(2/3)). B2 bounds unpruned second-order
    driven-minus-free populations; budget/8 reserves omitted-path error.
    Rows starting at zero use the full source density in the row's model basis:
    populations, actual pair coherence magnitudes and off-diagonal trace norm.
    Coherence phases are bounded, not evaluated. Later rows use an arbitrary-input
    density bound because preceding idle/flux segments may change the state.
    Certified shared no-drive prefixes instead use only r and r2 for admission;
    the engine binds their actual evolved input. Population changes only report.
    The r limit follows cubic off-resonant coherence-error scaling; the population
    limit follows the two-level estimate (2/3)*abs(delta_p)**(3/2). Coefficients
    are calibrated by synthetic scans, not a rigorous multilevel remainder
    proof. Cross-family fallback order belongs to the dispatch decision graph.
    Prepared modes are shared across frequency, gain, duration, phase and port.
    Preparation never reads a density. apply accepts arbitrary densities, not
    only diagonal or stationary states. bind_input greedily prunes the shared
    input-contracted table by summed entrywise bounds, dropping conjugate pairs
    together up to budget/8, consistent with B2's reserved margin. Retained paths
    accumulate in original order without further row cutoffs. Ordinary apply
    keeps its existing per-batch pruning policy.
    """

    name = "dyson"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Classify rows using r, r2 and B2+budget/8, without path evaluation.

        At start_ns==0, B2 uses the source density in the row model basis,
        including exact coherence magnitudes and off-diagonal trace norm.
        Certified no-drive prefixes require only r and r2. Driven prefixes
        use their engine-computed known input. Other later rows use
        an arbitrary-input density bound. The budget/8 margin
        covers pruning. Threshold coefficients are synthetic-scan calibrations,
        not rigorous multilevel error guarantees. Returns a row-aligned mask;
        invalid budgets raise ValueError.
        """
        source, budget = context.source, context.budget
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Dyson requires positive budget")
        accepted = constant_mask(table, rows)
        active = np.flatnonzero(accepted)
        for positions, model in model_groups(table, rows[active], source):
            original = active[positions]
            for port in np.unique(table.tone_port[rows[original], 0]):
                selected = original[table.tone_port[rows[original], 0] == port]
                params = parameters(table, rows[selected])
                one, two = drive_ratios(model, int(port), params)
                eligible = (one <= min(0.1, 0.25 * budget ** (1 / 3))) & (two <= 0.05)
                accepted[selected] = False
                selected, params = selected[eligible], params[eligible]
                if not len(selected):
                    continue
                accepted[selected] = accepts_population_bound(
                    table,
                    rows[selected],
                    source,
                    model,
                    int(port),
                    budget,
                )
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Model-level lookup keys, namespaced by the engine's DC model identity."""
        del context
        return model_keys(table, rows)

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Closed-form work does not scale with pulse length or carrier frequency."""
        return np.ones_like(table.duration_ns[rows])

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> _Prepared:
        """Factor one secular free generator per representative model row."""
        source, budget = context.source, context.budget
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

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline cached-domain admission; use the ordinary guard build path."""
        del prepared, table, rows, context

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline guard preparation handoff; the engine calls guard normally."""
        source, budget = context.source, context.budget
        del table, rows, source, budget

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> _BoundInput | None:
        """Contract one key/port's common source or prefix input in its row basis.

        C5 Kernel.bind_input defines the read-only, evolve-local contract.
        Empty rows return None. Max absolute gain and max duration over all
        supplied rows bound every candidate, independent of chunks. Greedy
        pruning drops complete conjugate pairs up to budget/8, matching B2's
        reserved margin. Workers retain all remaining paths in original order;
        they do not spend any further per-row pruning allowance.
        """
        budget = context.budget
        if not isinstance(prepared, _Prepared):
            raise ValueError("Dyson requires model-only prepared data for binding")
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Dyson requires positive budget")
        if not len(rows):
            return None
        entry = prepared.entries[key_index]
        port = int(table.tone_port[rows[0], 0])
        modes = source_initial.ravel() @ entry.inverse.T
        pruning = np.r_[np.max(abs(entry.vectors), axis=0), budget]
        params = parameters(table, rows)
        conjugates = entry.conjugates
        path_modes, spent = modes, 0.0
        if np.any(source_initial != np.diag(np.diag(source_initial))):
            # Reserve half the omission budget for modes, half for path pruning.
            strength = float(np.max(abs(params[:, 1])) * np.max(params[:, 2]))
            growth = float(
                np.exp(
                    max(0.0, float(np.max(entry.values.real))) * np.max(params[:, 2])
                )
            )
            path_modes, spent = contractible_modes(
                modes,
                entry.couplings[port],
                pruning[:-1] * growth,
                conjugates,
                strength=strength,
                allowance=budget / 16,
            )
        indices, coefficients = build_shared_paths(path_modes, entry.couplings[port])
        remaining = pruning.copy()
        remaining[-1] = budget - 8 * spent
        pruned = prune_dyson_paths(
            indices,
            coefficients,
            entry.values,
            conjugates,
            remaining,
            DriveExtent(float(np.max(abs(params[:, 1]))), float(np.max(params[:, 2]))),
        )
        indices, coefficients = pruned.indices, pruned.coefficients
        pairs, path_pairs = _group_first_factors(indices)
        for array in (pruning, indices, pairs, path_pairs):
            array.setflags(write=False)
        return _BoundInput(
            (entry,),
            port,
            freeze(modes),
            pruning,
            indices,
            freeze(coefficients),
            pairs,
            path_pairs,
        )

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Advance states in place; return a bool exceedance mask aligned with rows.

        Rows/key_index may be non-contiguous or reversed. Check the maximum
        absolute output population minus free-evolved input population against
        min(0.02, (1.5*budget)**(2/3)). Free evolution uses only the secular
        population block. Exceedance is not failure: retain the uncorrected
        output without retry. Numerical failure raises RuntimeError.
        """
        context.require_no_plan()
        budget = context.budget
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Dyson requires positive budget")
        if not isinstance(prepared, (_Prepared, _BoundInput)):
            raise ValueError("Dyson requires its own prepared payload")
        exceeded = np.zeros(len(rows), dtype=np.bool_)
        threshold = min(0.02, (1.5 * budget) ** (2 / 3))
        for key in np.unique(key_index):
            positions = np.flatnonzero(key_index == key)
            for port in np.unique(table.tone_port[rows[positions], 0]):
                selected = positions[table.tone_port[rows[positions], 0] == port]
                free = _free_populations(
                    prepared.entries[key],
                    table.duration_ns[rows[selected]],
                    states[selected],
                )
                params = parameters(table, rows[selected])
                if isinstance(prepared, _BoundInput):
                    if key != 0 or port != prepared.port:
                        raise ValueError(
                            "Dyson input-bound payload requires its own key/port"
                        )
                    result = _advance_bound(prepared, params, states[selected].shape)
                else:
                    result = _advance(
                        prepared.entries[key],
                        int(port),
                        params,
                        states[selected],
                        budget,
                    )
                if not np.isfinite(result).all():
                    raise RuntimeError(
                        f"Dyson nonfinite density in segment rows {rows[selected]}"
                    )
                transfer = np.diagonal(result, axis1=1, axis2=2) - free
                exceeded[selected] = np.max(abs(transfer), axis=1) > threshold
                states[selected] = result
        return exceeded
