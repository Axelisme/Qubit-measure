"""Compiled Fourier, omission and retained-graph numerics for joint RWA.

Callers establish valid, aligned numeric inputs before entering these functions.
Inputs may be read-only and are never modified. Returned arrays are newly
allocated and caller-owned. NumPy/Numba failures propagate without validation
or recovery here. Carrier order is preserved; each sign axis is (+1,-1).
"""

from typing import overload

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._dop853_data import Models

# Calibrated on 128--512 ns Gaussian simultaneous gates in the 16-level
# joint model: 50 MHz edges + 4 ns ceiling gave <= 1.76e-3 population change
# against lab DOP853 through depth 16. The calibrated half-flux pair has
# summed full-operator/carrier ratio 0.217, sampled normalized slope 0.0192/ns
# and curvature 0.00231/ns² (including DAC quantization).
# Covariant admission keeps margins up to 0.25, 0.025/ns and 0.003/ns².
# Ordinary multitone RWA instead resolves omitted channels below.
EDGE_GHZ = 0.05
# Half the strict-lab failing combination-channel boundary (0.0116496).
# The extended 96-point single-constant scan had no failures below 0.045813;
# the same-tone counterexample is 0.015629. Keep the stricter shared cap.
MAX_OMISSION_TRANSFER = 5.8e-3


@overload
def fourier_coefficient(
    x: np.complex128, y: np.complex128, sign: int | np.int64
) -> np.complex128: ...


@overload
def fourier_coefficient(
    x: NDArray[np.complex128], y: NDArray[np.complex128], sign: int | np.int64
) -> NDArray[np.complex128]: ...


@njit(nogil=True, cache=True)
def fourier_coefficient(
    x: np.complex128 | NDArray[np.complex128],
    y: np.complex128 | NDArray[np.complex128],
    sign: int | np.int64,
) -> np.complex128 | NDArray[np.complex128]:
    """Single owner of full Fourier coefficients before the retained window.

    x/y are complex128 quadratures in GHz/gain: either scalars or aligned
    (M,M) matrices. sign is +1 or -1, including np.int64 carriers.
    Return the scalar coefficient or a new caller-owned complex128 matrix
    in GHz/gain with the same shape.
    """
    return (x - sign * 1j * y) / 2


@njit(nogil=True, cache=True)
def _transfer_probability(coupling, detuning, duration):
    strength = (2 * abs(coupling)) ** 2
    if strength == 0:
        return 0.0
    return min(strength / (strength + detuning**2), strength * duration**2)


@njit(nogil=True, cache=True)
def _two_step_coupling(data, first, second, i, k):
    # Sum intermediates for ONE ordered pair of signed Fourier carriers.
    energies, operators, frequencies, retained = data
    a, sa = first
    b, sb = second
    sign_a, sign_b = 1 - 2 * sa, 1 - 2 * sb
    coupling = 0j
    for m in range(len(energies)):
        d_first = energies[i] - energies[m] + sign_a * frequencies[a]
        d_second = energies[m] - energies[k] + sign_b * frequencies[b]
        if not retained[a, sa, i, m]:
            denominator = d_first
        elif not retained[b, sb, m, k]:
            denominator = -d_second
        else:
            continue
        coupling += operators[a, sa, i, m] * operators[b, sb, m, k] / denominator
    return coupling


@njit(nogil=True, cache=True)
def _phase_estimate(coupling, detuning, duration, order):
    """Order 0: oscillating diagonal, 2: virtual shift, 4: corrected remainder."""
    strength = abs(coupling)
    if strength == 0:
        return 0.0
    if detuning == 0:
        return 2 * np.pi * duration * strength
    if order == 0:
        return min(2 * np.pi * duration * strength, 2 * strength / abs(detuning))
    shift = strength**2 / abs(detuning)
    if order == 4:
        # The explicit virtual diagonal already contains the second-order term.
        shift *= (strength / abs(detuning)) ** 2
    return 2 * np.pi * duration * min(strength, shift)


@njit(nogil=True, cache=True)
def _pair_probability(data, first, second, duration, additions, phase_limit):
    energies, _, frequencies, _ = data
    a, sa = first
    b, sb = second
    combination = (1 - 2 * sa) * frequencies[a] + (1 - 2 * sb) * frequencies[b]
    maximum = 0.0
    phase_maximum = 0.0
    for i in range(len(energies)):
        for k in range(len(energies)):
            if i == k and a == b and sa != sb:
                continue  # The static single-tone diagonal already retains it.
            coupling = _two_step_coupling(data, first, second, i, k)
            if first != second:
                coupling += _two_step_coupling(data, second, first, i, k)
            detuning = energies[i] - energies[k] + combination
            probability = _transfer_probability(coupling, detuning, duration)
            maximum = max(maximum, probability)
            phase_exceeds = False
            if phase_limit is not None:
                phase = _phase_estimate(
                    coupling, detuning, duration, 0 if i == k else 2
                )
                phase_maximum = max(phase_maximum, phase)
                phase_exceeds = phase > phase_limit
            if probability > MAX_OMISSION_TRANSFER or phase_exceeds:
                _restore_paths(data, first, second, i, k, additions)
                if first != second:
                    _restore_paths(data, second, first, i, k, additions)
    return maximum, phase_maximum


@njit(nogil=True, cache=True)
def omission_probability(
    energies: NDArray[np.float64],
    x: NDArray[np.complex128],
    y: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    amplitudes: NDArray[np.float64],
    duration: float,
) -> float:
    """Return the maximum omitted transfer proxy from the initial window.

    energies is (M,) GHz; x/y are (tones,M,M) GHz/gain quadratures.
    frequencies is (tones,) signed GHz in the same carrier order as x/y
    and the (tones,) nonnegative gain amplitudes. duration is in ns.
    This calibrated first/second-order proxy is not a rigorous error bound.
    """
    retained = window_mask(energies, frequencies)
    probability, _, _ = omission_analysis(
        energies, (x, y), frequencies, (amplitudes, duration), retained, None
    )
    return probability


@njit(nogil=True, cache=True)
def window_mask(
    energies: NDArray[np.float64], frequencies: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Return a new caller-owned (tones,2,M,M) bool initial-window mask.

    energies is (M,) GHz and frequencies is (tones,) signed GHz in caller
    order. Axis 1 orders signs (+1,-1) for each supplied carrier.
    """
    size = len(energies)
    retained = np.empty((len(frequencies), 2, size, size), dtype=np.bool_)
    for a in range(len(frequencies)):
        for sa in range(2):
            for i in range(size):
                for j in range(size):
                    retained[a, sa, i, j] = (
                        abs(energies[i] - energies[j] + (1 - 2 * sa) * frequencies[a])
                        < EDGE_GHZ
                    )
    return retained


@njit(nogil=True, cache=True)
def _restore_paths(data, first, second, i, k, additions):
    # Restore one strongest missing path per channel and signed time order.
    # Recompute the complete omission estimate next round; no amplitude cutoff
    # discards the weaker parents, and fast virtual edges need not all return.
    energies, operators, frequencies, retained = data
    a, sa = first
    b, sb = second
    largest = 0.0
    best = -1
    for m in range(len(energies)):
        if retained[a, sa, i, m] and retained[b, sb, m, k]:
            continue
        product = operators[a, sa, i, m] * operators[b, sb, m, k]
        if product == 0:
            continue
        first_detuning = energies[i] - energies[m] + (1 - 2 * sa) * frequencies[a]
        second_detuning = energies[m] - energies[k] + (1 - 2 * sb) * frequencies[b]
        denominator = first_detuning if not retained[a, sa, i, m] else -second_detuning
        strength = abs(product / denominator)
        if strength > largest:
            best, largest = m, strength
    if best == -1:
        return
    if not retained[a, sa, i, best]:
        additions[a, sa, i, best] = True
        additions[a, 1 - sa, best, i] = True
    if not retained[b, sb, best, k]:
        additions[b, sb, best, k] = True
        additions[b, 1 - sb, k, best] = True


@njit(nogil=True, cache=True)
def omission_analysis(
    energies: NDArray[np.float64],
    quadratures: tuple[NDArray[np.complex128], NDArray[np.complex128]],
    frequencies: NDArray[np.float64],
    pulse: tuple[NDArray[np.float64], float],
    retained: NDArray[np.bool_],
    phase_limit: float | None,
) -> tuple[float, float, NDArray[np.bool_]]:
    """One rule for every removed first/second-order Fourier coupling.

    energies is (M,) GHz; quadratures is (x,y), each (tones,M,M) GHz/gain.
    frequencies is (tones,) signed GHz in matching carrier order.
    pulse is ((tones,) nonnegative gain amplitudes, duration_ns).
    retained is (tones,2,M,M) bool with signs ordered (+1,-1).
    phase_limit is in rad, or None to skip phase compilation.
    Return (maximum transfer proxy, maximum phase proxy in rad, additions).
    additions is a new caller-owned bool mask with retained's shape.

    C_s=(X-s*iY)/2, d=Ei-Ej+s*f. Removed one-photon edges use
    g=C_s*amplitude. Two-step paths use g=sum(C_a*C_b/d_removed)
    over intermediates and distinct time orders; paths with both edges retained
    are excluded. A same-tone/same-sign path has one order, NOT a duplicated
    carrier. Mixed signs and distinct carriers include both orders. Full dressed
    gaps include conditional ZZ. Only zero-frequency same-tone diagonal
    terms are excluded: they already live in RWACoefficients.diagonal.
    Oscillating diagonal terms remain charged like every other omission.
    Each effective edge uses P=min(4|g|²/(4|g|²+delta²),4|g|²*T²).
    GHz/ns units and the finite-time cap follow the calibrated combination
    proxy, not a rigorous Dyson error bound (no claim of a 2*pi certificate).
    The maximum P is shared across window-excluded one-photon, same-tone and
    cross-tone two-photon terms, including empty retained graphs.
    Multitone rows weight Fourier operators by abs(gain)*peak(abs(envelope)),
    a shared envelope bound covering the full row including offsets and holds.
    A None phase_limit skips phase compilation for transfer-only callers.
    Otherwise off-diagonal one-photon terms use
    2*pi*T*min(|g|,|g|^4/|d|^3), since their second-order diagonal is explicit.
    Two-photon off-diagonal terms use 2*pi*T*min(|g_eff|,|g_eff|²/|d_eff|).
    Oscillating diagonal terms instead use min(2*pi*T*|g|,2*|g/d|).
    Zero detuning uses the finite-time term. Both maxima trigger restoration;
    legacy gate/single-constant callers retain their original transfer policy.

    Calibration: 96 once-only strict-lab points on the 16-level joint model,
    four flux fractions 0,0.4,0.8,1 between integer and near-half, theta 0/0.7,
    full-operator/carrier ratios 0.01/0.07/0.2, durations 128/1024 ns and carrier
    offsets 0/51 MHz. Minimum failing P (population error>0.1) was 0.045813;
    all 14 admitted points had error<=0.002947. The same-tone counterexample
    fails at P=0.015629. The existing 25-point cross-tone scan fails first at
    channel P=0.0116496. Its halved boundary, rounded down to 0.0058, is more
    conservative than either new boundary. A recheck of the dual-constant
    96-point calibration keeps 24/36 previously admitted points; twelve safe
    intermediate-flux points now fall back due to the new omissions.
    """
    x, y = quadratures
    amplitudes, duration = pulse
    maximum = 0.0
    phase_maximum = 0.0
    size, count = len(energies), len(frequencies)
    additions = np.zeros_like(retained)
    operators = np.empty((count, 2, size, size), dtype=np.complex128)
    for a in range(count):
        for sa in range(2):
            sign_a = 1 if sa == 0 else -1
            operators[a, sa] = fourier_coefficient(x[a], y[a], sign_a) * amplitudes[a]
            for i in range(size):
                for j in range(size):
                    d = energies[i] - energies[j] + sign_a * frequencies[a]
                    if not retained[a, sa, i, j]:
                        probability = _transfer_probability(
                            operators[a, sa, i, j], d, duration
                        )
                        maximum = max(maximum, probability)
                        phase = (
                            _phase_estimate(
                                operators[a, sa, i, j], d, duration, 0 if i == j else 4
                            )
                            if phase_limit is not None
                            else 0.0
                        )
                        phase_maximum = max(phase_maximum, phase)
                        if probability > MAX_OMISSION_TRANSFER or (
                            phase_limit is not None and phase > phase_limit
                        ):
                            additions[a, sa, i, j] = True
                            additions[a, 1 - sa, j, i] = True
    for a in range(count):
        for b in range(a, count):
            for sa in range(2):
                for sb in range(2):
                    if a == b and sa > sb:
                        continue
                    data = (energies, operators, frequencies, retained)
                    first, second = (a, sa), (b, sb)
                    probability, phase = _pair_probability(
                        data, first, second, duration, additions, phase_limit
                    )
                    maximum = max(maximum, probability)
                    phase_maximum = max(phase_maximum, phase)
    return maximum, phase_maximum, additions


@njit(nogil=True, cache=True)
def coefficients(
    models: Models,
    ports: NDArray[np.int64],
    frequencies: NDArray[np.float64],
    tones: NDArray[np.int64],
    retained: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    """Build Fourier edges and compact diagonal using the same resolved mask.

    models packs k equal-size M-state models. ports/frequencies are (k,P)
    int64 drive indices / signed GHz, ordered by carrier; tones is (k,)
    int64 active counts. Optional retained is (tones,2,M,M) bool, shared
    across keys with signs (+1,-1); None selects the initial window.
    Return new caller-owned (positive, negative, diagonal) arrays.
    positive/negative are complex128 (k,P,M,M) GHz/gain, mutual adjoints;
    diagonal is float64 (k,max(tones),M) GHz/gain² for virtual shifts.
    """
    size = models.energies.shape[1]
    positive = np.zeros((len(tones), ports.shape[1], size, size), np.complex128)
    negative = np.zeros_like(positive)
    max_tones = 0
    for count in tones:
        max_tones = max(max_tones, count)
    diagonal = np.zeros((len(tones), max_tones, size), np.float64)
    for key in range(len(tones)):
        for tone in range(tones[key]):
            port, frequency = ports[key, tone], frequencies[key, tone]
            for i in range(size):
                for j in range(size):
                    gap = models.energies[key, i] - models.energies[key, j]
                    x, y = models.x[key, port, i, j], models.y[key, port, i, j]
                    for sign in (1, -1):
                        coefficient = fourier_coefficient(x, y, sign)
                        detuning = gap + sign * frequency
                        keep = abs(detuning) < EDGE_GHZ
                        if retained is not None:
                            keep = retained[tone, 0 if sign == 1 else 1, i, j]
                        if keep:
                            if sign == 1:
                                positive[key, tone, i, j] = coefficient
                            else:
                                negative[key, tone, i, j] = coefficient
                        else:
                            # Exact resonances evolve explicitly above; no
                            # denominator regularization or double counting.
                            diagonal[key, tone, i] += abs(coefficient) ** 2 / detuning
    return positive, negative, diagonal


@njit(nogil=True, cache=True)
def retained_dimension(
    retained: NDArray[np.bool_],
    quadratures: tuple[NDArray[np.complex128], NDArray[np.complex128]],
) -> int:
    """Return sum(component_size**2), including isolated populations.

    quadratures is (x,y), each complex128 (tones,M,M) in GHz/gain.
    retained is (tones,2,M,M) bool in matching carrier order, with signs
    (+1,-1). The union graph uses the retained nonzero positive edges.
    """
    # Same measure as the single-tone real generator: sum(component_size**2),
    # including isolated populations, without imposing covariant phase charges.
    x, y = quadratures
    size = x.shape[1]
    adjacency = np.zeros((size, size), dtype=np.bool_)
    for a in range(len(x)):
        coefficient = fourier_coefficient(x[a], y[a], np.int64(1))
        for i in range(size):
            for j in range(size):
                if retained[a, 0, i, j] and coefficient[i, j] != 0:
                    adjacency[i, j] = adjacency[j, i] = True
    visited = np.zeros(size, dtype=np.bool_)
    queue = np.empty(size, dtype=np.int64)
    dimension = 0
    for root in range(size):
        if visited[root]:
            continue
        visited[root] = True
        queue[0] = root
        first, last = 0, 1
        while first < last:
            i = queue[first]
            first += 1
            for j in range(size):
                if adjacency[i, j] and not visited[j]:
                    visited[j] = True
                    queue[last] = j
                    last += 1
        dimension += last * last
    return dimension
