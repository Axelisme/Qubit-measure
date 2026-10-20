"""Two-tone Floquet k=0 admission, modal Dyson and periodic channel numerics."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

# All tuple arrays are borrowed inputs; read-only arrays are accepted.
FrameData = tuple[
    NDArray[np.float64],  # energies (M,) in GHz.
    NDArray[np.complex128],  # bath (M²,M²) per ns.
    NDArray[np.complex128],  # probe x (ports,M,M) in GHz per gain.
    NDArray[np.complex128],  # probe y (ports,M,M) in GHz per gain.
]
Bounds = tuple[
    bool,  # valid: whether the frame is valid.
    NDArray[np.float64],  # omitted drive norms (ports,) in GHz per gain.
    float,  # bath omitted norm per ns.
    float,  # convergence: max zero-mode residual (bath per ns, drive GHz/gain).
    float,  # frame error, dimensionless.
]
ModalData = tuple[
    NDArray[np.complex128],  # lam (M²,) modal eigenvalues per ns.
    NDArray[np.complex128],  # modes (M²,M²), dimensionless.
    NDArray[np.complex128],  # inverse (M²,M²) of modes, dimensionless.
    NDArray[np.complex128],  # rotation Fourier coefficients (K,M,M), dimensionless.
    NDArray[np.float64],  # harmonic (K,) integer-valued float64 indices.
    NDArray[np.complex128],  # couplings (ports,2,M²,M²) per ns/gain; sign axis -1,+1.
]
PeriodicData = tuple[
    NDArray[np.complex128],  # base (M²,M²) Liouvillian per ns.
    NDArray[np.complex128],  # probe x (ports,M²,M²) Liouvillians per ns per gain.
    NDArray[np.complex128],  # probe y (ports,M²,M²) Liouvillians per ns per gain.
    NDArray[np.complex128],  # rotation Fourier coefficients (K,M,M), dimensionless.
    NDArray[np.float64],  # harmonic (K,) integer-valued float64 indices.
]


@njit(nogil=True, cache=True)
def _f(q: complex, a: complex, duration: float) -> complex:
    z = (a - q) * duration
    if abs(z) < 1e-5:
        return (
            np.exp(q * duration)
            * duration
            * (1 + z / 2 + z * z / 6 + z**3 / 24 + z**4 / 120)
        )
    return (np.exp(a * duration) - np.exp(q * duration)) / (a - q)


@njit(nogil=True, cache=True)
def _k(q: complex, b: complex, a: complex, duration: float) -> complex:
    if abs(b - a) * duration < 1e-6:
        z = ((a + b) / 2 - q) * duration
        if abs(z) < 1e-4:
            derivative = 0.5 + z / 3 + z * z / 8 + z**3 / 30 + z**4 / 144
            return np.exp(q * duration) * duration**2 * derivative
        ez = np.exp((a + b) / 2 * duration)
        return duration**2 * ((z - 1) * ez + np.exp(q * duration)) / (z * z)
    return (_f(q, b, duration) - _f(q, a, duration)) / (b - a)


@njit(nogil=True, cache=True)
def _first_factors(lam, duration, omega):
    """Tabulate each exp[lambda_i, lambda_j + i*r*omega] once per row.

    Factors include duration. Base and carrier exponentials are also returned
    for free evolution and shifted second-order factors, respectively.
    """
    scaled = lam * duration
    exponentials = np.exp(scaled)
    carriers = np.exp(1j * np.arange(-2, 3) * omega * duration)
    factors = np.empty((2, len(lam), len(lam)), dtype=np.complex128)
    for outer in range(2):
        sign = 2 * outer - 1
        for i in range(len(lam)):
            for j in range(len(lam)):
                z = scaled[j] + 1j * sign * omega * duration - scaled[i]
                if abs(z) < 1e-5:
                    factors[outer, i, j] = (
                        exponentials[i]
                        * duration
                        * (1 + z / 2 + z * z / 6 + z**3 / 24 + z**4 / 120)
                    )
                else:
                    factors[outer, i, j] = (
                        exponentials[j] * carriers[sign + 2] - exponentials[i]
                    ) / (lam[j] + 1j * sign * omega - lam[i])
    return factors, exponentials, carriers


@njit(nogil=True, cache=True)
def rotation(
    entry_coefficients: NDArray[np.complex128],
    harmonic: NDArray[np.float64],
    phase: float | np.float64,
) -> NDArray[np.complex128]:
    """Evaluate the shared Fourier frame at scalar phase in radians.

    entry_coefficients is complex128 (K,M,M), with dimensionless Fourier
    matrices. harmonic is float64 (K,) containing integer harmonic indices
    aligned with those matrices. phase is float or np.float64 in radians.
    The caller supplies validated arrays from the same frame.

    Inputs remain caller-owned and unmodified, with no retained references.
    Returns a new caller-owned writable complex128 (M,M) Fourier sum.
    This evaluation does not guarantee an exactly unitary matrix.
    NumPy/Numba failures propagate without fallback, retry or coercion.
    """
    result = np.zeros(entry_coefficients.shape[1:], dtype=np.complex128)
    for k in range(len(harmonic)):
        result += np.exp(1j * harmonic[k] * phase) * entry_coefficients[k]
    return result


@njit(nogil=True, cache=True)
def _second_contraction(data, duration, omega, outer, inner):
    """Contract the middle mode with BLAS, retaining stable close-pole formulas.

    For separated endpoints the second factor is
    (F_outer[i,j] - carrier_outer * F_inner[j,k]) / gap[i,k].
    The denominator is independent of j, so two GEMMs give its numerator.
    Close endpoints and repeated inner poles still use the scalar divided
    differences; no modes or coupling paths are pruned.
    """
    lam, b, factors, carriers = data
    sign = 2 * outer - 1
    sign2 = 2 * inner - 1
    first = b[outer] * factors[outer]
    second = b[inner] * factors[inner]
    numerator = first @ b[inner] - carriers[sign + 2] * (b[outer] @ second)
    result = np.empty_like(numerator)
    for i in range(len(lam)):
        for k in range(len(lam)):
            bb = lam[k] + 1j * (sign + sign2) * omega
            gap = lam[i] - bb
            if abs(gap) * duration >= 0.5:
                result[i, k] = numerator[i, k] / gap
            else:
                value = 0.0j
                for j in range(len(lam)):
                    product = b[outer, i, j] * b[inner, j, k]
                    if abs(product) != 0:
                        a = lam[j] + 1j * sign * omega
                        value += product * _k(lam[i], bb, a, duration)
                result[i, k] = value
    # Preserve the old repeated-inner-pole branch even for separated endpoints.
    # Its edge list is independent of the outer endpoint, so scan only m^2 edges.
    for j in range(len(lam)):
        a = lam[j] + 1j * sign * omega
        for k in range(len(lam)):
            bb = lam[k] + 1j * (sign + sign2) * omega
            if abs(bb - a) * duration >= 1e-6:
                continue
            for i in range(len(lam)):
                gap = lam[i] - bb
                if abs(gap) * duration < 0.5:
                    continue
                product = b[outer, i, j] * b[inner, j, k]
                if abs(product) != 0:
                    rational = (
                        factors[outer, i, j] - carriers[sign + 2] * factors[inner, j, k]
                    ) / gap
                    result[i, k] += product * (_k(lam[i], bb, a, duration) - rational)
    return result


@njit(nogil=True, cache=True)
def advance_modal(
    entry_data: ModalData,
    parameters: NDArray[np.float64],
    states: NDArray[np.complex128],
) -> None:
    """Advance aligned densities in place with the second-order modal Dyson path.

    entry_data borrows, in order, lam (M²,) per ns, modes and inverse
    (M²,M²) dimensionless, rotation Fourier coefficients (K,M,M)
    dimensionless, harmonic (K,) float64 integers, and couplings
    (ports,2,M²,M²) per ns per gain. The coupling axis orders signs -1,+1.

    parameters is float64 (rows,7), ordered duration_ns, Stark start phase_rad,
    Stark end phase_rad, signed probe frequency_GHz, gain, probe start theta_rad,
    port. Port is an integer carried as float64. states is complex128
    (rows,M,M), in the lab frame at each endpoint. The caller owns states
    and supplies writable storage, aligned rows and valid port indices.

    All model arrays and parameters are borrowed without mutation or retention.
    Only states is modified; zero-duration rows stay unchanged. Returns None.
    Inputs are already validated and aligned by the caller. NumPy/Numba failures
    propagate without fallback or retry.
    """
    lam, modes, inverse, coefficients, harmonic, couplings = entry_data
    # This loop runs in one GIL-free chunk; parallelism belongs to the engine.
    for row in range(len(states)):
        duration, stark_phase, end_phase, frequency, gain, theta, port = parameters[row]
        if duration == 0:
            continue
        start = rotation(coefficients, harmonic, stark_phase)
        end = rotation(coefficients, harmonic, end_phase)
        density = start.conj().T @ states[row] @ start
        u = inverse @ density.ravel()
        omega = 2 * np.pi * frequency
        factors, exponentials, carriers = _first_factors(lam, duration, omega)
        phases = np.exp(1j * np.array([-1, 1]) * theta)
        b = couplings[int(port)] * gain
        transformed = exponentials * u
        for outer in range(2):
            transformed += phases[outer] * ((b[outer] * factors[outer]) @ u)
            for inner in range(2):
                second = _second_contraction(
                    (lam, b, factors, carriers), duration, omega, outer, inner
                )
                transformed += phases[outer] * phases[inner] * (second @ u)
        density = (modes @ transformed).reshape(states.shape[1:])
        states[row] = end @ density @ end.conj().T


@njit(nogil=True, cache=True)
def _probe_limits(data, parameters) -> bool:
    energies, bath, xx, yy = data
    duration, _, _, frequency, gain, _, port_value = parameters
    if duration == 0:
        return True
    port = int(port_value)
    positive = (xx[port] - 1j * yy[port]) / 2
    negative = (xx[port] + 1j * yy[port]) / 2
    size = len(energies)
    for i in range(size):
        for j in range(i + 1, size):
            gap = energies[j] - energies[i]
            gamma = max(0.0, -bath[i * size + j, i * size + j].real)
            width2 = gamma**2 + duration**-2
            strength = 2 * abs(gain) * max(abs(positive[i, j]), abs(negative[i, j]))
            if strength / np.sqrt((abs(frequency) - abs(gap)) ** 2 + width2) > 0.1:
                return False
            if abs(gain) >= 0.1:
                virtual = 0.0
                for k in range(size):
                    virtual += (
                        gain**2
                        * abs(positive[i, k] * positive[k, j])
                        / max(abs(energies[k] - energies[i] - frequency), 1 / duration)
                    )
                if virtual / np.sqrt((2 * frequency - gap) ** 2 + width2) > 0.05:
                    return False
    return True


@njit(nogil=True, cache=True)
def legacy_mask(
    data: FrameData,
    bounds: Bounds,
    parameters: NDArray[np.float64],
    budget: float,
) -> NDArray[np.bool_]:
    """Return a new bool (rows,) k=0 omission/convergence admission mask.

    data borrows, in order, energies (M,) GHz, bath (M²,M²) per ns,
    and probe x/y (ports,M,M) GHz per gain. bounds is ordered valid bool,
    omitted drive norms (ports,) GHz per gain, bath omitted norm per ns,
    convergence, frame error. Convergence is the existing maximum of bath
    per-ns and drive GHz-per-gain zero-mode residuals. Frame error is
    dimensionless. budget is the absolute per-segment error allowance.

    parameters is float64 (rows,7), ordered duration_ns, Stark start phase_rad,
    Stark end phase_rad, signed probe frequency_GHz, gain, probe start theta_rad,
    port. Port is an integer carried as float64. The caller supplies validated,
    aligned inputs and valid port indices from the same frame.

    Inputs remain caller-owned and unmodified, with no retained references.
    The caller owns the new writable mask. NumPy/Numba failures propagate
    without fallback or retry; no validation or coercion is performed here.
    """
    energies, _, _, _ = data
    valid, omitted_norms, bath_omitted, convergence, frame_error = bounds
    result = np.zeros(len(parameters), dtype=np.bool_)
    for row in range(len(parameters)):
        duration, _, _, _, gain, _, port_value = parameters[row]
        port = int(port_value)
        if not valid:
            continue
        omitted = duration * (
            bath_omitted + 4 * np.pi * abs(gain) * omitted_norms[port]
        )
        omitted *= np.sqrt(len(energies))
        result[row] = (
            omitted <= budget / 3 and convergence * duration + frame_error <= budget / 3
        )
    return result


@njit(nogil=True, cache=True)
def dyson_mask(
    data: FrameData,
    strengths: NDArray[np.float64],
    parameters: NDArray[np.float64],
    budget: float,
) -> NDArray[np.bool_]:
    """Return a new bool (rows,) probe-limit/Dyson-remainder admission mask.

    data borrows, in order, energies (M,) GHz, bath (M²,M²) per ns,
    and probe x/y (ports,M,M) GHz per gain. strengths is float64 (ports,),
    the drive norm bound in GHz per gain. budget is the absolute per-segment
    error allowance. The caller has already admitted these rows for k=0.

    parameters is float64 (rows,7), ordered duration_ns, Stark start phase_rad,
    Stark end phase_rad, signed probe frequency_GHz, gain, probe start theta_rad,
    port. Port is an integer carried as float64. The caller supplies validated,
    aligned inputs and valid port indices from the same frame.

    Inputs remain caller-owned and unmodified, with no retained references.
    The caller owns the new writable mask. NumPy/Numba failures propagate
    without fallback or retry; no validation or coercion is performed here.
    """
    result = np.zeros(len(parameters), dtype=np.bool_)
    for row in range(len(parameters)):
        duration, _, _, _, gain, _, port = parameters[row]
        # Trace-norm contractivity bounds all higher orders for every density.
        v = 4 * np.pi * abs(gain) * strengths[int(port)] * duration
        result[row] = (
            _probe_limits(data, parameters[row])
            and v <= 0.5
            and v + v * v / 2 <= 0.02
            and np.exp(v) * v**3 / 6 <= budget / 3
        )
    return result


@njit(nogil=True, cache=True)
def _generator(data, tone, time):
    base, xx, yy = data
    omega, gain, theta = tone
    phase = omega * time + theta
    return base + gain * (np.cos(phase) * xx + np.sin(phase) * yy)


@njit(nogil=True, cache=True)
def _rk4(data, tone, time, step, channel):
    first = _generator(data, tone, time) @ channel
    middle = _generator(data, tone, time + step / 2)
    second = middle @ (channel + step / 2 * first)
    third = middle @ (channel + step / 2 * second)
    fourth = _generator(data, tone, time + step) @ (channel + step * third)
    return channel + step / 6 * (first + 2 * second + 2 * third + fourth)


@njit(nogil=True, cache=True)
def _integrate_channel(data, tone, duration, tolerance):
    """Adaptive step-doubled RK4 channel; tolerance is absolute error per ns."""
    channel = np.eye(len(data[0]), dtype=np.complex128)
    time = 0.0
    step = min(0.1, duration)
    for _ in range(1_000_000):
        if time >= duration:
            return channel
        step = min(step, duration - time)
        whole = _rk4(data, tone, time, step, channel)
        half = _rk4(data, tone, time, step / 2, channel)
        halves = _rk4(data, tone, time + step / 2, step / 2, half)
        error = np.max(np.abs(halves - whole)) / 15
        if not np.isfinite(error) or time + step == time:
            raise RuntimeError("Floquet period integration failed")
        # Step doubling cannot resolve errors below the channel's roundoff.
        # Without this floor, a long row's per-ns tolerance can shrink a
        # rejected step until time + step == time, even for unitary dynamics.
        allowance = max(
            tolerance * step,
            8 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(channel))),
        )
        if error <= allowance:
            channel = halves + (halves - whole) / 15
            time += step
        factor = 2.0 if error == 0 else 0.9 * (allowance / error) ** 0.2
        step *= min(2.0, max(0.2, factor))
    raise RuntimeError("Floquet period integration exceeded step limit")


@njit(nogil=True, cache=True)
def _period_channel(data, tone, duration, budget):
    """One probe period, binary channel powers, then a directly integrated tail."""
    frequency = abs(tone[0]) / (2 * np.pi)
    # A DC probe is periodic with any period; one ns avoids integrating the
    # full segment before powering when the carrier frequency is zero.
    period = 1 / frequency if frequency > 0 else 1.0
    cycles = int(np.floor(duration / period))
    tolerance = min(1e-9, budget / 100) / max(1.0, duration)
    result = np.eye(len(data[0]), dtype=np.complex128)
    if cycles:
        channel = _integrate_channel(data, tone, period, tolerance)
        exponent = cycles
        while exponent:
            if exponent % 2:
                result = channel @ result
            exponent //= 2
            if exponent:
                channel = channel @ channel
    remainder = max(0.0, duration - cycles * period)
    if remainder:
        result = _integrate_channel(data, tone, remainder, tolerance) @ result
    return result


@njit(nogil=True, cache=True)
def advance_periodic(
    entry_data: PeriodicData,
    parameters: NDArray[np.float64],
    states: NDArray[np.complex128],
    budget: float,
) -> None:
    """Advance aligned densities in place with complete periodic k=0 channels.

    entry_data borrows, in order, base (M²,M²) per ns, probe x/y
    (ports,M²,M²) Liouvillians per ns per gain, rotation Fourier coefficients
    (K,M,M) dimensionless, and harmonic (K,) float64 integers.
    budget is the absolute per-segment error allowance.

    parameters is float64 (rows,7), ordered duration_ns, Stark start phase_rad,
    Stark end phase_rad, signed probe frequency_GHz, gain, probe start theta_rad,
    port. Port is an integer carried as float64. states is complex128
    (rows,M,M), in the lab frame at each endpoint. The caller owns states
    and supplies writable storage, aligned rows and valid port indices.

    All model arrays and parameters are borrowed without mutation or retention.
    Only states is modified; zero-duration rows stay unchanged. Returns None.
    Inputs are already validated and aligned by the caller. Integration
    RuntimeError and NumPy/Numba failures propagate without fallback or retry.
    """
    base, xx, yy, coefficients, harmonic = entry_data
    for row in range(len(states)):
        duration, start_phase, end_phase, frequency, gain, theta, port = parameters[row]
        if duration == 0:
            continue
        start = rotation(coefficients, harmonic, start_phase)
        end = rotation(coefficients, harmonic, end_phase)
        density = start.conj().T @ states[row] @ start
        channel = _period_channel(
            (base, xx[int(port)], yy[int(port)]),
            (2 * np.pi * frequency, gain, theta),
            duration,
            budget,
        )
        density = (channel @ density.ravel()).reshape(states.shape[1:])
        states[row] = end @ density @ end.conj().T
