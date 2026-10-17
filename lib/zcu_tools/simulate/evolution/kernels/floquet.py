"""Constant Stark tone plus weak probe, in an extended-zone Floquet frame.

The two tones may use different ports. The larger absolute gain is the Stark
slot (ties choose the first slot). ModelSource identity namespaces engine
caches; keys additionally encode flux level, Stark port, frequency and gain.
All probe parameters and both start phases remain apply-only parameters.

P8 probe limits are r <= .1, population transfer <= .02 and r2 <= .05
(the two-photon check is enabled for |probe gain| >= .1). The linewidth is
sqrt(Gamma**2 + duration**-2), without dividing Gamma by 2*pi. Since guard
cannot read density, transfer is bounded in trace norm for *every* input.
An additional exponential-series remainder bounds orders >= 3 by budget.

Floquet validity requires assigned bare-mode overlap >= .5, modal condition
<= 1e8, maximum Re(lambda) <= 1e-10/ns, and unitary/periodic residual <= 1e-9.
The norm sum of *all* omitted harmonics (including k=+/-1) bounds k=0's
error by Duhamel's formula. This is stricter than a sampled k=0 versus +/-1
comparison. Harmonic truncation and quadrature convergence must each fit a third of
budget. Probe limits or Dyson-remainder failures use a one-period propagator
of H_F+D0 plus the probe, binary powering, and direct remainder integration. Frames needing more than 4096 phase
samples (spectral span including drive / Stark frequency > 128) are rejected.
No density is clipped or renormalized.
Only Floquet-invalid rows return to the engine; this module never calls
another kernel.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import eig, schur
from scipy.optimize import linear_sum_assignment

from ..kernel import GuardPreparation, GuardResult, Prepared, SegmentTable
from ..secular import ModelSource, SecularModel


@dataclass(frozen=True)
class _Entry:
    frequency: float
    energies: NDArray[np.float64]
    rotations: NDArray[np.complex128]
    harmonic: NDArray[np.float64]
    bath: NDArray[np.complex128]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    omitted: NDArray[np.float64]
    strengths: NDArray[np.float64]
    bath_omitted: float
    convergence: float
    frame_error: float
    lam: NDArray[np.complex128]
    modes: NDArray[np.complex128]
    inverse: NDArray[np.complex128]
    couplings: NDArray[np.complex128]
    valid: bool

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.energies,
                self.rotations,
                self.harmonic,
                self.bath,
                self.x,
                self.y,
                self.omitted,
                self.strengths,
                self.lam,
                self.modes,
                self.inverse,
                self.couplings,
            )
        )


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Entry, ...]

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)


def _comm(h: NDArray[np.complex128]) -> NDArray[np.complex128]:
    eye = np.eye(len(h))
    return -2j * np.pi * (np.kron(h, eye) - np.kron(eye, h.T))


def _bath(model: SecularModel) -> NDArray[np.complex128]:
    size = len(model.energies_ghz)
    loss = model.rates_per_ns.sum(axis=0)
    gamma = (loss[:, None] + loss[None, :]) / 2 + model.dephasing_per_ns
    result = np.diag(-gamma.ravel()).astype(np.complex128)
    index = np.arange(size) * (size + 1)
    result[np.ix_(index, index)] = model.rates_per_ns - np.diag(loss)
    return result


def _slots(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.argmax(np.abs(table.tone_amplitude[rows, :2]), axis=1).astype(np.int64)


def _fourier_norm(values: NDArray[np.complex128]) -> float:
    # Spectral norms in Hilbert-Schmidt coordinates give an input-independent
    # bound, unlike checking a single stationary initial density.
    return float(sum(np.linalg.norm(a, 2) for a in values[1:]))


def _frame_samples(model: SecularModel, row: tuple[int, float, float], count: int):
    port, frequency, amplitude = row
    size = len(model.energies_ghz)
    drive = model.drives[port]
    period = 1 / frequency

    def rhs(time, flat):
        phase = 2 * np.pi * frequency * time
        h = np.diag(model.energies_ghz) + amplitude * (
            np.cos(phase) * drive.in_phase_ghz + np.sin(phase) * drive.quadrature_ghz
        )
        return (-2j * np.pi * h @ flat.reshape(size, size)).ravel()

    times = np.linspace(0, period, count + 1)
    solution = solve_ivp(
        rhs,
        (0, period),
        np.eye(size, dtype=complex).ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        t_eval=times,
        max_step=period / count,
    )
    if not solution.success:
        raise RuntimeError(f"Floquet period integration failed: {solution.message}")
    final = solution.y[:, -1].reshape(size, size)
    decomposition = schur(final, output="complex")
    triangular = np.asarray(decomposition[0], dtype=np.complex128)
    vectors = np.asarray(decomposition[1], dtype=np.complex128)
    _, assignment = linear_sum_assignment(-(np.abs(vectors) ** 2))
    vectors = vectors[:, assignment]
    energies = -np.angle(np.diag(triangular)[assignment]) / (2 * np.pi * period)
    energies += np.round((model.energies_ghz - energies) / frequency) * frequency
    rotations = np.array(
        [
            solution.y[:, j].reshape(size, size)
            @ (vectors * np.exp(2j * np.pi * energies * t))
            for j, t in enumerate(times)
        ]
    )
    residual = max(
        float(np.max(abs(final.conj().T @ final - np.eye(size)))),
        float(np.max(abs(rotations[-1] - vectors))),
    )
    overlap = float(np.min(abs(np.diag(vectors)) ** 2))
    return energies, rotations[:-1], residual, overlap


def _averages(model: SecularModel, rotations: NDArray[np.complex128]):
    dissipator = _bath(model)
    baths = []
    xs = []
    ys = []
    for rotation in rotations:
        transform = np.kron(rotation, rotation.conj())
        baths.append(transform.conj().T @ dissipator @ transform)
        xs.append([rotation.conj().T @ d.in_phase_ghz @ rotation for d in model.drives])
        ys.append(
            [rotation.conj().T @ d.quadrature_ghz @ rotation for d in model.drives]
        )
    count = len(rotations)
    return (
        np.fft.fft(baths, axis=0) / count,
        np.fft.fft(xs, axis=0) / count,
        np.fft.fft(ys, axis=0) / count,
    )


def _sampling_ratio(model: SecularModel, key: np.void) -> float:
    drive = model.drives[int(key["port"])]
    extent = float(np.ptp(model.energies_ghz)) + 2 * abs(float(key["gain"])) * (
        np.linalg.norm(drive.in_phase_ghz, 2) + np.linalg.norm(drive.quadrature_ghz, 2)
    )
    return float(extent / float(key["frequency"]))


def _build(
    model: SecularModel, port: int, frequency: float, amplitude: float
) -> _Entry:
    if frequency <= 0:
        raise ValueError("Floquet Stark frequency must be positive")
    drive = model.drives[port]
    span = np.ptp(model.energies_ghz) + 2 * abs(amplitude) * (
        np.linalg.norm(drive.in_phase_ghz, 2) + np.linalg.norm(drive.quadrature_ghz, 2)
    )
    count = 2 ** int(np.ceil(np.log2(max(128, 32 * span / frequency))))
    if count > 4096:
        raise RuntimeError("Floquet frame exceeds phase sampling limit")
    energies, rotations, residual, overlap = _frame_samples(
        model, (port, frequency, amplitude), count
    )
    bath, x, y = _averages(model, rotations)
    coarse_bath, coarse_x, coarse_y = _averages(model, rotations[::2])
    convergence = float(
        max(
            np.linalg.norm(bath[0] - coarse_bath[0], 2),
            np.max(abs(x[0] - coarse_x[0])),
            np.max(abs(y[0] - coarse_y[0])),
        )
    )
    l0 = _comm(np.diag(energies).astype(complex)) + bath[0]
    decomposition = eig(l0)
    lam = np.asarray(decomposition[0], dtype=np.complex128)
    modes = np.asarray(decomposition[1], dtype=np.complex128)
    condition = np.linalg.cond(modes)
    inverse = (
        np.linalg.inv(modes) if condition <= 1e8 else np.linalg.pinv(modes)
    ).astype(np.complex128)
    couplings = np.array(
        [
            [inverse @ _comm((xx - sign * 1j * yy) / 2) @ modes for sign in (-1, 1)]
            for xx, yy in zip(x[0], y[0], strict=True)
        ]
    )
    valid = (
        residual <= 1e-9
        and overlap >= 0.5
        and condition <= 1e8
        and np.max(lam.real) <= 1e-10
    )
    omitted = np.array(
        [
            _fourier_norm(x[:, p]) + _fourier_norm(y[:, p])
            for p in range(len(model.drives))
        ]
    )
    strengths = np.array(
        [
            min(
                np.linalg.norm(xx, 2) + np.linalg.norm(yy, 2),
                np.linalg.norm((xx - 1j * yy) / 2, 2)
                + np.linalg.norm((xx + 1j * yy) / 2, 2),
            )
            for xx, yy in zip(x[0], y[0], strict=True)
        ],
        dtype=np.float64,
    )
    coefficients = np.fft.fft(rotations, axis=0) / count
    harmonic = np.asarray(np.fft.fftfreq(count) * count, dtype=np.float64)
    edge = np.abs(harmonic) >= count / 4
    frame_error = float(4 * sum(np.linalg.norm(c, 2) for c in coefficients[edge]))
    # Keep only k=0, not views retaining the full Fourier sample arrays;
    # Prepared.nbytes must account for every retained array allocation.
    bath_zero, x_zero, y_zero = bath[0].copy(), x[0].copy(), y[0].copy()
    arrays = (
        energies,
        coefficients,
        harmonic,
        bath_zero,
        x_zero,
        y_zero,
        omitted,
        strengths,
        lam,
        modes,
        inverse,
        couplings,
    )
    for array in arrays:
        array.flags.writeable = False
    return _Entry(
        frequency,
        energies,
        coefficients,
        harmonic,
        bath_zero,
        x_zero,
        y_zero,
        omitted,
        strengths,
        _fourier_norm(bath),
        convergence,
        frame_error,
        lam,
        modes,
        inverse,
        couplings,
        bool(valid),
    )


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
def _rotation(entry_coefficients, harmonic, phase):
    result = np.zeros(entry_coefficients.shape[1:], dtype=np.complex128)
    for k in range(len(harmonic)):
        result += np.exp(1j * harmonic[k] * phase) * entry_coefficients[k]
    return result


@njit(nogil=True, cache=True)
def _modal(entry_data, parameters, states):
    lam, modes, inverse, coefficients, harmonic, couplings = entry_data
    # This loop runs in one GIL-free chunk; parallelism belongs to the engine.
    for row in range(len(states)):
        duration, stark_phase, end_phase, frequency, gain, theta, port = parameters[row]
        if duration == 0:
            continue
        start = _rotation(coefficients, harmonic, stark_phase)
        end = _rotation(coefficients, harmonic, end_phase)
        density = start.conj().T @ states[row] @ start
        u = inverse @ density.ravel()
        transformed = np.empty(len(lam), dtype=np.complex128)
        b = couplings[int(port)] * gain
        omega = 2 * np.pi * frequency
        factors, exponentials, carriers = _first_factors(lam, duration, omega)
        phases = np.exp(1j * np.array([-1, 1]) * theta)
        for i in range(len(lam)):
            value = exponentials[i] * u[i]
            for outer in range(2):
                sign = 2 * outer - 1
                phase = phases[outer]
                for j in range(len(lam)):
                    first = b[outer, i, j] * phase
                    if abs(first) == 0:
                        continue
                    a = lam[j] + 1j * sign * omega
                    value += first * u[j] * factors[outer, i, j]
                    for inner in range(2):
                        sign2 = 2 * inner - 1
                        phase2 = phases[inner]
                        for k in range(len(lam)):
                            product = first * b[inner, j, k] * phase2 * u[k]
                            if abs(product) != 0:
                                bb = lam[k] + 1j * (sign + sign2) * omega
                                gap = lam[i] - bb
                                if (
                                    abs(bb - a) * duration >= 1e-6
                                    and abs(gap) * duration >= 0.5
                                ):
                                    # Shift the second edge's first factor by
                                    # the outer carrier before differencing.
                                    second = (
                                        factors[outer, i, j]
                                        - carriers[sign + 2] * factors[inner, j, k]
                                    ) / gap
                                else:
                                    # Preserve the existing repeated-pole and
                                    # close-endpoint formulas without regrouping.
                                    second = _k(lam[i], bb, a, duration)
                                value += product * second
            transformed[i] = value
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
def _mask(data, bounds, parameters, budget):
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


def _acceptance(
    entry: _Entry, parameters: NDArray[np.float64], budget: float
) -> NDArray[np.bool_]:
    return _mask(
        (entry.energies, entry.bath, entry.x, entry.y),
        (
            entry.valid,
            entry.omitted,
            entry.bath_omitted,
            entry.convergence,
            entry.frame_error,
        ),
        parameters,
        budget,
    )


@njit(nogil=True, cache=True)
def _dyson_mask(data, strengths, parameters, budget):
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
        if error <= tolerance * step:
            channel = halves + (halves - whole) / 15
            time += step
        factor = 2.0 if error == 0 else 0.9 * (tolerance * step / error) ** 0.2
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
def _period_batch(entry_data, parameters, states, budget):
    base, xx, yy, coefficients, harmonic = entry_data
    for row in range(len(states)):
        duration, start_phase, end_phase, frequency, gain, theta, port = parameters[row]
        if duration == 0:
            continue
        start = _rotation(coefficients, harmonic, start_phase)
        end = _rotation(coefficients, harmonic, end_phase)
        density = start.conj().T @ states[row] @ start
        channel = _period_channel(
            (base, xx[int(port)], yy[int(port)]),
            (2 * np.pi * frequency, gain, theta),
            duration,
            budget,
        )
        density = (channel @ density.ravel()).reshape(states.shape[1:])
        states[row] = end @ density @ end.conj().T


def _parameters(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
    stark = _slots(table, rows)
    probe = 1 - stark
    duration = table.duration_ns[rows]
    phase = table.tone_theta_rad[rows, stark]
    return np.column_stack(
        (
            duration,
            phase,
            phase + 2 * np.pi * table.tone_frequency_ghz[rows, stark] * duration,
            table.tone_frequency_ghz[rows, probe],
            table.tone_amplitude[rows, probe],
            table.tone_theta_rad[rows, probe],
            table.tone_port[rows, probe],
        )
    ).astype(np.float64)


def _advance(
    entry: _Entry,
    parameters: NDArray[np.float64],
    states: NDArray[np.complex128],
    budget: float,
) -> None:
    """Dispatch two batches, never one Python call per probe point."""
    dyson = _dyson_mask(
        (entry.energies, entry.bath, entry.x, entry.y),
        entry.strengths,
        parameters,
        budget,
    )
    selected = states[dyson].copy()
    if len(selected):
        _modal(
            (
                entry.lam,
                entry.modes,
                entry.inverse,
                entry.rotations,
                entry.harmonic,
                entry.couplings,
            ),
            parameters[dyson],
            selected,
        )
        states[dyson] = selected
    selected = states[~dyson].copy()
    if len(selected):
        _period_batch(
            (
                _comm(np.diag(entry.energies).astype(complex)) + entry.bath,
                np.array([_comm(x) for x in entry.x]),
                np.array([_comm(y) for y in entry.y]),
                entry.rotations,
                entry.harmonic,
            ),
            parameters[~dyson],
            selected,
            budget,
        )
        states[~dyson] = selected


class FloquetKernel:
    """C5 state-independent Floquet preparation and batched modal Dyson apply.

    Methods accept the SegmentTable/ModelSource contract in ns/GHz/rad.
    prepare raises RuntimeError on failed numerical construction; apply
    requires this kernel's prepared payload and advances states in place.
    """

    name = "floquet"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Return validity for two constant tones, never reading initial density."""
        return self.guard_prepared(table, rows, source, budget).accepted

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> GuardResult:
        """Return the unchanged guard mask and evolve-local Floquet preparations.

        Each key's immutable model-only payload can replace prepare for the
        same source and budget. No preparation is retained on this kernel.
        """
        preparations: list[GuardPreparation] = []
        accepted = np.zeros(len(rows), dtype=bool)
        if table.tone_port.shape[1] < 2:
            return GuardResult(accepted)
        constants = np.array([e.kind == "constant" for e in table.envelopes])
        playing = table.tones[rows] == 2
        envelopes = np.maximum(0, table.tone_envelope[rows, :2])
        constant = (
            constants[envelopes].all(axis=1)
            if len(constants)
            else np.zeros(len(rows), dtype=bool)
        )
        stark = _slots(table, rows)
        frequency = table.tone_frequency_ghz[rows, stark]
        positions = np.flatnonzero(playing & constant & (frequency > 0))
        if not len(positions):
            return GuardResult(accepted)
        selected = rows[positions]
        keys, representatives, inverse = np.unique(
            self.prepare_keys(table, selected), return_index=True, return_inverse=True
        )
        params = _parameters(table, selected)
        for key_index, representative in enumerate(representatives):
            level = tuple(int(v) for v in keys[key_index]["level"])
            model = source.model(level)
            if _sampling_ratio(model, keys[key_index]) > 128:
                continue
            prepared = self.prepare(
                keys[key_index : key_index + 1],
                table,
                selected[representative : representative + 1],
                source,
                budget,
            )
            preparations.append(
                GuardPreparation(keys[key_index : key_index + 1], prepared)
            )
            members = np.flatnonzero(inverse == key_index)
            accepted[positions[members]] = _acceptance(
                prepared.entries[0], params[members], budget
            )
        return GuardResult(accepted, tuple(preparations))

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Encode level, Stark port/frequency/gain; probe and phases are excluded."""
        keys = np.empty(
            len(rows),
            dtype=[
                ("level", "i8", (table.level.shape[1],)),
                ("port", "i8"),
                ("frequency", "f8"),
                ("gain", "f8"),
            ],
        )
        keys["level"] = table.level[rows]
        stark = _slots(table, rows)
        keys["port"] = table.tone_port[rows, stark]
        keys["frequency"] = table.tone_frequency_ghz[rows, stark]
        keys["gain"] = table.tone_amplitude[rows, stark]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Finite relative estimates; modal work is independent of carrier cycles."""
        return np.ones(len(table.duration_ns[rows]))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> _Prepared:
        """Prepare one immutable, density-independent Floquet frame per key."""
        if budget <= 0 or not np.array_equal(keys, self.prepare_keys(table, rows)):
            raise ValueError(
                "Floquet requires positive budget and aligned representative rows"
            )
        entries = []
        for key in keys:
            model = source.model(tuple(int(v) for v in key["level"]))
            try:
                entries.append(
                    _build(
                        model,
                        int(key["port"]),
                        float(key["frequency"]),
                        float(key["gain"]),
                    )
                )
            except np.linalg.LinAlgError as error:
                raise RuntimeError("Floquet modal construction failed") from error
        return _Prepared(tuple(entries))

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
        """Advance arbitrary densities, retaining start and end micromotion.

        Rows must already pass guard. Wrong payload raises ValueError; nonfinite
        density or an out-of-budget row raises RuntimeError with its segment.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        if not isinstance(prepared, _Prepared):
            raise ValueError("Floquet requires its own prepared payload")
        parameters = _parameters(table, rows)
        for key in np.unique(key_index):
            positions = np.flatnonzero(key_index == key)
            entry = prepared.entries[int(key)]
            mask = _acceptance(entry, parameters[positions], budget)
            if not mask.all():
                failed = rows[positions[~mask]]
                raise RuntimeError(f"Floquet guard failed at segments {failed}")
            chunk = states[positions].copy()
            _advance(entry, parameters[positions], chunk, budget)
            if not np.isfinite(chunk).all():
                raise RuntimeError(
                    f"Floquet nonfinite density at segments {rows[positions]}"
                )
            states[positions] = chunk
        return np.zeros(len(rows), dtype=np.bool_)
