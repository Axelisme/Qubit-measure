"""Calibrated finite-Fourier admission retaining slow modal bath/probe edges.

Bounds assume exact finite Fourier data and base/full trace contractivity.
Continuous-frame aliasing, full finite-Fourier generator contractivity, global
ODE error and directed rounding of norms/eigensystems are not certified.
Retained components need not be positive: their norm growth is charged explicitly.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

Complex = NDArray[np.complex128]
Real = NDArray[np.float64]
Integer = NDArray[np.int64]


@dataclass(frozen=True)
class Certificate:
    """Immutable omission data for a complete 2 MHz probe bin.

    center is the probe frequency in GHz. rows/columns index retained modal
    edges; values are their unit-gain coefficients in /ns. frequencies are
    bath harmonic angular frequencies in rad/ns. signs are 0 for bath and
    +/-1 for probe terms. q, r, e and tail each have two entries, bath then
    unit-gain probe: bounds on the averaging correction, retained generator,
    omitted resolved generator and sampled unresolved generator, respectively.
    q is dimensionless; r/e/tail are induced trace-norm bounds in /ns.
    """

    center: float
    rows: Integer
    columns: Integer
    values: Complex
    frequencies: Real
    signs: Integer
    q: Real
    r: Real
    e: Real
    tail: Real

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.rows,
                self.columns,
                self.values,
                self.frequencies,
                self.signs,
                self.q,
                self.r,
                self.e,
                self.tail,
            )
        )


@dataclass(frozen=True)
class Frame:
    """Immutable modal Fourier candidates and optional reusable certificates.

    frequencies are angular bath harmonic frequencies in rad/ns; coefficients
    are their modal generator matrices in /ns. ports are -1 for bath or the
    model drive index for probe. signs are 0 for bath and +/-1 for probe.
    tail holds spectral-norm bounds for all sampled |k|>4 terms, bath first
    then each unit-gain probe port. frequency is the Stark frequency in GHz.
    certificates pairs each probe port with its complete-bin omission data.
    """

    frequencies: Real
    coefficients: Complex
    ports: Integer
    signs: Integer
    tail: Real
    frequency: float
    certificates: tuple[tuple[int, Certificate], ...] = ()

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for a in (
                self.frequencies,
                self.coefficients,
                self.ports,
                self.signs,
                self.tail,
            )
        ) + sum(c.nbytes for _, c in self.certificates)


@dataclass(frozen=True)
class Plan:
    """Sparse interaction-picture equation and conditional trace-error bound.

    rows/columns index modal edges. values include signed gain and both tone
    phases, in /ns; exponents are complex modal divisors in /ns. bound is the
    dimensionless finite-Fourier trace-error estimate, infinity on unsafe
    dynamic range. An edge contributes values*exp(exponents*t) to the RHS.
    """

    rows: Integer
    columns: Integer
    values: Complex
    exponents: Complex
    bound: float


def _cheap_upper(matrix: Complex) -> float:
    absolute = np.abs(matrix)
    return float(
        min(
            np.linalg.norm(matrix),
            np.sqrt(absolute.sum(0).max() * absolute.sum(1).max()),
        )
    )


def _upper(matrix: Complex) -> float:
    gram = matrix.conj().T @ matrix
    # Schatten-8 dominates the spectral norm and uses only two GEMMs.
    return min(_cheap_upper(matrix), float(np.linalg.norm(gram @ gram) ** 0.25))


def build(
    bath: Complex,
    x: Complex,
    y: Complex,
    frequency: float,
    basis: tuple[Complex, Complex],
    comm: Callable[[Complex], Complex],
) -> Frame:
    """Build candidates for |k|<=4 and charge every other sampled term to tail.

    bath has shape (samples,M**2,M**2); x/y have shape (samples,ports,M,M),
    in /ns and GHz respectively. All use FFT ordering. frequency is the positive
    Stark frequency in GHz; basis is (right modes, inverse modes). comm maps
    a Hilbert Hamiltonian in GHz to its Liouville generator in /ns. Returned
    arrays own immutable allocations, charged by Frame.nbytes.
    """
    modes, inverse = basis
    frequencies, coefficients, ports, signs = [], [], [], []
    tail = np.zeros(x.shape[1] + 1)
    for index, harmonic in enumerate(np.fft.fftfreq(len(bath)) * len(bath)):
        if harmonic != 0:
            if abs(harmonic) <= 4:
                frequencies.append(2 * np.pi * harmonic * frequency)
                coefficients.append(inverse @ bath[index] @ modes)
                ports.append(-1)
                signs.append(0)
            else:
                tail[0] += _cheap_upper(bath[index])
        for port in range(x.shape[1]):
            for sign in (-1, 1):
                coefficient = comm((x[index, port] - sign * 1j * y[index, port]) / 2)
                if abs(harmonic) <= 4:
                    frequencies.append(2 * np.pi * harmonic * frequency)
                    coefficients.append(inverse @ coefficient @ modes)
                    ports.append(port)
                    signs.append(sign)
                else:
                    tail[port + 1] += _cheap_upper(coefficient)
    arrays = (
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(coefficients, dtype=np.complex128),
        np.asarray(ports, dtype=np.int64),
        np.asarray(signs, dtype=np.int64),
        tail,
    )
    for array in arrays:
        array.flags.writeable = False
    return Frame(*arrays, frequency)


def _certify(
    frame: Frame,
    lam: Complex,
    basis: tuple[Complex, Complex],
    center: float,
    port: int,
) -> Certificate:
    modes, inverse = basis
    size = int(round(np.sqrt(len(lam))))
    condition = np.linalg.norm(modes, 2) * np.linalg.norm(inverse, 2)
    q, r, e = np.zeros(2), np.zeros(2), np.zeros(2)
    rows, columns, values, frequencies, signs = [], [], [], [], []
    for nu, coefficient, term_port, sign in zip(
        frame.frequencies, frame.coefficients, frame.ports, frame.signs, strict=True
    ):
        if term_port >= 0 and term_port != port:
            continue
        slot = int(term_port >= 0)
        width = 2 * np.pi * 0.001 if slot else 0.0
        divisor = 1j * (nu + sign * 2 * np.pi * center) - lam[:, None] + lam[None, :]
        keep = np.abs(divisor) <= 0.1 + width
        retained = np.where(keep, coefficient, 0.0)
        omitted = np.where(keep, 0.0, coefficient)
        safe = np.where(keep, 1.0, divisor)
        r[slot] += np.sqrt(size) * _upper(modes @ retained @ inverse)
        e[slot] += np.sqrt(size) * _upper(modes @ omitted @ inverse)
        q[slot] += np.sqrt(size) * _upper(modes @ (omitted / safe) @ inverse)
        # Resolvent identity encloses Q across the entire bin, not just samples.
        delta = (
            np.abs(omitted)
            * width
            / (np.abs(safe) * np.maximum(np.abs(safe) - width, 1e-300))
        )
        q[slot] += np.sqrt(size) * condition * float(np.linalg.norm(delta))
        ri, ci = np.nonzero(retained)
        rows.extend(ri)
        columns.extend(ci)
        values.extend(retained[ri, ci])
        frequencies.extend(np.full(len(ri), nu))
        signs.extend(np.full(len(ri), sign))
    tail = np.sqrt(size) * frame.tail[[0, port + 1]]
    arrays = (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(values, dtype=np.complex128),
        np.asarray(frequencies, dtype=np.float64),
        np.asarray(signs, dtype=np.int64),
        q,
        r,
        e,
        tail,
    )
    for array in arrays:
        array.flags.writeable = False
    return Certificate(center, *arrays)


def _bound(certificate: Certificate, lam: Complex, parameters: Real) -> float:
    duration, gain = parameters[0], abs(parameters[4])
    q, r, e, h = (
        float(a @ np.array([1.0, gain]))
        for a in (certificate.q, certificate.r, certificate.e, certificate.tail)
    )
    if r * duration > 700 or np.ptp(lam.real) * duration > 14:
        return np.inf
    growth = np.exp(r * duration)
    integral = np.expm1(r * duration) / r if r else duration
    return float(q * (1 + growth) + ((e + h + 2 * r) * q + h) * integral)


def prepare(
    frame: Frame,
    lam: Complex,
    modes: Complex,
    inverse: Complex,
    parameters: Real,
) -> Frame:
    """Build immutable 2 MHz-bin certificates for the supplied parameter rows.

    lam/modes/inverse are the k=0 generator eigensystem. Each parameter row is
    (duration_ns, Stark start phase, Stark end phase, probe_GHz, probe gain,
    probe phase, probe port); phases are in radians. Only probe frequency/port
    determine the certificate. The sole retention window is 0.1/ns plus the
    bin half-width for probes. Return a new Frame; never mutate the input.
    """
    certificates = []
    centers = np.round(parameters[:, 3] / 0.002) * 0.002
    for port_value in np.unique(parameters[:, -1]):
        port = int(port_value)
        for center in np.unique(centers[parameters[:, -1] == port]):
            certificate = _certify(frame, lam, (modes, inverse), float(center), port)
            certificates.append((port, certificate))
    return replace(frame, certificates=tuple(certificates))


def plan(
    frame: Frame,
    lam: Complex,
    modes: Complex,
    inverse: Complex,
    parameters: Real,
) -> Plan:
    """Return sparse retained edges and a finite-Fourier omission bound.

    Eigensystem and the single parameter row follow prepare's units/layout.
    Missing bins are computed without mutating frame. Exact zero divisors are
    retained directly, never inverted. For Q'=AQ-QA+E the bound is q*(1+exp(r*T)) +
    ((e+h+2*r)*q+h)*integral(exp(r*t)). Base/full trace contractivity and exact
    Fourier/eigen data are assumptions, not certified by this calibrated guard.
    """
    port = int(parameters[-1])
    center = float(np.round(parameters[3] / 0.002) * 0.002)
    selected = next(
        (c for p, c in frame.certificates if p == port and c.center == center), None
    )
    if selected is None:
        selected = prepare(frame, lam, modes, inverse, parameters[None]).certificates[
            0
        ][1]
    phases = (
        selected.frequencies / (2 * np.pi * frame.frequency) * parameters[1]
        + selected.signs * parameters[5]
    )
    values = (
        selected.values
        * np.where(selected.signs == 0, 1.0, parameters[4])
        * np.exp(1j * phases)
    )
    exponents = (
        1j * (selected.frequencies + selected.signs * 2 * np.pi * parameters[3])
        - lam[selected.rows]
        + lam[selected.columns]
    )
    return Plan(
        selected.rows,
        selected.columns,
        values,
        exponents,
        _bound(selected, lam, parameters),
    )


@njit(nogil=True, cache=True)
def _rhs(time, state, rows, columns, values, exponents):
    result = np.zeros_like(state)
    for index in range(len(values)):
        result[rows[index]] += (
            values[index] * np.exp(exponents[index] * time) * state[columns[index]]
        )
    return result


def advance(
    selected: Plan,
    modal: tuple[Complex, Complex, Complex],
    start: Complex,
    end: Complex,
    duration: float,
    density: Complex,
) -> Complex:
    """Integrate slow edges with modal=(eigenvalues, right modes, inverse modes).

    Start/end map frame to physical basis. duration is in ns; density is an
    arbitrary (M,M) input, left unchanged. Return an unnormalized density.
    Failed integration raises RuntimeError; the kernel checks finite output.
    """
    lam, modes, inverse = modal
    initial = inverse @ (start.conj().T @ density @ start).ravel()
    if duration == 0:
        return density.copy()
    solution = solve_ivp(
        lambda time, state: _rhs(
            time,
            state,
            selected.rows,
            selected.columns,
            selected.values,
            selected.exponents,
        ),
        (0.0, duration),
        initial,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Retained Floquet integration failed: {solution.message}")
    result = (modes @ (np.exp(lam * duration) * solution.y[:, -1])).reshape(
        density.shape
    )
    return end @ result @ end.conj().T
