"""Model-only phase convergence for short gate-cache Hilbert preparation."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ..kernel import SegmentTable
from ..secular import SecularModel
from ._dop853 import Tolerances, integrate

_SUBSEGMENTS = 4


class _GateContext(NamedTuple):
    # Hamiltonian in GHz, envelope-local offset and interval origin in ns.
    h0: NDArray[np.complex128]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    frequency: float
    amplitude: float
    offset: float
    sample_ns: NDArray[np.float64]
    samples: NDArray[np.complex128]
    origin: float
    phases: NDArray[np.float64]


@njit(nogil=True, cache=True)
def _gate_rhs(time, flat, context):
    time += context.origin
    envelope = context.amplitude * sample_value(
        time + context.offset, context.sample_ns, context.samples
    )
    size = len(context.h0)
    count = len(flat) // (size * size)
    states = flat.reshape((count, size, size))
    result = np.empty_like(states)
    for phase in range(count):
        z = envelope * np.exp(
            1j * (context.phases[phase] + 2 * np.pi * context.frequency * time)
        )
        h = context.h0 + z.real * context.x + z.imag * context.y
        result[phase] = (-2j * np.pi) * (h @ states[phase])
    return result.ravel()


def _unitaries(
    model: SecularModel,
    table: SegmentTable,
    row: int,
    budget: float,
    phases: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Integrate phase nodes in compiled DOP853, restarting at sample knots."""
    count = len(phases)
    size = len(model.energies_ghz)
    drive = model.drives[int(table.tone_port[row, 0])]
    envelope = table.envelopes[int(table.tone_envelope[row, 0])]
    offset = float(table.tone_envelope_offset_ns[row, 0])
    frequency = float(table.tone_frequency_ghz[row, 0])
    amplitude = float(table.tone_amplitude[row, 0])
    duration = float(table.duration_ns[row])
    energies = model.energies_ghz
    h0 = np.diag(energies - (energies[0] + energies[-1]) / 2).astype(complex)
    identity = np.broadcast_to(np.eye(size, dtype=complex), (count, size, size))
    samples = np.empty((_SUBSEGMENTS, count, size, size), dtype=complex)
    # Reserve the segment budget for Strang/Fourier errors. A 1e-8 U tolerance
    # avoids oversolving high-energy modes while retaining budget scaling.
    tolerance = min(1e-8, budget * 1e-4)
    controls = Tolerances(
        tolerance, tolerance * 0.01, min(0.1, 0.25 / max(abs(frequency), 1e-12))
    )
    knots = envelope.sample_ns - offset
    for segment in range(_SUBSEGMENTS):
        start, end = (
            duration * segment / _SUBSEGMENTS,
            duration * (segment + 1) / _SUBSEGMENTS,
        )
        boundaries = np.concatenate(
            ([start], knots[(knots > start) & (knots < end)], [end])
        )
        unitary = identity.copy().ravel()
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            if right == left:
                continue
            context = _GateContext(
                h0,
                drive.in_phase_ghz,
                drive.quadrature_ghz,
                frequency,
                amplitude,
                offset,
                envelope.sample_ns,
                envelope.samples,
                float(left),
                phases,
            )
            unitary = integrate(
                _gate_rhs, context, unitary, float(right - left), controls, 0.01
            )
        samples[segment] = unitary.reshape(identity.shape)
    # The solver's small norm loss must not accumulate across repeated gates.
    # Retract the Hilbert endpoints, never the evolving density.
    left, _, right = np.linalg.svd(samples)
    return left @ right


def _interpolate(
    samples: NDArray[np.complex128], phases: NDArray[np.float64]
) -> NDArray[np.complex128]:
    count = samples.shape[1]
    modes = np.fft.fftfreq(count) * count
    coefficients = np.fft.fft(samples, axis=1) / count
    return np.einsum(
        "pm,smij->spij", np.exp(1j * phases[:, None] * modes), coefficients
    )


def build(
    model: SecularModel, table: SegmentTable, row: int, budget: float
) -> NDArray[np.complex128] | None:
    """Return checked (4,16,M,M) coefficients, or None to request legacy build.

    Require a sampled single tone of positive duration <=30 ns and a finite
    positive budget. Integrate six equispaced phases and two independent probes
    (pi/6 and sqrt(3) radians), using unchanged temporal tolerances and knot
    restarts. Every Strang endpoint must match both probes in full-unitary
    Frobenius norm within min(1e-4, budget*0.1). A failed phase check returns None;
    the caller must build its original sixteen-node payload instead.

    Retract interpolated unitaries at the original sixteen phase nodes before
    Fourier transformation. No density is accessed or normalized. This empirical
    convergence check is not a rigorous global bound for arbitrary harmonics.
    Invalid rows/budgets raise ValueError; numerical failures propagate.
    """
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("phase budget must be finite and positive")
    if table.tones[row] != 1 or not 0 < table.duration_ns[row] <= 30:
        raise ValueError("phase preparation requires a short single-tone gate")
    envelope = table.envelopes[int(table.tone_envelope[row, 0])]
    if envelope.kind != "sampled":
        raise ValueError("phase preparation requires a sampled envelope")
    phases = np.asarray(np.arange(6) * (2 * np.pi / 6), dtype=np.float64)
    # The irrational second probe avoids sharing a single rational phase grid's
    # harmonic aliases. Both probes share the adaptive integration batch.
    probes = np.array([np.pi / 6, np.sqrt(3)], dtype=np.float64)
    samples = _unitaries(model, table, row, budget, np.r_[phases, probes])
    predicted = _interpolate(samples[:, :6], probes)
    defect = float(np.max(np.linalg.norm(predicted - samples[:, 6:], axis=(-2, -1))))
    if defect > min(1e-4, budget * 0.1):
        return None
    # Polar projection removes radial interpolation error, not density error.
    full_phases = np.asarray(np.arange(16) * (2 * np.pi / 16), dtype=np.float64)
    expanded = _interpolate(samples[:, :6], full_phases)
    left, _, right = np.linalg.svd(expanded)
    return np.fft.fft(left @ right, axis=1) / 16
