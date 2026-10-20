"""Shared Hilbert endpoint integration for gate-cache phase grids."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ..kernel import SegmentTable
from ..secular import SecularModel
from ._dop853 import Tolerances, integrate

GATE_SUBSEGMENTS = 4


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


def build_endpoints(
    model: SecularModel,
    table: SegmentTable,
    row: int,
    budget: float,
    phases: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return unitary subsegments with shape (4, len(phases), M, M).

    phases contains carrier phases in radians at the segment's local time zero.
    model and the sampled single-tone table row supply the Hamiltonian and
    envelope; M is the model dimension. budget sets the DOP853 tolerances.
    Each of four Strang subsegments starts from identity, restarts integration
    at envelope knots, and retracts its endpoint with an SVD polar projection.
    No density is accessed. Numerical integration failures propagate.
    """
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
    samples = np.empty((GATE_SUBSEGMENTS, count, size, size), dtype=complex)
    # Reserve the segment budget for Strang/Fourier errors. A 1e-8 U tolerance
    # avoids oversolving high-energy modes while retaining budget scaling.
    tolerance = min(1e-8, budget * 1e-4)
    controls = Tolerances(
        tolerance,
        tolerance * 0.01,
        min(0.1, 0.25 / max(abs(frequency), 1e-12)),
        np.inf,
        0,
    )
    knots = envelope.sample_ns - offset
    for segment in range(GATE_SUBSEGMENTS):
        start, end = (
            duration * segment / GATE_SUBSEGMENTS,
            duration * (segment + 1) / GATE_SUBSEGMENTS,
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
