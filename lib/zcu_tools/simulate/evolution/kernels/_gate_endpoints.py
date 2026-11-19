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


class _MultitoneContext(NamedTuple):
    # Both carrier axes share time-only envelope/carrier evaluation.
    h0: NDArray[np.complex128]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    frequencies: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    offsets: NDArray[np.float64]
    times: NDArray[np.float64]
    samples: NDArray[np.complex128]
    lengths: NDArray[np.int64]
    origin: float
    phase_factors: NDArray[np.complex128]


@njit(nogil=True, cache=True)
def _multitone_rhs(time, flat, context):
    time += context.origin
    size = len(context.h0)
    states = flat.reshape((-1, size, size))
    result = np.empty_like(states)
    carriers = np.empty(2, dtype=np.complex128)
    for tone in range(2):
        length = context.lengths[tone]
        carriers[tone] = (
            context.amplitudes[tone]
            * sample_value(
                time + context.offsets[tone],
                context.times[tone, :length],
                context.samples[tone, :length],
            )
            * np.exp(2j * np.pi * context.frequencies[tone] * time)
        )
    for phase in range(len(states)):
        h = context.h0.copy()
        for tone in range(2):
            z = carriers[tone] * context.phase_factors[phase, tone]
            h += z.real * context.x[tone] + z.imag * context.y[tone]
        result[phase] = (-2j * np.pi) * (h @ states[phase])
    return result.ravel()


def build_multitone_endpoints(
    model: SecularModel,
    table: SegmentTable,
    row: int,
    phases: NDArray[np.float64],
    controls: Tolerances,
) -> NDArray[np.complex128]:
    """Integrate full-drive (4, phases, M, M) dual-tone Hilbert endpoints.

    phases has shape (count,2), in radians at row-local zero. controls sets
    local DOP853 error and step limits. Both ordered ports, signed gains and
    complex envelopes come from the row. All phase nodes share one batch
    integration, with time-only factors evaluated once per RHS.
    Restart at the union of envelope knots. Each quarter starts at identity
    and is polar-retracted independently. No RWA, pruning or density is used.
    Require two tones and positive duration; solver failures propagate.
    """
    if table.tones[row] != 2 or table.duration_ns[row] <= 0:
        raise ValueError("full-drive endpoints require a positive dual-tone row")
    if phases.ndim != 2 or phases.shape[1] != 2:
        raise ValueError("dual-tone phases must have shape (count,2)")
    envelopes = [table.envelopes[e] for e in table.tone_envelope[row, :2]]
    lengths = np.array([len(env.samples) for env in envelopes], dtype=np.int64)
    width = max(1, int(lengths.max()))
    times = np.zeros((2, width), dtype=np.float64)
    values = np.zeros((2, width), dtype=np.complex128)
    for tone, env in enumerate(envelopes):
        times[tone, : lengths[tone]] = env.sample_ns
        values[tone, : lengths[tone]] = env.samples
    energies = model.energies_ghz
    h0 = np.diag(energies - (energies[0] + energies[-1]) / 2).astype(complex)
    drives = [model.drives[p] for p in table.tone_port[row, :2]]
    context = _MultitoneContext(
        h0,
        np.stack([drive.in_phase_ghz for drive in drives]),
        np.stack([drive.quadrature_ghz for drive in drives]),
        table.tone_frequency_ghz[row, :2].copy(),
        table.tone_amplitude[row, :2].copy(),
        table.tone_envelope_offset_ns[row, :2].copy(),
        times,
        values,
        lengths,
        0.0,
        np.exp(1j * phases),
    )
    knots = np.unique(
        np.concatenate(
            [
                env.sample_ns - context.offsets[tone]
                for tone, env in enumerate(envelopes)
            ]
        )
    )
    size = len(energies)
    identity = np.broadcast_to(np.eye(size, dtype=complex), (len(phases), size, size))
    endpoints = np.empty((GATE_SUBSEGMENTS, *identity.shape), dtype=complex)
    duration = float(table.duration_ns[row])
    for segment in range(GATE_SUBSEGMENTS):
        start = duration * segment / GATE_SUBSEGMENTS
        end = duration * (segment + 1) / GATE_SUBSEGMENTS
        boundaries = np.r_[start, knots[(knots > start) & (knots < end)], end]
        unitary = identity.copy().ravel()
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            unitary = integrate(
                _multitone_rhs,
                context._replace(origin=float(left)),
                unitary,
                float(right - left),
                controls,
                0.01,
            )
        left_u, _, right_u = np.linalg.svd(unitary.reshape(identity.shape))
        endpoints[segment] = left_u @ right_u
    return endpoints
