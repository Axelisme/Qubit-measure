"""Shared packed secular operators and playback for both DOP853 kernels."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ..kernel import SegmentTable
from ..secular import SecularModel


class Models(NamedTuple):
    """Read-only secular arrays packed over models (k), size M and ports P.

    energies: (k,M) GHz; rates/decay: (k,M,M) directional rates and total
    coherence loss per ns. x/y: (k,P,M,M) drive quadratures in GHz. basis:
    (k,M,M) eigenvectors in fixed coordinates. ports: (k,) valid port counts.
    """

    energies: NDArray[np.float64]
    rates: NDArray[np.float64]
    decay: NDArray[np.float64]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    basis: NDArray[np.complex128]
    ports: NDArray[np.int64]


class Segments(NamedTuple):
    """Numba playback arrays with SegmentTable semantics.

    timeline/start/duration/tones: (rows,) timeline index, ns support and count.
    port/frequency/gain/theta/envelope/offset: (rows,P) active tone slots with
    model port index, GHz, real gain, rad, envelope index and local offset ns.
    sample_offsets: (envelopes+1,) ragged bounds into sample_time (ns) and
    samples (complex amplitudes); an empty span denotes constant unity.
    flux_offsets/time: optional ragged absolute flux-waveform sample times for
    the row's sampled-control step ceiling. Empty offsets denote no waveform.
    """

    timeline: NDArray[np.int64]
    start: NDArray[np.float64]
    duration: NDArray[np.float64]
    tones: NDArray[np.int64]
    port: NDArray[np.int64]
    frequency: NDArray[np.float64]
    gain: NDArray[np.float64]
    theta: NDArray[np.float64]
    envelope: NDArray[np.int64]
    offset: NDArray[np.float64]
    sample_offsets: NDArray[np.int64]
    sample_time: NDArray[np.float64]
    samples: NDArray[np.complex128]
    flux_offsets: NDArray[np.int64]
    flux_time: NDArray[np.float64]


class Context(NamedTuple):
    """One RHS evaluation: packed arrays plus selected row/model indices."""

    models: Models
    segments: Segments
    row: int
    model: int


def pack_models(models: list[SecularModel]) -> Models:
    """Copy equal-size secular models to read-only arrays; reject size mismatch."""
    size = len(models[0].energies_ghz) if models else 0
    if any(len(model.energies_ghz) != size for model in models):
        raise ValueError("DOP853 models must share a state dimension")
    count = len(models)
    width = max((len(model.drives) for model in models), default=0)
    energies = np.empty((count, size))
    rates = np.empty((count, size, size))
    decay = np.empty_like(rates)
    x = np.zeros((count, width, size, size), dtype=np.complex128)
    y = np.zeros_like(x)
    basis = np.empty((count, size, size), dtype=np.complex128)
    ports = np.empty(count, dtype=np.int64)
    for index, model in enumerate(models):
        energies[index] = model.energies_ghz
        rates[index] = model.rates_per_ns
        outgoing = model.rates_per_ns.sum(axis=0)
        decay[index] = (
            outgoing[:, None] + outgoing[None, :]
        ) / 2 + model.dephasing_per_ns
        basis[index] = model.eigenvectors
        ports[index] = len(model.drives)
        for port, drive in enumerate(model.drives):
            x[index, port] = drive.in_phase_ghz
            y[index, port] = drive.quadrature_ghz
    packed = Models(energies, rates, decay, x, y, basis, ports)
    for array in packed:
        array.setflags(write=False)
    return packed


def pack_segments(table: SegmentTable) -> Segments:
    """Share table columns and concatenate its envelopes into numeric arrays."""
    offsets = np.zeros(len(table.envelopes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(envelope.samples) for envelope in table.envelopes])
    if table.envelopes:
        times = np.concatenate([envelope.sample_ns for envelope in table.envelopes])
        samples = np.concatenate([envelope.samples for envelope in table.envelopes])
    else:
        times = np.empty(0)
        samples = np.empty(0, dtype=np.complex128)
    return Segments(
        table.timeline,
        table.start_ns,
        table.duration_ns,
        table.tones,
        table.tone_port,
        table.tone_frequency_ghz,
        table.tone_amplitude,
        table.tone_theta_rad,
        table.tone_envelope,
        table.tone_envelope_offset_ns,
        offsets,
        times,
        samples,
        table.flux_waveform.offsets
        if table.flux_waveform is not None
        else np.empty(0, dtype=np.int64),
        table.flux_waveform.time_ns if table.flux_waveform is not None else np.empty(0),
    )


@njit(nogil=True, cache=True)
def hamiltonian(time, context):
    """Return H0 plus drives in row eigenbasis at segment-local time in ns."""
    models, segments, row, model = context
    size = models.energies.shape[1]
    h = np.zeros((size, size), dtype=np.complex128)
    for index in range(size):
        h[index, index] = models.energies[model, index]
    for tone in range(segments.tones[row]):
        env = segments.envelope[row, tone]
        begin, end = segments.sample_offsets[env : env + 2]
        amplitude = sample_value(
            time + segments.offset[row, tone],
            segments.sample_time[begin:end],
            segments.samples[begin:end],
        )
        phase = (
            segments.theta[row, tone] + 2 * np.pi * segments.frequency[row, tone] * time
        )
        z = segments.gain[row, tone] * amplitude * np.exp(1j * phase)
        port = segments.port[row, tone]
        h += z.real * models.x[model, port] + z.imag * models.y[model, port]
    return h
