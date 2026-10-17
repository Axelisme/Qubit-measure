"""Common-control density integration in the exact diagonal-H0 interaction frame.

Secular dissipation is invariant under the diagonal phase transformation. The
full transformed drive is retained, without RWA, splitting or resampling.
"""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ._dop853 import Tolerances, integrate_rows
from ._dop853_chain import sample_ceiling
from ._dop853_data import Models, Segments


class GroupContext(NamedTuple):
    """Common playback for independent single-tone rows, differing only in gain.

    models/segments: packed read-only model and playback arrays. rows: int64
    vector of selected table rows. model: their common prepared-model index.
    Time is local to the interval; densities enter and leave in model eigenbasis.
    """

    models: Models
    segments: Segments
    rows: NDArray[np.int64]
    model: int


@njit(nogil=True, cache=True)
def _rhs(time, states, context):
    models, segments, rows, model = context
    size = models.energies.shape[1]
    densities = states.reshape(len(rows), size, size)
    result = np.empty_like(densities)
    representative = rows[0]
    envelope = segments.envelope[representative, 0]
    begin, end = segments.sample_offsets[envelope : envelope + 2]
    amplitude = sample_value(
        time + segments.offset[representative, 0],
        segments.sample_time[begin:end],
        segments.samples[begin:end],
    )
    phase = (
        segments.theta[representative, 0]
        + 2 * np.pi * segments.frequency[representative, 0] * time
    )
    z = amplitude * np.exp(1j * phase)
    port = segments.port[representative, 0]
    # Build the unit-gain drive directly: subtracting H0 from a full Hamiltonian
    # would lose small diagonal drive terms to cancellation at weak gains.
    h = z.real * models.x[model, port] + z.imag * models.y[model, port]
    phases = np.exp(2j * np.pi * models.energies[model] * time)
    for i in range(size):
        for j in range(size):
            h[i, j] *= phases[i] * phases[j].conjugate()
    for index in range(len(rows)):
        rho = densities[index]
        gain = segments.gain[rows[index], 0]
        product = h @ rho
        for i in range(size):
            for j in range(size):
                result[index, i, j] = (
                    -2j * np.pi * gain * (product[i, j] - product[j, i].conjugate())
                    - models.decay[model, i, j] * rho[i, j]
                )
            for j in range(size):
                result[index, i, i] += models.rates[model, i, j] * rho[j, j]
    return result.ravel()


@njit(nogil=True, cache=True)
def integrate_group(context, states, controls):
    """Return model-basis endpoints for one independent common-control group.

    Use the original tableau, tolerances, 0.1 ns seed and active-knot ceiling.
    Every row must pass its own normalized RMS estimator. The callback-free
    compiled signature depends on array types, never group or model dimension.
    Numerical failures raise RuntimeError without retry or density correction.
    """
    models, segments, rows, model = context
    count, size, _ = states.shape
    state = states.copy().ravel()
    duration = segments.duration[rows[0]]
    ceiling = min(controls.max_step, sample_ceiling(segments, rows[0]))
    state = integrate_rows(
        _rhs,
        context,
        state,
        size * size,
        duration,
        Tolerances(controls.rtol, controls.atol, ceiling),
    )
    result = state.reshape(count, size, size)
    phases = np.exp(-2j * np.pi * models.energies[model] * duration)
    for i in range(size):
        for j in range(size):
            result[:, i, j] *= phases[i] * phases[j].conjugate()
    return result


def grouped_apply(
    models: Models,
    segments: Segments,
    rows: NDArray[np.int64],
    key_index: NDArray[np.int64],
    states: NDArray[np.complex128],
    controls: Tolerances,
) -> bool:
    """Advance independent single-tone rows grouped by exact controls and octave.

    Group by prepared model, duration, port, signed carrier, phase, envelope
    identity, envelope-local offset and floor(log2(abs(gain))). Only real gain
    differs. Update states in place and return True. Return False without mutation
    if any row has zero gain, a non-single tone count, or touches its preceding
    same-timeline row; the caller must use the unchanged lab-frame chain runner.
    """
    if np.any(segments.tones[rows] != 1) or np.any(segments.gain[rows, 0] == 0):
        return False
    for previous, row in zip(rows[:-1], rows[1:], strict=True):
        if (
            segments.timeline[previous] == segments.timeline[row]
            and abs(
                segments.start[row]
                - segments.start[previous]
                - segments.duration[previous]
            )
            <= 1e-9
        ):
            return False
    groups: dict[tuple[int, float, int, float, float, int, float, int], list[int]] = {}
    for slot, (row, model) in enumerate(zip(rows, key_index, strict=True)):
        gain = segments.gain[row, 0]
        key = (
            int(model),
            float(segments.duration[row]),
            int(segments.port[row, 0]),
            float(segments.frequency[row, 0]),
            float(segments.theta[row, 0]),
            int(segments.envelope[row, 0]),
            float(segments.offset[row, 0]),
            int(np.floor(np.log2(abs(gain)))),
        )
        groups.setdefault(key, []).append(slot)
    for key, slots in groups.items():
        indices = np.asarray(slots, dtype=np.int64)
        states[indices] = integrate_group(
            GroupContext(models, segments, rows[indices], key[0]),
            states[indices],
            controls,
        )
    return True
