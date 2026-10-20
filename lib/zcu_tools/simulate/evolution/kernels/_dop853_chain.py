"""Shared single-threaded interval-chain control for both DOP853 kernels."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._dop853 import Tolerances, integrate


@njit(nogil=True, cache=True)
def _sample_ceiling(segments, row):
    ceiling = np.inf
    for tone in range(segments.tones[row]):
        envelope = segments.envelope[row, tone]
        begin, end = segments.sample_offsets[envelope : envelope + 2]
        local_start = segments.offset[row, tone]
        local_end = local_start + segments.duration[row]
        times = segments.sample_time[begin:end]
        first = begin + np.searchsorted(times, local_start, side="left")
        stop = begin + np.searchsorted(times, local_end, side="right")
        for index in range(first + 1, stop):
            ceiling = min(
                ceiling, segments.sample_time[index] - segments.sample_time[index - 1]
            )
    if len(segments.flux_offsets):
        point = segments.timeline[row]
        begin, end = segments.flux_offsets[point : point + 2]
        start = segments.start[row]
        times = segments.flux_time[begin:end]
        first = begin + np.searchsorted(times, start, side="left")
        stop = begin + np.searchsorted(
            times, start + segments.duration[row], side="right"
        )
        for index in range(first + 1, stop):
            ceiling = min(
                ceiling, segments.flux_time[index] - segments.flux_time[index - 1]
            )
    return ceiling


# Inline callback-taking control into the kernel's cached entry point, so no
# process-specific dispatcher pointer becomes a dynamic global in its cache.
@njit(nogil=True, cache=True, inline="always")
def run_chain(
    rhs,
    make_context,
    data,
    row_models: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]],
    states: NDArray[np.complex128],
    controls: Tolerances,
):
    """Advance touching same-timeline rows and store every interval endpoint.

    rhs(time, flat_state, context) and make_context(data, row, model) are
    compiled kernel-owned callables. data supplies packed models and segments.
    row_models is (rows, model_index, global_chain_ids). The first two vectors
    align with states (rows, M, M); chain IDs are indexed by global table row.
    Each chain reads only its first input slot; subsequent rows use the previous
    endpoint, changing basis on model changes. Disjoint rows read their own
    input. Each row resets a deterministic 0.1 ns initial seed, then adapts up
    to the smallest spacing between knots inside that row's envelope-local
    windows and, when packed, absolute flux window. Constant envelopes and
    ordinary DOP853's unpacked flux add no sampling ceiling. Chain membership
    comes from dispatch, not apply-call adjacency. A finite controls.phase_budget caps
    local rtol by that allowance divided by bare spectral phase over this row,
    scaling atol by the same factor. Lab-frame ordinary density propagation
    supplies the allowance; interaction and continuous-flux callers retain
    their original controls. This is a calibrated margin, not a global bound.
    Integration failures propagate; earlier endpoints may already be stored.
    """
    models, segments = data.models, data.segments
    rows, model_index, chain = row_models
    for index in range(len(rows)):
        row, model = rows[index], model_index[index]
        state = states[index].copy()
        if index > 0:
            previous = rows[index - 1]
            if chain[row] >= 0 and chain[previous] == chain[row]:
                previous_model = model_index[index - 1]
                state = states[index - 1].copy()
                if previous_model != model:
                    overlap = (
                        models.basis[model].conj().T @ models.basis[previous_model]
                    )
                    state = overlap @ state @ overlap.conj().T
        context = make_context(data, row, model)
        spectral_phase = (
            2
            * np.pi
            * (np.max(models.energies[model]) - np.min(models.energies[model]))
            * segments.duration[row]
        )
        phase_rtol = controls.phase_budget / max(1.0, spectral_phase)
        scale = min(1.0, phase_rtol / controls.rtol)
        row_controls = Tolerances(
            controls.rtol * scale,
            controls.atol * scale,
            min(controls.max_step, _sample_ceiling(segments, row)),
        )
        result = integrate(
            rhs,
            context,
            state.ravel(),
            float(segments.duration[row]),
            row_controls,
            min(0.1, row_controls.max_step),
        )
        states[index] = result.reshape(state.shape)
