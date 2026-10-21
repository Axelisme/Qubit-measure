"""Shared single-threaded interval-chain control for both DOP853 kernels."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._dop853 import Tolerances, integrate
from ._dop853_intervals import row_intervals


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

    rhs(time, flat_state, context) and make_context(data, row, model, interval)
    are compiled kernel-owned callables. RHS time is interval-local; context
    retains row, absolute-flux and pulse-local playback clocks. data supplies
    packed models and segments.
    row_models is (rows, model_index, global_chain_ids). The first two vectors
    align with states (rows, M, M); chain IDs are indexed by global table row.
    Each chain reads only its first input slot; subsequent rows use the previous
    endpoint, changing basis on model changes. Disjoint rows read their own
    input. Each active sample knot stops integration and resets a deterministic
    0.1 ns seed. A narrow interval limits only its own steps, not the whole row.
    Source-coordinate endpoints preserve event sides and nextafter bridges.
    Constant envelopes and ordinary DOP853's unpacked flux add no stops.
    Chain membership comes from dispatch, not apply-call adjacency.
    A finite controls.phase_budget caps
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
            controls.max_step,
            np.inf,
            0,
        )
        result = state.ravel()
        for interval in row_intervals(segments, row):
            context = make_context(data, row, model, interval)
            result = integrate(
                rhs,
                context,
                result,
                interval.duration,
                row_controls,
                min(0.1, interval.duration, row_controls.max_step),
            )
        states[index] = result.reshape(state.shape)
