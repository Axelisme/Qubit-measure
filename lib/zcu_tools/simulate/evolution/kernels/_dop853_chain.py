"""Shared single-threaded interval-chain control for both DOP853 kernels."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._dop853 import Tolerances, integrate
from ._dop853_data import Models, Segments


# Compiled callables specialize this runner for each kernel's context and RHS.
@njit(nogil=True)
def run_chain(
    rhs,
    make_context,
    data,
    models: Models,
    segments: Segments,
    rows: NDArray[np.int64],
    model_index: NDArray[np.int64],
    states: NDArray[np.complex128],
    controls: Tolerances,
):
    """Advance touching same-timeline rows and store every interval endpoint.

    rhs(time, flat_state, context) and make_context(data, row, model) are
    compiled kernel-owned callables. Models and segments share their packed
    layout with data. rows/model_index align with states (rows, M, M).
    Each chain reads only its first input slot; subsequent rows use the previous
    endpoint, changing basis on model changes. Disjoint rows reset the adaptive
    step to controls.max_step and read their own input. Contiguity uses 1e-9 ns.
    Integration failures propagate; earlier endpoints may already be stored.
    """
    last_step = controls.max_step
    for index in range(len(rows)):
        row, model = rows[index], model_index[index]
        state = states[index].copy()
        if index > 0:
            previous = rows[index - 1]
            boundary = segments.start[previous] + segments.duration[previous]
            if (
                segments.timeline[row] == segments.timeline[previous]
                and abs(segments.start[row] - boundary) <= 1e-9
            ):
                previous_model = model_index[index - 1]
                state = states[index - 1].copy()
                if previous_model != model:
                    overlap = (
                        models.basis[model].conj().T @ models.basis[previous_model]
                    )
                    state = overlap @ state @ overlap.conj().T
            else:
                last_step = controls.max_step
        context = make_context(data, row, model)
        result, last_step = integrate(
            rhs,
            context,
            state.ravel(),
            float(segments.duration[row]),
            controls,
            last_step,
        )
        states[index] = result.reshape(state.shape)
