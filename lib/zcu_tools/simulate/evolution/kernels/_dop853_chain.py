"""Shared single-threaded interval-chain control for both DOP853 kernels."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..state import transform_density
from ._dop853 import (
    DP54_TOLERANCE_DIVISOR,
    ROW_INITIAL_STEP_NS,
    IntervalStep,
    Tolerances,
    integrate_workspace,
    make_workspace,
)
from ._dop853_intervals import row_intervals


class ChainRows(NamedTuple):
    """Rows and solver choices passed to one chronological CPU apply.

    rows contains global row IDs. model_index supplies one packed model index
    per row. chain is indexed by global row ID and identifies touching rows.
    dp54 is row-aligned; True selects DP5(4) and False selects DOP853.
    """

    rows: NDArray[np.int64]
    model_index: NDArray[np.int64]
    chain: NDArray[np.int64]
    dp54: NDArray[np.bool_]


# Inline callback-taking control into the kernel's cached entry point, so no
# process-specific dispatcher pointer becomes a dynamic global in its cache.
@njit(nogil=True, cache=True, inline="always")
def run_chain(
    rhs,
    make_context,
    data,
    chain_rows: ChainRows,
    states: NDArray[np.complex128],
    controls: Tolerances,
):
    """Advance touching same-timeline rows and store every interval endpoint.

    rhs(time, flat_state, context) and make_context(data, row, model, interval, previous)
    are compiled kernel-owned callables. RHS time is interval-local; context
    retains row, absolute-flux and pulse-local playback clocks. data supplies
    packed models and segments. Each context has a drive field containing its
    interval clocks and call-owned RHS buffers reused across that row's knots.
    Passing previous=None allocates those buffers; a previous context reuses them.
    Step buffers belong to this invocation and are reused across rows.
    chain_rows supplies the rows and model indices for states (rows, M, M).
    Each chain reads only its first input slot; subsequent rows use the previous
    endpoint, changing basis on model changes. Disjoint rows read their own
    input. Each active sample knot stops integration and resets a deterministic
    0.1 ns seed. A narrow interval limits only its own steps, not the whole row.
    Source-coordinate endpoints preserve event sides and nextafter bridges.
    Constant envelopes and ordinary DOP853's unpacked flux add no stops.
    Chain membership comes from dispatch, not apply-call adjacency.
    Both tableaux use the same frame, intervals and workspace.
    A finite controls.phase_budget caps
    local rtol by that allowance divided by bare spectral phase over this row,
    scaling atol by the same factor. Lab-frame ordinary density propagation
    supplies the allowance; interaction and continuous-flux callers retain
    their original controls. DP5(4) divides both local errors by three after
    the phase cap. These are calibrated margins, not global bounds.
    Integration failures propagate; earlier endpoints may already be stored.
    """
    models, segments = data.models, data.segments
    rows, model_index, chain, dp54 = chain_rows
    if not len(rows):
        return
    workspace = make_workspace(states[0].ravel())
    for index in range(len(rows)):
        row, model = rows[index], model_index[index]
        state = states[index].copy()
        if index > 0:
            previous = rows[index - 1]
            if chain[row] >= 0 and chain[previous] == chain[row]:
                previous_model = model_index[index - 1]
                state = states[index - 1].copy()
                if previous_model != model:
                    state = transform_density(
                        state, models.basis[previous_model], models.basis[model]
                    )
        spectral_phase = (
            2
            * np.pi
            * (np.max(models.energies[model]) - np.min(models.energies[model]))
            * segments.duration[row]
        )
        phase_rtol = controls.phase_budget / max(1.0, spectral_phase)
        scale = min(1.0, phase_rtol / controls.rtol)
        use_dp54 = dp54[index]
        if use_dp54:
            scale /= DP54_TOLERANCE_DIVISOR
        row_controls = Tolerances(
            controls.rtol * scale,
            controls.atol * scale,
            controls.max_step,
            np.inf,
            0,
        )
        result = state.ravel()
        intervals = row_intervals(segments, row)
        if not len(intervals):
            states[index] = state
            continue
        context = make_context(data, row, model, intervals[0], None)
        for interval in intervals:
            context = make_context(data, row, model, interval, context)
            workspace.current[:] = result
            result = integrate_workspace(
                rhs,
                context,
                IntervalStep(
                    interval.duration,
                    min(ROW_INITIAL_STEP_NS, interval.duration, row_controls.max_step),
                    use_dp54,
                ),
                row_controls,
                workspace,
            )
        states[index] = result.reshape(state.shape)
