"""Batched arbitrary-envelope Lindblad fallback in each secular eigenbasis."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ._dop853 import tolerances
from ._dop853_chain import run_chain
from ._dop853_data import (
    Context,
    Models,
    Segments,
    drive_amplitude,
    fill_hamiltonian,
    pack_models,
)
from ._dop853_intervals import playback_time
from ._workload_facts import packed_segments


@dataclass(frozen=True)
class DOP853Plan:
    """Global-row frames and chain IDs fixed by dispatch for this evolve."""

    frame: NDArray[np.str_]
    chain: NDArray[np.int64]


class _Data(NamedTuple):
    models: Models
    segments: Segments


class _LabContext(NamedTuple):
    # Mutable scratch belongs to one interval, not to the model-only preparation.
    drive: Context
    hamiltonian: NDArray[np.complex128]
    product: NDArray[np.complex128]
    derivative: NDArray[np.complex128]


@dataclass(frozen=True)
class DOP853Prepared:
    models: Models

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.models)


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    drive, h, product, derivative = context
    size = drive.models.energies.shape[1]
    rho = state.reshape(size, size)
    fill_hamiltonian(time, drive, h)
    # A density and every real-weighted Runge--Kutta stage are Hermitian.
    # Thus rho @ h = (h @ rho).H; one dense product supplies both terms.
    np.dot(h, rho, product)
    for target in range(size):
        for source in range(size):
            derivative[target, source] = (
                -2j
                * np.pi
                * (product[target, source] - product[source, target].conjugate())
                - drive.models.decay[drive.model, target, source] * rho[target, source]
            )
        for source in range(size):
            derivative[target, target] += (
                drive.models.rates[drive.model, target, source] * rho[source, source]
            )
    return derivative.ravel()


@njit(nogil=True, cache=True)
def _context(data, row, model, interval):
    models, segments = data
    return Context(models, segments, row, model, interval)


@njit(nogil=True, cache=True)
def _lab_context(data, row, model, interval):
    size = data.models.energies.shape[1]
    h = np.empty((size, size), dtype=np.complex128)
    return _LabContext(
        _context(data, row, model, interval), h, np.empty_like(h), np.empty_like(h)
    )


@njit(nogil=True, cache=True)
def _run(data, row_models, states, controls):
    # Cache the linked solver behind a callback-free signature. Passing compiled
    # callables from Python gives Numba process-specific specialization keys.
    run_chain(_rhs, _lab_context, data, row_models, states, controls)


@njit(nogil=True, cache=True, inline="always")
def _tone_hamiltonian(time, context, tone):
    models, segments, row, model, _ = context
    z = drive_amplitude(time, context, tone)
    port = segments.port[row, tone]
    return z.real * models.x[model, port] + z.imag * models.y[model, port]


@njit(nogil=True, cache=True)
def _interaction_rhs(time, state, context):
    models, segments, row, model, interval = context
    size = models.energies.shape[1]
    rho = state.reshape(size, size)
    # Build only the drives: subtracting H0 from a full Hamiltonian would
    # lose small diagonal drive terms to cancellation at weak gains. The first
    # tone keeps the original single-tone arithmetic without an extra M² buffer.
    if segments.tones[row]:
        h = _tone_hamiltonian(time, context, 0)
        for tone in range(1, segments.tones[row]):
            h += _tone_hamiltonian(time, context, tone)
    else:
        h = np.zeros((size, size), dtype=np.complex128)
    row_time = playback_time(
        time, interval.row_start, interval.row_end, interval.duration
    )
    phases = np.exp(2j * np.pi * models.energies[model] * row_time)
    for i in range(size):
        for j in range(size):
            h[i, j] *= phases[i] * phases[j].conjugate()
    product = h @ rho
    derivative = np.empty_like(rho)
    for target in range(size):
        for source in range(size):
            derivative[target, source] = (
                -2j
                * np.pi
                * (product[target, source] - product[source, target].conjugate())
                - models.decay[model, target, source] * rho[target, source]
            )
        for source in range(size):
            derivative[target, target] += (
                models.rates[model, target, source] * rho[source, source]
            )
    return derivative.ravel()


@njit(nogil=True, cache=True)
def _run_interaction(data, row_models, states, controls):
    # Only independent rows enter: each starts at sigma(0)=rho(0), with its own
    # controller and seed. The callback-free entry point has a stable disk key.
    run_chain(_interaction_rhs, _context, data, row_models, states, controls)
    rows, model_index, _ = row_models
    for index in range(len(rows)):
        row, model = rows[index], model_index[index]
        phases = np.exp(
            -2j * np.pi * data.models.energies[model] * data.segments.duration[row]
        )
        for i in range(len(phases)):
            for j in range(len(phases)):
                states[index, i, j] *= phases[i] * phases[j].conjugate()


class DOP853Kernel:
    """C5 fallback for all tone counts and complex sampled envelopes.

    The kernel is stateless. Keys encode flux level; the engine must scope its
    cache by ModelSource identity. prepare owns read-only copies of secular
    operators, never a density. All row iteration/integration releases the GIL
    in single-threaded numba code. Input ports must exist in each row's model.
    """

    name = "dop853"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Accept every valid row; no fast-path approximation guard applies."""
        source, budget = context.source, context.budget
        del table, source
        tolerances(budget)
        return np.ones(len(rows), dtype=np.bool_)

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Return fixed-width level keys, excluding every apply-only parameter."""
        del context
        keys = np.empty(
            len(rows),
            dtype=[("level", "<i8", (table.level.shape[1],)), ("shift", "S64")],
        )
        keys["level"] = table.level[rows]
        keys["shift"] = table.shift_content[rows]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate RHS work from duration and tone count, with no timing probe."""
        return np.asarray(
            0.8
            * (1.0 + table.tones[rows])
            * np.maximum(1.0, table.duration_ns[rows] / 0.1),
            dtype=np.float64,
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Copy each key's secular operators; table/rows are C5 representatives.

        Raises ValueError for an invalid budget or inconsistent model dimensions.
        ModelSource lookup errors propagate, without reading initial_density.
        """
        source, budget = context.source, context.budget
        del table, rows
        tolerances(budget)
        models = [
            source.shifted_model(
                tuple(int(v) for v in key["level"]), bytes(key["shift"])
            )
            for key in keys
        ]
        return prepare_models(models)

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline guard preparation handoff; the engine calls guard normally."""
        source, budget = context.source, context.budget
        del table, rows, source, budget

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """Decline input binding; apply uses the model-only prepared payload."""
        budget = context.budget
        del prepared, key_index, table, rows, source_initial, budget

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Advance independent rows or consecutive same-timeline intervals.

        Rows chain when their time supports touch within 1e-9 ns. A chain's
        first states slot supplies its input; each slot receives that interval's
        endpoint in its own model eigenbasis. Later input slots are ignored.
        A basis overlap is applied on model changes within a chain. Across
        calls, the engine supplies the next input in the next model's basis.
        Each active envelope knot stops integration and restarts a deterministic
        0.1 ns step seed. A narrow interval limits only its own steps.
        Pulse-local sample endpoints and row-local carrier phases are retained.
        Samples outside the row's envelope window and all flux waveform samples
        are ignored. Constant envelopes add no internal stops.
        Independent rows use sigma(t)=exp(+i*2*pi*H0*t) rho(t)
        exp(-i*2*pi*H0*t), integrating the full transformed drive and unchanged
        secular dissipator, then return to model eigenbasis. Each row keeps its
        own adaptive step controller and original normalized RMS error norm,
        independent of the other rows in the call. There is no gain grouping or
        pooled RMS. Each interval uses the complete row's tolerances; this is
        local control, not a new end-to-end error bound. No adaptive history
        carries between intervals, rows or calls, so row order and worker/apply partition
        do not change an independent row's numerical path, including mixed
        IP/lab calls. Every row in a touching chain retains the lab-frame
        runner, with local tolerances capped by budget divided by accumulated
        bare spectral phase. This prevents local GHz integration error from
        becoming a biased waiting frequency. Dispatch fixes
        frames and chain membership from the complete timeline. Apply executes
        those decisions even when a prepared-payload boundary splits a chain.
        No RWA, drive truncation, splitting or envelope changes are made.

        Raises ValueError for incompatible prepared data, state/index shapes or
        nonexistent ports. RuntimeError signals numerical failure; earlier rows
        may already have been advanced. No state normalization/clipping occurs.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        plan = context.plan
        if plan is None:
            raise ValueError("DOP853 requires a row plan")
        if not isinstance(plan, DOP853Plan):
            raise TypeError("DOP853 requires its own row plan")
        budget = context.budget
        tolerances(budget)
        if not isinstance(prepared, DOP853Prepared):
            raise ValueError("DOP853 requires its own prepared data")
        if key_index.shape != rows.shape or states.shape[0] != len(rows):
            raise ValueError("DOP853 state/index shape mismatch")
        if len(rows) == 0:
            return np.zeros(len(rows), dtype=np.bool_)
        models = prepared.models
        if states.shape[1:] != (models.energies.shape[1],) * 2:
            raise ValueError("DOP853 state shape mismatch")
        if np.any(key_index < 0) or np.any(key_index >= len(models.ports)):
            raise ValueError("DOP853 prepared index out of range")
        playing = np.arange(table.tone_port.shape[1])[None, :] < table.tones[rows, None]
        if np.any(playing & (table.tone_port[rows] >= models.ports[key_index, None])):
            raise ValueError("DOP853 tone port absent from model")
        advance_endpoints(
            prepared.models,
            packed_segments(table, context.work_plan),
            (rows, key_index),
            states,
            budget,
            plan,
        )
        return np.zeros(len(rows), dtype=np.bool_)


def prepare_models(models: list[SecularModel]) -> DOP853Prepared:
    """Copy secular operators for direct endpoints, without kernel selection."""
    return DOP853Prepared(pack_models(models))


def advance_endpoints(
    models: Models,
    segments: Segments,
    row_models: tuple[NDArray[np.int64], NDArray[np.int64]],
    states: NDArray[np.complex128],
    budget: float,
    plan: DOP853Plan,
) -> None:
    """Integrate endpoints using explicit global-row frames and chain IDs.

    Inputs are validated packed secular operators and segments. states has one
    slot per row, in its model eigenbasis. Each chain reads its first slot and
    writes every endpoint; independent rows read their own slots. Numerical
    failure can leave earlier endpoints written. No frame selection occurs here.
    """
    controls = tolerances(budget)
    rows, key_index = row_models
    if not len(rows):
        return
    interaction = plan.frame[rows] == "interaction"
    boundaries = np.r_[
        0, np.flatnonzero(interaction[1:] != interaction[:-1]) + 1, len(rows)
    ]
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        data = _Data(models, segments)
        chain_rows = (rows[begin:end], key_index[begin:end], plan.chain)
        if interaction[begin]:
            _run_interaction(data, chain_rows, states[begin:end], controls)
        else:
            _run(
                data,
                chain_rows,
                states[begin:end],
                controls._replace(phase_budget=budget),
            )
