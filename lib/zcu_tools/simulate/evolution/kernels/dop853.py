"""Batched arbitrary-envelope Lindblad fallback in each secular eigenbasis."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import SecularModel
from ._dop853 import tolerances
from ._dop853_chain import run_chain
from ._dop853_data import (
    Context,
    Models,
    Segments,
    fill_hamiltonian,
    pack_models,
)
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
    # Mutable scratch belongs to one row, not to the model-only preparation.
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
def _context(data, row, model):
    models, segments = data
    return Context(models, segments, row, model)


@njit(nogil=True, cache=True)
def _lab_context(data, row, model):
    size = data.models.energies.shape[1]
    h = np.empty((size, size), dtype=np.complex128)
    return _LabContext(
        _context(data, row, model), h, np.empty_like(h), np.empty_like(h)
    )


@njit(nogil=True, cache=True)
def _run(data, row_models, states, controls):
    # Cache the linked solver behind a callback-free signature. Passing compiled
    # callables from Python gives Numba process-specific specialization keys.
    run_chain(_rhs, _lab_context, data, row_models, states, controls)


@njit(nogil=True, cache=True)
def _interaction_rhs(time, state, context):
    models, segments, row, model = context
    size = models.energies.shape[1]
    rho = state.reshape(size, size)
    envelope = segments.envelope[row, 0]
    begin, end = segments.sample_offsets[envelope : envelope + 2]
    amplitude = sample_value(
        time + segments.offset[row, 0],
        segments.sample_time[begin:end],
        segments.samples[begin:end],
    )
    phase = segments.theta[row, 0] + 2 * np.pi * segments.frequency[row, 0] * time
    z = segments.gain[row, 0] * amplitude * np.exp(1j * phase)
    port = segments.port[row, 0]
    # Build the drive directly: subtracting H0 from a full Hamiltonian would
    # lose small diagonal drive terms to cancellation at weak gains.
    h = z.real * models.x[model, port] + z.imag * models.y[model, port]
    phases = np.exp(2j * np.pi * models.energies[model] * time)
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
        keys = np.empty(len(rows), dtype=[("level", "<i8", (table.level.shape[1],))])
        keys["level"] = table.level[rows]
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
            source.model(tuple(int(value) for value in key["level"])) for key in keys
        ]
        return prepare_models(models)

    def admit_prepared(
        self,
        prepared: Prepared,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> None:
        """Decline cached-domain admission; use the ordinary guard build path."""
        del prepared, table, rows, context

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
        Each row starts from a deterministic 0.1 ns step seed and adapts up to
        the smallest spacing between envelope knots inside the row's local
        envelope window. Samples outside that window and all flux waveform
        samples are ignored. Constant envelopes have no sampling ceiling.
        Independent single-tone rows use sigma(t)=exp(+i*2*pi*H0*t) rho(t)
        exp(-i*2*pi*H0*t), integrating the full transformed drive and unchanged
        secular dissipator, then return to model eigenbasis. Each row keeps its
        own adaptive step controller and original normalized RMS error norm,
        independent of the other rows in the call. There is no gain grouping or
        pooled RMS. Tolerances, seed and active-knot ceiling are unchanged. This
        is local control, not a new end-to-end error bound. No adaptive history
        carries between rows or calls, so row order and worker/apply partition
        do not change an independent row's numerical path, including mixed
        IP/lab calls. Zero-gain and non-single-tone rows, and every row in a
        touching chain, retain the unchanged lab-frame runner. Dispatch fixes
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
        runner = _run_interaction if interaction[begin] else _run
        runner(
            _Data(models, segments),
            (rows[begin:end], key_index[begin:end], plan.chain),
            states[begin:end],
            controls,
        )
