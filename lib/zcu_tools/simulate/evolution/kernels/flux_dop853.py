"""Continuous-flux Lindblad integration with quantized secular baths."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..secular import FluxExpansion, ModelSource
from ._dop853 import tolerances
from ._dop853_chain import run_chain
from ._dop853_data import (
    Context,
    Models,
    Segments,
    hamiltonian,
    pack_models,
)
from ._dop853_intervals import playback_time
from ._flux_expansion import TERM_KINDS, PackedExpansion, flux_hamiltonian
from ._workload_facts import packed_segments


@dataclass(frozen=True)
class FluxDOP853Plan:
    """Global-row chain IDs fixed by dispatch for this evolve."""

    chain: NDArray[np.int64]


class _Waveform(NamedTuple):
    offsets: NDArray[np.int64]
    times: NDArray[np.float64]
    flux: NDArray[np.float64]


class _FluxData(NamedTuple):
    models: Models
    segments: Segments
    expansion: PackedExpansion
    waveform: _Waveform
    energy_shifts: NDArray[np.float64]


class _FluxContext(NamedTuple):
    drive: Context
    expansion: PackedExpansion
    waveform: _Waveform
    energy_shifts: NDArray[np.float64]


@dataclass(frozen=True)
class _Prepared:
    source: ModelSource
    expansion: PackedExpansion
    key_count: int

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.expansion)


def _pack_expansion(expansion: FluxExpansion) -> PackedExpansion:
    packed = PackedExpansion(
        expansion.static,
        np.array(
            [term.matrix for term in expansion.terms], dtype=np.complex128
        ).reshape((-1, *expansion.static.shape)),
        np.array([term.loop for term in expansion.terms], dtype=np.int64),
        np.array(
            [TERM_KINDS.index(term.kind) for term in expansion.terms], dtype=np.int64
        ),
    )
    for array in packed:
        array.setflags(write=False)
    return packed


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    drive, expansion, waveform, energy_shifts = context
    models, segments, row, model, interval = drive
    size = models.energies.shape[1]
    rho = state.reshape(size, size)
    point = segments.timeline[row]
    begin, end = waveform.offsets[point : point + 2]
    absolute = playback_time(
        time, interval.flux_start, interval.flux_end, interval.duration
    )
    fixed_h = flux_hamiltonian(
        expansion, waveform.times[begin:end], waveform.flux[begin:end], absolute
    )
    basis = models.basis[model]
    h = hamiltonian(time, drive)
    for i in range(size):
        h[i, i] -= models.energies[model, i] - energy_shifts[model, i]
    h += basis.conj().T @ fixed_h @ basis
    derivative = -2j * np.pi * (h @ rho - rho @ h)
    derivative -= models.decay[model] * rho
    for target in range(size):
        for source in range(size):
            derivative[target, target] += (
                models.rates[model, target, source] * rho[source, source]
            )
    return derivative.ravel()


@njit(nogil=True, cache=True)
def _context(data, row, model, interval):
    models, segments, expansion, waveform, energy_shifts = data
    return _FluxContext(
        Context(models, segments, row, model, interval),
        expansion,
        waveform,
        energy_shifts,
    )


@njit(nogil=True, cache=True)
def _run(data, row_models, states, controls):
    # Keep RHS/context dispatchers inside the compiled entry point so its disk
    # cache key contains only data types, not process-specific callback IDs.
    run_chain(_rhs, _context, data, row_models, states, controls)


class FluxDOP853Kernel:
    """C5 continuous H(flux(t)) with a constant quantized bath per row.

    Keys are source-scoped, independent of level, drive or waveform. The engine
    namespaces its cache by the DC model identity (which identifies the whole
    source and expansion). prepare retains the immutable expansion and source's
    thread-safe model cache, never density. apply packs only unique used bath
    levels; model references/workspaces are outside Prepared.nbytes, as for the
    other kernels. Operators and states use each row's bath eigenbasis. Chained
    rows switch basis at boundaries. The shared single-threaded, nogil DOP853
    core stops at each active envelope/flux knot and restarts a deterministic
    initial step. Each interval limits only its own steps. Source-coordinate
    endpoints preserve event sides without imposing their gap on the full row.
    No flux/DAC midpoint freezing occurs.
    """

    name = "flux_dop853"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Accept driven rows with samples; engine owns flux-change routing."""
        source, budget = context.source, context.budget
        tolerances(budget)
        del source
        available = np.full(len(rows), table.flux_waveform is not None)
        if table.flux_waveform_available is not None:
            available &= table.flux_waveform_available[table.timeline[rows]]
        return (table.tones[rows] > 0) & available

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Return one source-scoped key, excluding every apply-only parameter."""
        del context
        keys = np.zeros(len(rows), dtype=[("source", "i8"), ("shift", "S64")])
        keys["shift"] = table.shift_content[rows]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate integration and basis-multiplication work without timing."""
        return np.asarray(
            (2 + table.tones[rows]) * np.maximum(1, table.duration_ns[rows] / 0.1),
            dtype=np.float64,
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Retain source and exact fixed-coordinate expansion, never density.

        Missing expansion or out-of-range loop indices raises ValueError.
        """
        source, budget = context.source, context.budget
        del rows
        tolerances(budget)
        expansion = source.flux_expansion()
        if any(term.loop >= table.level.shape[1] for term in expansion.terms):
            raise ValueError("flux expansion loop exceeds waveform width")
        return _Prepared(source, _pack_expansion(expansion), len(keys))

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
        """Advance independent rows or touching same-timeline interval chains.

        The first slot is a chain's input; each slot receives its own endpoint
        in that row's bath eigenbasis. Later input slots in a chain are ignored.
        Waveforms are indexed by table.timeline and interpolated at absolute ns;
        endpoint samples are held outside their span. Bath levels are table.level.
        Stops at every row boundary and active sample knot, resetting the initial
        step per interval and preserving source-coordinate endpoints. Raises
        ValueError for absent samples, incompatible payload/shapes/ports/indices;
        RuntimeError on integration failure, without clipping or renormalizing.
        Earlier rows may already be advanced on numerical failure.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        plan = context.plan
        if plan is None:
            raise ValueError("flux DOP853 requires a row plan")
        if not isinstance(plan, FluxDOP853Plan):
            raise TypeError("flux DOP853 requires its own row plan")
        budget = context.budget
        controls = tolerances(budget)
        if not isinstance(prepared, _Prepared):
            raise ValueError("flux DOP853 requires its own prepared data")
        waveform = table.flux_waveform
        if waveform is None:
            raise ValueError("continuous flux requires a flux waveform")
        if table.flux_waveform_available is not None and not np.all(
            table.flux_waveform_available[table.timeline[rows]]
        ):
            raise ValueError("continuous flux row has no flux waveform")
        if key_index.shape != rows.shape or states.shape != (
            len(rows),
            *prepared.expansion.static.shape,
        ):
            raise ValueError("flux DOP853 state/index shape mismatch")
        if np.any(key_index < 0) or np.any(key_index >= prepared.key_count):
            raise ValueError("flux DOP853 prepared index out of range")
        if not len(rows):
            return np.zeros(len(rows), dtype=np.bool_)
        from ._dyson_model import model_keys

        _, first, inverse = np.unique(
            model_keys(table, rows), return_index=True, return_inverse=True
        )
        selected_models = [
            table.model(prepared.source, int(rows[index])) for index in first
        ]
        models = pack_models(selected_models)
        shifts = np.array(
            [
                model.energies_ghz - prepared.source.model(model.level).energies_ghz
                for model in selected_models
            ]
        )
        if models.energies.shape[1] != states.shape[1]:
            raise ValueError("flux DOP853 expansion/model shape mismatch")
        playing = np.arange(table.tone_port.shape[1])[None, :] < table.tones[rows, None]
        if np.any(playing & (table.tone_port[rows] >= models.ports[inverse, None])):
            raise ValueError("flux DOP853 tone port absent from model")
        segments = packed_segments(table, context.work_plan, include_flux=True)
        data = _FluxData(
            models,
            segments,
            prepared.expansion,
            _Waveform(waveform.offsets, waveform.time_ns, waveform.flux),
            shifts,
        )
        _run(data, (rows, inverse, plan.chain), states, controls)
        return np.zeros(len(rows), dtype=np.bool_)
