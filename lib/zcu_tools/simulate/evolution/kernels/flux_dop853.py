"""Continuous-flux Lindblad integration with quantized secular baths."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import Prepared, SegmentTable
from ..secular import FluxExpansion, ModelSource
from ._dop853 import tolerances
from ._dop853_chain import run_chain
from ._dop853_data import (
    Context,
    Models,
    Segments,
    hamiltonian,
    pack_models,
    pack_segments,
)


class _Expansion(NamedTuple):
    static: NDArray[np.complex128]
    matrices: NDArray[np.complex128]
    loops: NDArray[np.int64]
    kinds: NDArray[np.int64]


class _Waveform(NamedTuple):
    offsets: NDArray[np.int64]
    times: NDArray[np.float64]
    flux: NDArray[np.float64]


class _FluxData(NamedTuple):
    models: Models
    segments: Segments
    expansion: _Expansion
    waveform: _Waveform


class _FluxContext(NamedTuple):
    drive: Context
    expansion: _Expansion
    waveform: _Waveform


@dataclass(frozen=True)
class _Prepared:
    source: ModelSource
    expansion: _Expansion

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.expansion)


def _pack_expansion(expansion: FluxExpansion) -> _Expansion:
    kinds = ("cos_2pi", "sin_2pi", "cos_pi", "sin_pi")
    packed = _Expansion(
        expansion.static,
        np.array(
            [term.matrix for term in expansion.terms], dtype=np.complex128
        ).reshape((-1, *expansion.static.shape)),
        np.array([term.loop for term in expansion.terms], dtype=np.int64),
        np.array([kinds.index(term.kind) for term in expansion.terms], dtype=np.int64),
    )
    for array in packed:
        array.setflags(write=False)
    return packed


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    drive, expansion, waveform = context
    models, segments, row, model = drive
    size = models.energies.shape[1]
    rho = state.reshape(size, size)
    point = segments.timeline[row]
    begin, end = waveform.offsets[point : point + 2]
    absolute = segments.start[row] + time
    fixed_h = expansion.static.copy()
    for term in range(len(expansion.loops)):
        loop = expansion.loops[term]
        flux = np.interp(
            absolute, waveform.times[begin:end], waveform.flux[begin:end, loop]
        )
        kind = expansion.kinds[term]
        angle = np.pi * flux * (2 if kind < 2 else 1)
        value = np.cos(angle) if kind % 2 == 0 else np.sin(angle)
        fixed_h += value * expansion.matrices[term]
    basis = models.basis[model]
    h = hamiltonian(time, drive)
    for i in range(size):
        h[i, i] -= models.energies[model, i]
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
def _context(data, row, model):
    models, segments, expansion, waveform = data
    return _FluxContext(Context(models, segments, row, model), expansion, waveform)


class FluxDOP853Kernel:
    """C5 continuous H(flux(t)) with a constant quantized bath per row.

    Keys are source-scoped, independent of level, drive or waveform. The engine
    namespaces its cache by the DC model identity (which identifies the whole
    source and expansion). prepare retains the immutable expansion and source's
    thread-safe model cache, never density. apply packs only unique used bath
    levels; model references/workspaces are outside Prepared.nbytes, as for the
    other kernels. Operators and states use each row's bath eigenbasis. Chained
    rows switch basis at boundaries and reuse the accepted step of the shared
    single-threaded, nogil DOP853 core. No flux/DAC midpoint freezing occurs.
    """

    name = "flux_dop853"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Accept driven rows with samples; engine owns flux-change routing."""
        tolerances(budget)
        del source
        available = np.full(len(rows), table.flux_waveform is not None)
        if table.flux_waveform_available is not None:
            available &= table.flux_waveform_available[table.timeline[rows]]
        return (table.tones[rows] > 0) & available

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Return one source-scoped key, excluding every apply-only parameter."""
        del table
        return np.zeros(len(rows), dtype=[("source", "i8")])

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
        source: ModelSource,
        budget: float,
    ) -> Prepared:
        """Retain source and exact fixed-coordinate expansion, never density.

        Missing expansion or out-of-range loop indices raises ValueError.
        """
        del keys, rows
        tolerances(budget)
        expansion = source.flux_expansion()
        if any(term.loop >= table.level.shape[1] for term in expansion.terms):
            raise ValueError("flux expansion loop exceeds waveform width")
        return _Prepared(source, _pack_expansion(expansion))

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> NDArray[np.bool_]:
        """Advance independent rows or touching same-timeline interval chains.

        The first slot is a chain's input; each slot receives its own endpoint
        in that row's bath eigenbasis. Later input slots in a chain are ignored.
        Waveforms are indexed by table.timeline and interpolated at absolute ns;
        endpoint samples are held outside their span. Bath levels are table.level.
        Stops at every row boundary, retaining the last accepted step. Raises
        ValueError for absent samples, incompatible payload/shapes/ports/indices;
        RuntimeError on integration failure, without clipping or renormalizing.
        Earlier rows may already be advanced on numerical failure.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
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
        if np.any(key_index != 0):
            raise ValueError("flux DOP853 prepared index out of range")
        if not len(rows):
            return np.zeros(len(rows), dtype=np.bool_)
        levels, inverse = np.unique(table.level[rows], axis=0, return_inverse=True)
        models = pack_models(
            [prepared.source.model(tuple(int(v) for v in level)) for level in levels]
        )
        if models.energies.shape[1] != states.shape[1]:
            raise ValueError("flux DOP853 expansion/model shape mismatch")
        playing = np.arange(table.tone_port.shape[1])[None, :] < table.tones[rows, None]
        if np.any(playing & (table.tone_port[rows] >= models.ports[inverse, None])):
            raise ValueError("flux DOP853 tone port absent from model")
        segments = pack_segments(table)
        data = _FluxData(
            models,
            segments,
            prepared.expansion,
            _Waveform(waveform.offsets, waveform.time_ns, waveform.flux),
        )
        run_chain(_rhs, _context, data, (rows, inverse), states, controls)
        return np.zeros(len(rows), dtype=np.bool_)
