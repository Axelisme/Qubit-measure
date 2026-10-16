"""Batched arbitrary-envelope Lindblad fallback in each secular eigenbasis."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .._envelope import sample_value
from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularModel
from ._dop853 import Tolerances, integrate, tolerances


class _Models(NamedTuple):
    energies: NDArray[np.float64]
    rates: NDArray[np.float64]
    decay: NDArray[np.float64]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    basis: NDArray[np.complex128]
    ports: NDArray[np.int64]


@dataclass(frozen=True)
class _Prepared:
    models: _Models

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.models)


class _Segments(NamedTuple):
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


class _Context(NamedTuple):
    models: _Models
    segments: _Segments
    row: int
    model: int


def _pack_models(models: list[SecularModel]) -> _Models:
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
    packed = _Models(energies, rates, decay, x, y, basis, ports)
    for array in packed:
        array.setflags(write=False)
    return packed


def _pack_segments(table: SegmentTable) -> _Segments:
    offsets = np.zeros(len(table.envelopes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(envelope.samples) for envelope in table.envelopes])
    if table.envelopes:
        times = np.concatenate([envelope.sample_ns for envelope in table.envelopes])
        samples = np.concatenate([envelope.samples for envelope in table.envelopes])
    else:
        times = np.empty(0)
        samples = np.empty(0, dtype=np.complex128)
    return _Segments(
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
    )


@njit(nogil=True, cache=True)
def _hamiltonian(time, context):
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


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    size = context.models.energies.shape[1]
    rho = state.reshape(size, size)
    h = _hamiltonian(time, context)
    derivative = -2j * np.pi * (h @ rho - rho @ h)
    derivative -= context.models.decay[context.model] * rho
    for target in range(size):
        for source in range(size):
            derivative[target, target] += (
                context.models.rates[context.model, target, source]
                * rho[source, source]
            )
    return derivative.ravel()


# Passing a compiled RHS into the generic stepper prevents disk caching here.
@njit(nogil=True)
def _apply_batch(
    models: _Models,
    segments: _Segments,
    rows: NDArray[np.int64],
    key_index: NDArray[np.int64],
    states: NDArray[np.complex128],
    controls: Tolerances,
):
    last_step = controls.max_step
    for index in range(len(rows)):
        row = rows[index]
        model = key_index[index]
        state = states[index].copy()
        if index > 0:
            previous_row = rows[index - 1]
            boundary = segments.start[previous_row] + segments.duration[previous_row]
            chained = (
                segments.timeline[row] == segments.timeline[previous_row]
                and abs(segments.start[row] - boundary) <= 1e-9
            )
            if chained:
                previous_model = key_index[index - 1]
                state = states[index - 1].copy()
                if previous_model != model:
                    overlap = (
                        models.basis[model].conj().T @ models.basis[previous_model]
                    )
                    state = overlap @ state @ overlap.conj().T
            else:
                last_step = controls.max_step
        context = _Context(models, segments, row, model)
        result, last_step = integrate(
            _rhs,
            context,
            state.ravel(),
            float(segments.duration[row]),
            controls,
            last_step,
        )
        states[index] = result.reshape(state.shape)


class DOP853Kernel:
    """C5 fallback for all tone counts and complex sampled envelopes.

    The kernel is stateless. Keys encode flux level; the engine must scope its
    cache by ModelSource identity. prepare owns read-only copies of secular
    operators, never a density. All row iteration/integration releases the GIL
    in single-threaded numba code. Input ports must exist in each row's model.
    """

    name = "dop853"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Accept every valid row; no fast-path approximation guard applies."""
        del table, source
        tolerances(budget)
        return np.ones(len(rows), dtype=np.bool_)

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Return fixed-width level keys, excluding every apply-only parameter."""
        keys = np.empty(len(rows), dtype=[("level", "<i8", (table.level.shape[1],))])
        keys["level"] = table.level[rows]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate RHS work from duration and tone count, with no timing probe."""
        return np.asarray(
            (1.0 + table.tones[rows]) * np.maximum(1.0, table.duration_ns[rows] / 0.1),
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
        """Copy each key's secular operators; table/rows are C5 representatives.

        Raises ValueError for an invalid budget or inconsistent model dimensions.
        ModelSource lookup errors propagate, without reading initial_density.
        """
        del table, rows
        tolerances(budget)
        models = [
            source.model(tuple(int(value) for value in key["level"])) for key in keys
        ]
        return _Prepared(_pack_models(models))

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Advance independent rows or consecutive same-timeline intervals.

        Rows chain when their time supports touch within 1e-9 ns. A chain's
        first states slot supplies its input; each slot receives that interval's
        endpoint in its own model eigenbasis. Later input slots are ignored.
        A basis overlap is applied on model changes within a chain. Across
        calls, the engine supplies the next input in the next model's basis.
        The integrator stops at boundaries and retains its last accepted step.

        Raises ValueError for incompatible prepared data, state/index shapes or
        nonexistent ports. RuntimeError signals numerical failure; earlier rows
        may already have been advanced. No state normalization/clipping occurs.
        """
        controls = tolerances(budget)
        if not isinstance(prepared, _Prepared):
            raise ValueError("DOP853 requires its own prepared data")
        if key_index.shape != rows.shape or states.shape[0] != len(rows):
            raise ValueError("DOP853 state/index shape mismatch")
        if len(rows) == 0:
            return
        models = prepared.models
        if states.shape[1:] != (models.energies.shape[1],) * 2:
            raise ValueError("DOP853 state shape mismatch")
        if np.any(key_index < 0) or np.any(key_index >= len(models.ports)):
            raise ValueError("DOP853 prepared index out of range")
        playing = np.arange(table.tone_port.shape[1])[None, :] < table.tones[rows, None]
        if np.any(playing & (table.tone_port[rows] >= models.ports[key_index, None])):
            raise ValueError("DOP853 tone port absent from model")
        _apply_batch(models, _pack_segments(table), rows, key_index, states, controls)
