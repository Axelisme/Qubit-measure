"""Fourier closed-system gates interleaved with compact secular dissipation."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, Prepared, SegmentTable
from ..secular import ModelSource
from . import gate_phase
from ._gate_endpoints import build_endpoints
from ._secular import coherence_decay, population_generator
from ._shaped import drive_bounds, envelope_bounds

_PHASE_COUNT = 16
_SUBSEGMENTS = 4
_MODES = np.fft.fftfreq(_PHASE_COUNT) * _PHASE_COUNT


@dataclass(frozen=True)
class _GatePrepared:
    coefficients: NDArray[np.complex128]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        return (
            self.coefficients.nbytes + self.populations.nbytes + self.coherences.nbytes
        )


def _check_budget(budget: float) -> None:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("gate cache budget must be finite and positive")


@njit(nogil=True, cache=True)
def _apply_gates(
    coefficients: NDArray[np.complex128],
    populations: NDArray[np.float64],
    coherences: NDArray[np.float64],
    key_index: NDArray[np.int64],
    phases: NDArray[np.float64],
    states: NDArray[np.complex128],
) -> None:
    size = states.shape[-1]
    for row in range(len(states)):
        key = key_index[row]
        rho = states[row].copy()
        for segment in range(_SUBSEGMENTS):
            unitary = np.zeros((size, size), dtype=np.complex128)
            for mode in range(_PHASE_COUNT):
                unitary += (
                    np.exp(1j * _MODES[mode] * phases[row])
                    * coefficients[key, segment, mode]
                )
            for half in range(2):
                diagonal = np.empty(size, dtype=np.complex128)
                for i in range(size):
                    diagonal[i] = rho[i, i]
                updated = populations[key].astype(np.complex128) @ diagonal
                rho *= coherences[key]
                for i in range(size):
                    rho[i, i] = updated[i]
                if half == 0:
                    rho = unitary @ rho @ unitary.conj().T
        states[row] = rho


class GateCacheKernel:
    """Cache a shaped single-tone segment independently of its start phase.

    The engine compares amortized preparation/apply work. It namespaces keys by
    the DC model identity. Keys encode flux level, port, envelope content identity,
    envelope-local offset, frequency, signed amplitude and duration. Absolute
    segment start and carrier theta do not affect preparation. Four closed U
    subsegments retain 16 phase modes each, with half-step secular D on both
    sides. Dispatch selects which keys first try six nodes checked at two
    independent phases; failed convergence uses the sixteen-node builder.
    A model-only secular split guard rejects long dissipative gates before
    preparation. Numerical build failures raise RuntimeError; invalid budgets,
    unsupported rows or mismatched representative keys raise ValueError. This
    fixed-order approximation does not certify arbitrary drives against budget.
    """

    name = "gate_cache"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Accept shaped single tones with a small model-only split estimate.

        Repetition count belongs to the engine. The four-way secular splitting
        uses duration**3 * Gamma * V**2 / 16 <= 0.01, without reading density.
        """
        _check_budget(budget)
        accepted = table.tones[rows] == 1
        if table.tone_envelope.shape[1]:
            shaped = np.array(
                [e.kind == "sampled" for e in table.envelopes], dtype=bool
            )
            candidates = np.flatnonzero(accepted)
            accepted[candidates] &= shaped[table.tone_envelope[rows[candidates], 0]]
        if not np.any(accepted):
            return accepted
        selected = rows[accepted]
        strength, gamma = drive_bounds(table, selected, source)
        peaks, _ = envelope_bounds(table)
        peak = peaks[table.tone_envelope[selected, 0]]
        estimate = (
            table.duration_ns[selected] ** 3
            * gamma
            * (table.tone_amplitude[selected, 0] * peak * strength) ** 2
            / _SUBSEGMENTS**2
        )
        accepted[accepted] = estimate <= 0.01
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Return fixed-width content keys, excluding theta and absolute start."""
        if np.any(table.tones[rows] != 1):
            raise ValueError("gate cache keys require single-tone rows")
        dtype = np.dtype(
            [
                ("level", "<i8", (table.level.shape[1],)),
                ("port", "<i8"),
                ("envelope", "S64"),
                ("offset", "<f8"),
                ("frequency", "<f8"),
                ("amplitude", "<f8"),
                ("duration", "<f8"),
            ]
        )
        keys = np.empty(len(rows), dtype=dtype)
        if not len(rows):
            # A table without tones has no tone columns to index.
            return keys
        keys["level"] = table.level[rows]
        keys["port"] = table.tone_port[rows, 0]
        identities = np.array(
            [envelope.identity for envelope in table.envelopes], dtype="S64"
        )
        keys["envelope"] = identities[table.tone_envelope[rows, 0]]
        keys["offset"] = table.tone_envelope_offset_ns[rows, 0]
        keys["frequency"] = table.tone_frequency_ghz[rows, 0]
        keys["amplitude"] = table.tone_amplitude[rows, 0]
        keys["duration"] = table.duration_ns[rows]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return constant matrix-action work per row after preparation."""
        return np.full(
            table.duration_ns[rows].shape, float(_SUBSEGMENTS * _PHASE_COUNT)
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
        *,
        phase_adaptive: NDArray[np.bool_] | None = None,
    ) -> Prepared:
        """Build immutable U Fourier coefficients and compact D from representatives.

        rows aligns one representative per key; equal keys give equal payloads.
        phase_adaptive is a key-aligned bool vector selected by dispatch. True
        tries six phase nodes with two probes, falling back to sixteen nodes
        on failed convergence. None uses sixteen nodes for every key. A
        misaligned mask raises ValueError. No duration-based selection occurs
        here. No density is accessed; nbytes counts all retained table arrays.
        """
        _check_budget(budget)
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows)
        ):
            raise ValueError("gate cache representative rows must match keys")
        if not self.guard(table, rows, source, budget).all():
            raise ValueError("gate cache requires shaped single-tone rows")
        if phase_adaptive is None:
            phase_adaptive = np.zeros(len(keys), dtype=bool)
        if phase_adaptive.shape != (len(keys),) or phase_adaptive.dtype != np.bool_:
            raise ValueError(
                "gate phase builder mask must be a key-aligned bool vector"
            )
        coefficients, populations, coherences = [], [], []
        for row, adaptive in zip(rows, phase_adaptive, strict=True):
            model = source.model(tuple(int(v) for v in table.level[row]))
            half_step = float(table.duration_ns[row]) / (2 * _SUBSEGMENTS)
            population = expm(half_step * population_generator(model.rates_per_ns))
            coherence = np.exp(-half_step * coherence_decay(model))
            coefficient = None
            if adaptive:
                coefficient = gate_phase.build(model, table, int(row), budget)
            if coefficient is None:
                phases = np.asarray(
                    np.arange(_PHASE_COUNT) * (2 * np.pi / _PHASE_COUNT),
                    dtype=np.float64,
                )
                endpoints = build_endpoints(model, table, int(row), budget, phases)
                coefficient = np.fft.fft(endpoints, axis=1) / _PHASE_COUNT
            coefficients.append(coefficient)
            populations.append(population)
            coherences.append(coherence)
        if len(rows):
            arrays = (
                np.stack(coefficients),
                np.stack(populations),
                np.stack(coherences),
            )
        else:
            arrays = (
                np.empty((0, _SUBSEGMENTS, _PHASE_COUNT, 0, 0), complex),
                np.empty((0, 0, 0)),
                np.empty((0, 0, 0)),
            )
        for array in arrays:
            if not np.isfinite(array).all():
                raise RuntimeError("gate cache preparation produced nonfinite data")
            array.setflags(write=False)
        return _GatePrepared(*arrays)

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> None:
        """Decline guard preparation handoff; the engine calls guard normally."""
        del table, rows, source, budget

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Decline input binding; apply uses the model-only prepared payload."""
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
        """Advance aligned densities in place using read-only prepared data.

        key_index selects prepared entries; rows selects table theta. Splitting
        or reordering rows does not change their results. The worker loop is
        single-threaded numba and releases the GIL.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        context.require_no_plan()
        budget = context.budget
        _check_budget(budget)
        if not isinstance(prepared, _GatePrepared):
            raise ValueError("gate cache requires its own prepared payload")
        if len(rows):
            _apply_gates(
                prepared.coefficients,
                prepared.populations,
                prepared.coherences,
                key_index,
                table.tone_theta_rad[rows, 0],
                states,
            )
        return np.zeros(len(rows), dtype=np.bool_)
