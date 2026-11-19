"""Full-drive simultaneous gates with raw dual-phase playback."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ..workplan import FactKey
from . import _multitone_gate_phase as phase
from ._gate_endpoints import GATE_SUBSEGMENTS
from ._rwa_gate import canonical_keys
from ._secular import secular_half_step, split_step
from ._workload_facts import fixed_flux


@dataclass(frozen=True)
class _Prepared:
    """Read-only key-aligned Fourier U and half-quarter secular D maps."""

    coefficients: NDArray[np.complex128]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (self.coefficients, self.populations, self.coherences)
        )


@njit(nogil=True, cache=True)
def _play(coefficients, baths, key_index, phases, states):
    populations, coherences = baths
    size = states.shape[-1]
    count_a, count_b = coefficients.shape[2:4]
    for row in range(len(states)):
        key = key_index[row]
        rho = states[row].copy()
        for segment in range(GATE_SUBSEGMENTS):
            unitary = np.zeros((size, size), dtype=np.complex128)
            for a in range(count_a):
                mode_a = a if a <= (count_a - 1) // 2 else a - count_a
                factor_a = np.exp(1j * mode_a * phases[row, 0])
                for b in range(count_b):
                    mode_b = b if b <= (count_b - 1) // 2 else b - count_b
                    factor = factor_a * np.exp(1j * mode_b * phases[row, 1])
                    unitary += factor * coefficients[key, segment, a, b]
            rho = split_step(rho, unitary, populations[key], coherences[key])
        states[row] = rho


def _build(table: SegmentTable, row: int, context: KernelContext) -> _Prepared:
    model = table.model(context.source, row)
    coefficient = phase.build(model, table, row)
    half = float(table.duration_ns[row]) / (2 * GATE_SUBSEGMENTS)
    population, coherence = secular_half_step(model, half)
    arrays = (coefficient[None], population[None], coherence[None])
    for array in arrays:
        array.setflags(write=False)
    return _Prepared(*arrays)


def gate_candidates(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return fixed-flux, positive-duration, sampled dual-tone candidate rows.

    Indices are local to table. This check does not build phase products.
    """
    selected = rows[(table.tones[rows] == 2) & (table.duration_ns[rows] > 0)]
    selected = selected[~table.constant_envelopes(selected)]
    if table.flux_waveform is None:
        return selected
    return np.array(
        [row for row in selected if fixed_flux(table, int(row))], dtype=np.int64
    )


class MultitoneGateKernel:
    """Prepare one full-drive key for arbitrary joint density and two phases.

    Require two tones, at least one sampled envelope, fixed flux and positive
    duration. Fit one 5x6 phase grid at rtol/atol 2.75e-7/2.75e-9.
    Build after selection and propagate numerical failures. Phase and absolute
    start are apply-only. Products live only within one evolve (ADR-0078).
    No RWA or pruning.
    """

    name = "multitone_gate"
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Check only the row's structural eligibility."""
        del context
        return np.isin(rows, gate_candidates(table, rows))

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Structural eligibility does not require a prepared product."""
        del table, rows, context

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Hash complete physics/policy and budget, excluding start and phases."""
        canonical = canonical_keys(table, context)
        return np.array(
            [
                np.void(sha256(canonical[row] + repr(context.budget).encode()).digest())
                for row in rows
            ],
            dtype="V32",
        )

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return the four-quarter apply work for the existing dispatch preference."""
        del table
        return np.full(len(rows), float(GATE_SUBSEGMENTS))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Prepare selected keys in order, reusing builds across all evolve consumers."""
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows, context)
        ):
            raise ValueError("full-drive gate representatives must match keys")
        if not self.guard(table, rows, context).all():
            raise ValueError("full-drive gate requires eligible dual-tone keys")
        ordered = [
            self._prepare_key(table, int(row), context, key)
            for key, row in zip(keys, rows, strict=True)
        ]
        if len(ordered) == 1:
            return ordered[0]
        arrays = (
            np.concatenate([product.coefficients for product in ordered]),
            np.concatenate([product.populations for product in ordered]),
            np.concatenate([product.coherences for product in ordered]),
        )
        for array in arrays:
            array.setflags(write=False)
        return _Prepared(*arrays)

    def _prepare_key(
        self, table: SegmentTable, row: int, context: KernelContext, key: np.void
    ) -> _Prepared:
        def build() -> _Prepared:
            payload = _build(table, row, context)
            context.work_plan.routing.prepared_keys(self.name, 1)
            return payload

        return context.work_plan.fact(
            FactKey(("full-drive-gate-preparation", key.tobytes()), _Prepared), build
        )

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """Keep the model-only payload; playback accepts arbitrary joint density."""

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Interleave four full-drive U with D half-steps at both row-start phases.

        No density clipping, trace repair or per-qubit factorization occurs.
        Invalid payload, plan or state/index shape raises ValueError.
        """
        context.require_no_plan()
        if not isinstance(prepared, _Prepared):
            raise ValueError("full-drive gate requires its own payload")
        size = prepared.coefficients.shape[-1]
        if (
            key_index.shape != rows.shape
            or states.shape != (len(rows), size, size)
            or np.any(key_index < 0)
            or np.any(key_index >= len(prepared.coefficients))
        ):
            raise ValueError("full-drive gate state/index shape mismatch")
        _play(
            prepared.coefficients,
            (prepared.populations, prepared.coherences),
            key_index,
            table.tone_theta_rad[rows, :2],
            states,
        )
        return np.zeros(len(rows), dtype=bool)
