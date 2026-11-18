"""Full-drive simultaneous gates with checked raw dual-phase playback."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ...baths import population_generator
from ..kernel import (
    ApplyContext,
    GuardPreparation,
    GuardResult,
    KernelContext,
    Prepared,
    SegmentTable,
)
from . import _multitone_gate_phase as phase
from ._gate_endpoints import GATE_SUBSEGMENTS
from ._multitone_gate_split import SPLIT_PHASES, advance_probes, probe_densities
from ._rwa_gate import canonical_keys
from ._secular import coherence_decay, split_step
from ._workload_facts import fixed_flux

_SPLIT_BUDGET_FRACTION = 0.1
_SPLIT_FALLBACK_FACTOR = 1.5


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


def _split_defects(
    prepared: _Prepared, table: SegmentTable, row: int, context: KernelContext
) -> tuple[float, float]:
    """Return max-entry defects of playback and fallback vs one tight reference.

    A complete Hermitian operator basis would cost M² solves and store an M⁴
    channel per phase. Four physical densities instead sample these maps;
    neither defect bounds other phases, inputs or complete sequences.
    """
    model = table.model(context.source, row)
    split = probe_densities(len(model.energies_ghz))
    reference, fallback = advance_probes(model, table, row, split, context.budget)
    _play(
        prepared.coefficients,
        (prepared.populations, prepared.coherences),
        np.zeros(len(SPLIT_PHASES), dtype=np.int64),
        SPLIT_PHASES,
        split,
    )
    return (
        float(np.max(np.abs(split - reference))),
        float(np.max(np.abs(fallback - reference))),
    )


def _build(table: SegmentTable, row: int, context: KernelContext) -> _Prepared:
    model = table.model(context.source, row)
    coefficient = phase.build(model, table, row)
    half = float(table.duration_ns[row]) / (2 * GATE_SUBSEGMENTS)
    arrays = (
        coefficient[None],
        np.asarray(
            expm(half * population_generator(model.rates_per_ns)), dtype=np.float64
        )[None],
        np.exp(-half * coherence_decay(model))[None],
    )
    for array in arrays:
        array.setflags(write=False)
    return _Prepared(*arrays)


class MultitoneGateKernel:
    """Prepare one full-drive key for arbitrary joint density and two phases.

    Require two tones, at least one sampled envelope, fixed flux and positive
    duration. Fit one 5x6 phase grid at rtol/atol 2.75e-7/2.75e-9.
    Compare playback and Engine-budget DOP853 with the same tight Lindblad
    endpoints at four phase/density pairs, three outside the grid. This checks
    interpolation and splitting together; it does not bound sequence error.
    Return a rejected guard on excess defect without rebuilding.
    Numerical failures propagate. Phase and absolute start are apply-only.
    Guard products live only within one evolve (ADR-0078). No RWA or pruning.
    """

    name = "multitone_gate"
    guard_builds = True

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Return row-local admission, discarding built products without handoff."""
        return self.guard_prepared(table, rows, context).accepted

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

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> GuardResult:
        """Return checked key products once, aligned to all admitted consumers.

        Numerical products remain in GuardPreparation, never on this kernel or
        in routing-history facts. Reordering/subsetting rows preserves admission.
        """
        accepted = np.zeros(len(rows), dtype=bool)
        positions = np.array(
            [
                index
                for index, row in enumerate(rows)
                if table.tones[row] == 2
                and table.duration_ns[row] > 0
                and not table.constant_envelopes(np.array([row]))[0]
                and fixed_flux(table, int(row))
            ],
            dtype=np.int64,
        )
        if not len(positions):
            return GuardResult(accepted)
        selected = rows[positions]
        keys, representatives, inverse = np.unique(
            self.prepare_keys(table, selected, context),
            return_index=True,
            return_inverse=True,
        )
        products = []
        for index, representative in enumerate(representatives):
            row = int(selected[representative])
            prepared = _build(table, row, context)
            defect, fallback_defect = _split_defects(prepared, table, row, context)
            # Stay comparable to the direct solver this gate replaces. Keep
            # a budget-scaled allowance when the fallback is especially accurate.
            floor = _SPLIT_BUDGET_FRACTION * min(context.budget, 1e-3)
            threshold = max(floor, _SPLIT_FALLBACK_FACTOR * fallback_defect)
            if defect <= threshold:
                accepted[positions[inverse == index]] = True
                products.append(GuardPreparation(keys[index : index + 1], prepared))
        return GuardResult(accepted, tuple(products))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build checked representatives if no guard handoff exists; refuse invalid keys."""
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows, context)
        ):
            raise ValueError("full-drive gate representatives must match keys")
        result = self.guard_prepared(table, rows, context)
        if not result.accepted.all():
            raise ValueError("full-drive gate requires validated dual-tone keys")
        by_key: dict[bytes, _Prepared] = {}
        for entry in result.preparations:
            if not isinstance(entry.prepared, _Prepared):
                raise TypeError("full-drive gate requires its own preparation")
            by_key[entry.keys[0].tobytes()] = entry.prepared
        ordered = [by_key[key.tobytes()] for key in keys]
        arrays = (
            np.concatenate([product.coefficients for product in ordered]),
            np.concatenate([product.populations for product in ordered]),
            np.concatenate([product.coherences for product in ordered]),
        )
        for array in arrays:
            array.setflags(write=False)
        return _Prepared(*arrays)

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
