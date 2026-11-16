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
from ._rwa_gate import canonical_keys
from ._secular import coherence_decay, split_step
from ._workload_facts import envelope_bounds, fixed_flux


@dataclass(frozen=True)
class _Prepared:
    """Read-only key-aligned Fourier U and half-quarter secular D maps.

    coefficients may pad either Fourier axis for batch storage. grid_sizes
    contains each key's active (Na,Nb); padding never defines Fourier modes.
    """

    coefficients: NDArray[np.complex128]
    grid_sizes: NDArray[np.int64]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.coefficients,
                self.grid_sizes,
                self.populations,
                self.coherences,
            )
        )


@njit(nogil=True, cache=True)
def _play(coefficients, grid_sizes, baths, key_index, phases, states):
    populations, coherences = baths
    size = states.shape[-1]
    for row in range(len(states)):
        key = key_index[row]
        count_a, count_b = grid_sizes[key]
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


def _split_admitted(table: SegmentTable, row: int, context: KernelContext) -> bool:
    """Bound the four-quarter secular split using complete drive quadratures."""
    model = table.model(context.source, row)
    peaks = envelope_bounds(table, context.work_plan).peaks
    strength = 0.0
    for tone in range(2):
        drive = model.drives[table.tone_port[row, tone]]
        strength += (
            abs(table.tone_amplitude[row, tone])
            * peaks[table.tone_envelope[row, tone]]
            * (
                np.linalg.norm(drive.in_phase_ghz, 2)
                + np.linalg.norm(drive.quadrature_ghz, 2)
            )
        )
    gamma = max(
        float(np.max(coherence_decay(model))),
        float(np.max(model.rates_per_ns.sum(axis=0))),
    )
    return bool(table.duration_ns[row] ** 3 * gamma * strength**2 / 16 <= 0.01)


def _build(table: SegmentTable, row: int, context: KernelContext) -> _Prepared | None:
    model = table.model(context.source, row)
    coefficient = phase.build(model, table, row)
    if coefficient is None:
        return None
    half = float(table.duration_ns[row]) / (2 * GATE_SUBSEGMENTS)
    arrays = (
        coefficient[None],
        np.array([coefficient.shape[1:3]], dtype=np.int64),
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
    duration. A model-only four-way secular split proxy must be <=0.01. Build
    a 9x10 Fourier grid and 64 independent validation points together, at
    rtol/atol 2.75e-7/2.75e-9. Failed validation rebuilds once at 13x13 and
    1e-8/1e-10. Admit only keys whose every quarter has full-unitary Frobenius
    defect <=1.9e-7. Two failed stages refuse the key; solver failures propagate.
    Phase and absolute start are apply-only, not key fields. Guard preparations
    are handed to the scheduler and live only within one evolve (ADR-0078).
    No RWA or graph pruning is performed. The empirical validation and split
    proxy do not certify arbitrary sequences against the final density budget.
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
            if not _split_admitted(table, row, context):
                continue
            prepared = _build(table, row, context)
            if prepared is not None:
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
        grid_sizes = np.concatenate([product.grid_sizes for product in ordered])
        width_a, width_b = grid_sizes.max(axis=0)
        coefficients = np.concatenate(
            [
                np.pad(
                    product.coefficients,
                    (
                        (0, 0),
                        (0, 0),
                        (0, width_a - product.grid_sizes[0, 0]),
                        (0, width_b - product.grid_sizes[0, 1]),
                        (0, 0),
                        (0, 0),
                    ),
                )
                for product in ordered
            ]
        )
        arrays = (
            coefficients,
            grid_sizes,
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
            prepared.grid_sizes,
            (prepared.populations, prepared.coherences),
            key_index,
            table.tone_theta_rad[rows, :2],
            states,
        )
        return np.zeros(len(rows), dtype=bool)
