"""Exact static secular propagation: population semigroup and coherences."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ._secular import coherence_decay, population_generator


@dataclass(frozen=True)
class _Static:
    population: NDArray[np.float64]
    coherence: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return self.population.nbytes + self.coherence.nbytes


@njit(nogil=True, cache=True)
def _apply(population, coherence, indices, states):
    for row in range(len(states)):
        key = indices[row]
        size = states.shape[1]
        diagonal = np.empty(size, dtype=np.complex128)
        for i in range(size):
            diagonal[i] = states[row, i, i]
        for i in range(size):
            for j in range(size):
                if i != j:
                    states[row, i, j] *= coherence[key, i, j]
            value = 0j
            for j in range(size):
                value += population[key, i, j] * diagonal[j]
            states[row, i, i] = value


class T4:
    """Zero-tone static Lindblad channel; keys encode level and duration in ns.

    Population propagation is exp(Q*t). Each off-diagonal element acquires
    its exact energy phase and secular decay. No density is read in prepare.
    """

    name = "t4"
    guard_builds = False

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> NDArray[np.bool_]:
        """Accept precisely rows without playing tones."""
        source, budget = context.source, context.budget
        del source, budget
        return table.tones[rows] == 0

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Encode flux level and duration, excluding absolute start time."""
        del context
        keys = np.empty(
            len(rows),
            dtype=[
                ("level", "<i8", table.level.shape[1]),
                ("shift", "S64"),
                ("duration", "<f8"),
            ],
        )
        keys["level"] = table.level[rows]
        keys["shift"] = table.shift_content[rows]
        keys["duration"] = table.duration_ns[rows]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Static channel application has constant cost per row."""
        del table
        return np.ones(len(rows))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build immutable population matrices and coherence factors per key."""
        source, budget = context.source, context.budget
        del table, rows, budget
        populations, coherences = [], []
        for key in keys:
            model = source.shifted_model(
                tuple(int(v) for v in key["level"]), bytes(key["shift"])
            )
            duration = float(key["duration"])
            populations.append(
                expm(duration * population_generator(model.rates_per_ns))
            )
            gamma = coherence_decay(model)
            energy = model.energies_ghz
            coherences.append(
                np.exp(
                    duration
                    * (-2j * np.pi * (energy[:, None] - energy[None, :]) - gamma)
                )
            )
        population = np.asarray(populations, dtype=np.float64)
        coherence = np.asarray(coherences, dtype=np.complex128)
        population.setflags(write=False)
        coherence.setflags(write=False)
        return _Static(population, coherence)

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
        """Advance states in place without changing the shared payload.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        context.require_no_plan()
        budget = context.budget
        del table, budget
        if not isinstance(prepared, _Static):
            raise ValueError("T4 requires its own prepared payload")
        _apply(prepared.population, prepared.coherence, key_index, states)
        return np.zeros(len(rows), dtype=np.bool_)
