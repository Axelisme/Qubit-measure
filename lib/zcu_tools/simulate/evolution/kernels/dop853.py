"""Batched arbitrary-envelope Lindblad fallback in each secular eigenbasis."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource
from ._dop853 import tolerances
from ._dop853_chain import run_chain
from ._dop853_data import (
    Context,
    Models,
    hamiltonian,
    pack_models,
    pack_segments,
)


@dataclass(frozen=True)
class _Prepared:
    models: Models

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.models)


@njit(nogil=True, cache=True)
def _rhs(time, state, context):
    size = context.models.energies.shape[1]
    rho = state.reshape(size, size)
    h = hamiltonian(time, context)
    derivative = -2j * np.pi * (h @ rho - rho @ h)
    derivative -= context.models.decay[context.model] * rho
    for target in range(size):
        for source in range(size):
            derivative[target, target] += (
                context.models.rates[context.model, target, source]
                * rho[source, source]
            )
    return derivative.ravel()


@njit(nogil=True, cache=True)
def _context(data, row, model):
    models, segments = data
    return Context(models, segments, row, model)


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
        return _Prepared(pack_models(models))

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
        segments = pack_segments(table)
        run_chain(
            _rhs,
            _context,
            (models, segments),
            models,
            segments,
            rows,
            key_index,
            states,
            controls,
        )
