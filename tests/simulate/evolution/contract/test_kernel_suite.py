"""Self-check the reusable contract with a test-only static secular kernel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zcu_tools.simulate.evolution as evolution
from numpy.typing import NDArray
from scipy.linalg import expm

from .kernel_suite import KernelCase, KernelContract, synthetic_model

if TYPE_CHECKING:
    from zcu_tools.simulate.evolution import kernel


@dataclass(frozen=True)
class DemoPrepared:
    """Dense static channels, used only by this test's demonstration kernel."""

    channels: NDArray[np.complex128]

    @property
    def nbytes(self) -> int:
        return self.channels.nbytes


class DemoKernel:
    """A deliberately small zero-tone kernel, independent of the ODE oracle."""

    name = "demo-static"

    def guard(
        self,
        table: kernel.SegmentTable,
        rows: NDArray[np.int64],
        source: evolution.ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        return table.tones[rows] == 0

    def prepare_keys(
        self, table: kernel.SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        keys = np.empty(len(rows), dtype=[("duration", "f8")])
        keys["duration"] = table.duration_ns[rows]
        return keys

    def cost(
        self, table: kernel.SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        return np.ones(len(rows))

    def prepare(
        self,
        keys: NDArray[np.void],
        table: kernel.SegmentTable,
        rows: NDArray[np.int64],
        source: evolution.ModelSource,
        budget: float,
    ) -> DemoPrepared:
        model = source.model((0,))
        size = len(model.energies_ghz)
        rates = model.rates_per_ns
        outgoing = rates.sum(axis=0)
        generator = np.diag(
            (
                -2j
                * np.pi
                * (model.energies_ghz[:, None] - model.energies_ghz[None, :])
                - (outgoing[:, None] + outgoing[None, :]) / 2
                - model.dephasing_per_ns
            ).ravel()
        )
        populations = np.arange(size) * (size + 1)
        generator[np.ix_(populations, populations)] = rates - np.diag(outgoing)
        return DemoPrepared(
            np.stack([expm(float(key["duration"]) * generator) for key in keys]).astype(
                np.complex128
            )
        )

    def apply(
        self,
        prepared: kernel.Prepared,
        key_index: NDArray[np.int64],
        table: kernel.SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> None:
        if not isinstance(prepared, DemoPrepared):
            raise ValueError("demo requires DemoPrepared")
        for i, key in enumerate(key_index):
            states[i] = (prepared.channels[key] @ states[i].ravel()).reshape(
                states.shape[-2:]
            )


@pytest.fixture(params=[3, 4])
def kernel_case(request: pytest.FixtureRequest) -> KernelCase:
    size: int = request.param
    model = synthetic_model(size)
    from zcu_tools.simulate.evolution import kernel

    source = evolution.ModelSource(
        (np.eye(size, dtype=np.complex128) / size).astype(np.complex128),
        lambda level: replace(model, level=level),
    )
    rows = 4
    table = kernel.SegmentTable(
        np.arange(rows),
        np.array([0.0, 2.0, 7.0, 1.0]),
        np.array([0.7, 1.3, 0.7, 2.1]),
        np.zeros((rows, 1), dtype=np.int64),
        np.zeros(rows, dtype=np.int64),
        np.empty((rows, 0), dtype=np.int64),
        np.empty((rows, 0)),
        np.empty((rows, 0)),
        np.empty((rows, 0)),
        np.empty((rows, 0), dtype=np.int64),
        np.empty((rows, 0)),
        (),
    )
    vector = np.exp(1j * np.arange(size)) / np.sqrt(size)
    state = np.outer(vector, vector.conj())
    return KernelCase(DemoKernel(), source, table, np.repeat(state[None], rows, axis=0))


class TestDemoKernel(KernelContract):
    pass


def test_segment_table_rejects_misaligned_rows_and_invalid_tones(
    kernel_case: KernelCase,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        replace(kernel_case.table, duration_ns=np.ones(2))
    with pytest.raises(ValueError, match="tone counts"):
        replace(kernel_case.table, tones=np.ones(4, dtype=np.int64))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(kernel_case.table, duration_ns=-np.ones(4))


def test_suite_detects_incorrect_dynamics(kernel_case: KernelCase) -> None:
    class NoEvolution(DemoKernel):
        def apply(
            self,
            prepared: kernel.Prepared,
            key_index: NDArray[np.int64],
            table: kernel.SegmentTable,
            rows: NDArray[np.int64],
            states: NDArray[np.complex128],
            budget: float,
        ) -> None:
            pass

    with pytest.raises(AssertionError):
        KernelContract().test_error_within_budget(
            replace(kernel_case, kernel=NoEvolution())
        )


def test_suite_detects_prepare_reading_density(kernel_case: KernelCase) -> None:
    class ReadsDensity(DemoKernel):
        def prepare(
            self,
            keys: NDArray[np.void],
            table: kernel.SegmentTable,
            rows: NDArray[np.int64],
            source: evolution.ModelSource,
            budget: float,
        ) -> DemoPrepared:
            _ = source.initial_density
            return super().prepare(keys, table, rows, source, budget)

    with pytest.raises(AssertionError, match="must not read density"):
        KernelContract().test_prepare_is_density_independent(
            replace(kernel_case, kernel=ReadsDensity())
        )


def test_suite_detects_row_position_dependence(kernel_case: KernelCase) -> None:
    class PositionDependent(DemoKernel):
        def apply(
            self,
            prepared: kernel.Prepared,
            key_index: NDArray[np.int64],
            table: kernel.SegmentTable,
            rows: NDArray[np.int64],
            states: NDArray[np.complex128],
            budget: float,
        ) -> None:
            super().apply(prepared, key_index, table, rows, states, budget)
            states[:] *= (np.arange(len(rows)) + 1)[:, None, None]

    with pytest.raises(AssertionError):
        KernelContract().test_rows_can_be_split(
            replace(kernel_case, kernel=PositionDependent())
        )
