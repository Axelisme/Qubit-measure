"""Reduced rotating-wave evolution retains coherent inputs and secular baths."""

from dataclasses import replace

import numpy as np

from ..contract.kernel_suite import KernelCase, KernelContract

KERNEL_NAME = "rwa"


class TestRWAKernel(KernelContract):
    pass


def test_rwa_accepts_weak_rows_independently_of_dyson_branch(
    kernel_case: KernelCase,
) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    weak = replace(case.table, tone_amplitude=case.table.tone_amplitude * 0.01)
    assert case.kernel.guard(weak, rows, case.source, case.budget).all()
    assert len(np.unique(case.kernel.prepare_keys(case.table, rows))) == 1
    assert (case.kernel.cost(case.table, rows) >= 0).all()
    strong = replace(
        case.table, tone_amplitude=np.ones((4, 1)), duration_ns=np.full(4, 20.0)
    )
    assert not case.kernel.guard(strong, rows, case.source, case.budget).any()
