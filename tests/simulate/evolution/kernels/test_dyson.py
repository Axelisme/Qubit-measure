"""Full-carrier, arbitrary-state second-order evolution at the C5 seam."""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec

from ..contract.kernel_suite import KernelCase, KernelContract, assert_budget, reference

KERNEL_NAME = "dyson"


class TestDysonKernel(KernelContract):
    pass


def test_preparation_shared_over_all_sweep_axes(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    keys = case.kernel.prepare_keys(case.table, rows)
    assert len(np.unique(keys)) == 1
    changed = replace(case.table, level=np.array([[0], [1], [0], [1]]))
    assert len(np.unique(case.kernel.prepare_keys(changed, rows))) == 2
    assert np.isfinite(case.kernel.cost(case.table, rows)).all()
    assert (case.kernel.cost(case.table, rows) >= 0).all()


def test_guard_rejects_only_local_invalidity(kernel_case: KernelCase) -> None:
    case = kernel_case
    rows = np.arange(4, dtype=np.int64)
    strong = replace(
        case.table, tone_amplitude=np.full((4, 1), 0.3), duration_ns=np.full(4, 20.0)
    )
    assert not case.kernel.guard(strong, rows, case.source, case.budget).any()
    shaped = replace(
        case.table,
        envelopes=(EnvelopeSpec("sampled", np.array([0.0, 1.0]), np.ones(2, complex)),),
    )
    assert not case.kernel.guard(shaped, rows, case.source, case.budget).any()


def test_full_carrier_and_tight_budget(kernel_case: KernelCase) -> None:
    case = kernel_case
    # Non-circular drive requires both carrier sidebands. Tiny gains isolate
    # divided-difference accuracy and pruning from the third-order remainder.
    model = case.source.model((0,))
    from zcu_tools.simulate.evolution import ModelSource

    model = replace(
        model,
        drives=(
            replace(
                model.drives[0],
                quadrature_ghz=np.zeros_like(model.drives[0].quadrature_ghz),
            ),
        ),
    )
    source = ModelSource(
        case.source.initial_density, lambda level: replace(model, level=level)
    )
    table = replace(case.table, tone_amplitude=case.table.tone_amplitude * 1e-3)
    case = replace(case, table=table, source=source, budget=1e-11)
    rows = np.arange(4, dtype=np.int64)
    keys, first, inverse = np.unique(
        case.kernel.prepare_keys(table, rows), return_index=True, return_inverse=True
    )
    prepared = case.kernel.prepare(keys, table, rows[first], source, case.budget)
    before = prepared.nbytes
    actual = case.states.copy()
    case.kernel.apply(prepared, inverse, table, rows, actual, case.budget)
    expected = np.stack(
        [reference(case, i, state) for i, state in enumerate(case.states)]
    )
    assert_budget(actual, expected, case.budget)
    assert prepared.nbytes == before
