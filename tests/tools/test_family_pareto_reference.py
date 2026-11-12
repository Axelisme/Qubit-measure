"""Disk reference reuse at the solver seam, without expensive integration."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
from zcu_tools.simulate.evolution import ModelSource, TimelineBatch

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    SolverTolerances,
)
from tools.family_pareto_reference import solve_reference


def test_cache_reuses_density_but_invalidates_physics_and_tolerances(
    tmp_path: Path,
) -> None:
    calls = 0

    def solver(
        source: ModelSource,
        batch: TimelineBatch,
        *,
        coarse: SolverTolerances,
        fine: SolverTolerances,
    ) -> ReferenceResult:
        nonlocal calls
        calls += 1
        return ReferenceResult(
            np.tile(source.initial_density, (len(batch.points), 1, 1)), 1e-9
        )

    case = accuracy_case("idle_coherence")
    with closing(case.source):
        first = solve_reference(case, solver=solver, cache_dir=tmp_path)
        second = solve_reference(case, solver=solver, cache_dir=tmp_path)
        np.testing.assert_array_equal(second.density, first.density)
        assert second.convergence_error == 1e-9
        assert calls == 1

        changed = replace(case, batch=replace(case.batch, end_ns=case.batch.end_ns + 1))
        solve_reference(changed, solver=solver, cache_dir=tmp_path)
        assert calls == 2

        solve_reference(
            case,
            solver=solver,
            cache_dir=tmp_path,
            fine=SolverTolerances(1e-12, 1e-14),
        )
        assert calls == 3

        solve_reference(case, solver=solver)
        solve_reference(case, solver=solver)
        assert calls == 5
