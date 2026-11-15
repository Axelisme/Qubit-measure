"""Disk reference reuse at the solver seam, without expensive integration."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import ModelSource, TimelineBatch

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    SolverTolerances,
)
from tools.family_pareto import FamilyResult, run_table
from tools.family_pareto_inputs import Input
from tools.family_pareto_reference import (
    ReferenceCache,
    reference_cache_key,
    solve_reference,
)
from tools.family_pareto_scenarios import Dop853Coordinates, dop853_segment_input


def test_segment_reference_key_predicts_cache_hit_and_separates_repeats_and_points(
    tmp_path: Path,
) -> None:
    def solver(
        source: ModelSource,
        batch: TimelineBatch,
        *,
        coarse: SolverTolerances,
        fine: SolverTolerances,
    ) -> ReferenceResult:
        return ReferenceResult(
            np.tile(source.initial_density, (len(batch.points), 1, 1)), 1e-9
        )

    keys = []
    for coordinates in (
        Dop853Coordinates(levels=3),
        Dop853Coordinates(levels=3, repeats=2),
        Dop853Coordinates(levels=3, points=2),
    ):
        item = dop853_segment_input(coordinates)
        with closing((case := item.build()).source):
            key = reference_cache_key(case)
            keys.append(key)
            with pytest.raises(FileNotFoundError, match=key):
                solve_reference(case, cache_dir=tmp_path, cache_only=True)
            expected = solve_reference(case, solver=solver, cache_dir=tmp_path)
            assert (tmp_path / f"{key}.npz").exists()
        with closing((fresh := item.build()).source):
            assert reference_cache_key(fresh) == key
            actual = solve_reference(fresh, cache_dir=tmp_path, cache_only=True)
            np.testing.assert_array_equal(actual.density, expected.density)
    assert len(set(keys)) == 3


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


@pytest.mark.parametrize("change", ["missing", "physics", "tolerance"])
def test_cache_only_never_solves_on_missing_or_changed_keys(
    tmp_path: Path, change: str
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
        if change != "missing":
            solve_reference(case, solver=solver, cache_dir=tmp_path)
            hit = solve_reference(
                case, solver=solver, cache_dir=tmp_path, cache_only=True
            )
            assert hit.convergence_error == 1e-9
        before = calls
        changed = (
            replace(case, batch=replace(case.batch, end_ns=case.batch.end_ns + 1))
            if change == "physics"
            else case
        )
        fine = (
            SolverTolerances(1e-12, 1e-14)
            if change == "tolerance"
            else SolverTolerances(2e-12, 2e-14)
        )
        with pytest.raises(FileNotFoundError, match="reference cache miss"):
            solve_reference(
                changed, solver=solver, cache_dir=tmp_path, cache_only=True, fine=fine
            )
        assert calls == before


def test_table_cache_only_stops_before_timing_an_uncached_input(tmp_path: Path) -> None:
    timed: list[str] = []

    def policy(item: Input, family: str, reference: ReferenceResult) -> FamilyResult:
        timed.append(family)
        return FamilyResult(family, None, "unused", [])

    with pytest.raises(FileNotFoundError, match="reference cache miss"):
        run_table(
            [Input("idle", lambda: accuracy_case("idle_coherence"), "cache-only")],
            ("t4",),
            reference_cache=ReferenceCache(tmp_path, only=True),
            policy_runner=policy,
        )
    assert timed == []
