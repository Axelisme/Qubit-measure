"""Frozen-input to independent-reference handoff, without tool imports."""

from contextlib import closing
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from zcu_tools.simulate.evolution import (
    FrozenRecord,
    load_reference,
    physical_input,
    write_frozen_manifest,
)

from . import family_pareto_reference_producer as producer
from .accuracy_cases import accuracy_case
from .interaction_reference import interaction_reference
from .reference import converged_reference


@pytest.mark.parametrize("method", ["converged", "interaction"])
def test_producer_uses_explicit_solver_on_frozen_physics(
    tmp_path: Path,
    method: str,
) -> None:
    case = accuracy_case("weak_plateau")
    solver = converged_reference if method == "converged" else interaction_reference
    with closing(case.source):
        inputs = tmp_path / "inputs.json.gz"
        cache = tmp_path / "references"
        # Deliberately unrelated metadata and names cannot select the solver.
        write_frozen_manifest(
            inputs,
            [
                FrozenRecord(
                    "caller-label",
                    physical_input(case.source, case.batch),
                    {"category": "not-a-solver-selector"},
                )
            ],
        )
        keys = producer.prepare_references(inputs, cache, solver=solver)
        assert list(keys) == ["caller-label"]
        assert (
            producer.main(
                [
                    f"--{method}",
                    str(inputs),
                    "--cache",
                    str(cache),
                ]
            )
            == 0
        )
        result = load_reference(cache, case.source, case.batch)
        expected = solver(case.source, case.batch)
        assert_allclose(result.density, expected.density, rtol=0, atol=1e-14)
        assert result.convergence_error == expected.convergence_error
