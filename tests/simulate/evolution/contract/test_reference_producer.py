"""Frozen-input to independent-reference handoff, without tool imports."""

from contextlib import closing
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from zcu_tools.simulate.evolution import (
    FrozenIndexEntry,
    FrozenRecord,
    ReferenceResult,
    load_reference,
    physical_input,
    write_frozen_index,
    write_scene_record,
)

from . import family_pareto_reference_producer as producer
from .accuracy_cases import accuracy_case
from .interaction_reference import interaction_reference
from .reference import converged_reference


def test_missing_scene_stops_reference_batch_before_any_solver(tmp_path: Path) -> None:
    from dataclasses import replace

    case = accuracy_case("weak_plateau")
    with closing(case.source):
        record = FrozenRecord("present", physical_input(case.source, case.batch))
    scenes, index = tmp_path / "scenes", tmp_path / "inputs.index.json"
    first = write_scene_record(scenes, record)
    missing = FrozenIndexEntry.from_record(replace(record, name="missing"))
    write_frozen_index(index, (first, missing))
    calls: list[str] = []

    def forbidden_solver(*args: object) -> ReferenceResult:
        calls.append("solver")
        raise AssertionError("missing numeric scenes must not start any solver")

    with pytest.raises(FileNotFoundError) as failure:
        producer.prepare_references(
            index, tmp_path / "references", scene_cache=scenes, solver=forbidden_solver
        )
    assert "missing" in str(failure.value)
    assert missing.physical_sha256 in str(failure.value)
    assert "python -m tools.family_pareto_scenarios" in str(failure.value)
    assert calls == []


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
        scenes = tmp_path / "scenes"
        record = FrozenRecord(
            "caller-label",
            physical_input(case.source, case.batch),
            {"category": "not-a-solver-selector"},
        )
        write_frozen_index(inputs, (write_scene_record(scenes, record),))
        keys = producer.prepare_references(
            inputs, cache, scene_cache=scenes, solver=solver
        )
        assert list(keys) == ["caller-label"]
        assert (
            producer.main(
                [
                    f"--{method}",
                    str(inputs),
                    "--scene-cache",
                    str(scenes),
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
