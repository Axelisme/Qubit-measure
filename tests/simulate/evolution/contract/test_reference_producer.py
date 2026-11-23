"""Frozen-input to independent-reference handoff, without tool imports."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.simulate.evolution import (
    FrozenIndexEntry,
    FrozenRecord,
    ModelSource,
    ReferenceResult,
    TimelineBatch,
    load_reference,
    physical_input,
    write_frozen_index,
    write_scene_record,
)

from . import family_pareto_reference_producer as producer
from .accuracy_cases import accuracy_case
from .interaction_reference import interaction_reference
from .reference import converged_reference


@pytest.mark.parametrize("interface", ["function", "main"])
def test_scenario_solves_only_selected_row_with_its_explicit_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interface: str
) -> None:
    case = accuracy_case("weak_plateau")
    with closing(case.source):
        record = FrozenRecord("chosen", physical_input(case.source, case.batch))
        expected = ReferenceResult(
            np.stack([case.source.initial_density] * len(case.batch.points)), 0.0
        )
    scenes, cache = tmp_path / "scenes", tmp_path / "references"
    index, other_index = tmp_path / "chosen.index.json", tmp_path / "other.index.json"
    write_frozen_index(
        index,
        (
            FrozenIndexEntry.from_record(replace(record, name="unselected")),
            write_scene_record(scenes, record),
        ),
    )
    write_frozen_index(
        other_index, (FrozenIndexEntry.from_record(replace(record, name="other")),)
    )
    calls: list[str] = []

    def selected_solver(source: ModelSource, batch: TimelineBatch) -> ReferenceResult:
        calls.append("interaction")
        return expected

    def forbidden_solver(source: ModelSource, batch: TimelineBatch) -> ReferenceResult:
        raise AssertionError("unselected solver must not run")

    if interface == "function":
        keys = producer.prepare_references(
            index, cache, scene_cache=scenes, solver=selected_solver, scenario="chosen"
        )
        assert list(keys) == ["chosen"]
    else:
        # Solvers are outside selection/cache I/O; the CLI has no solver injection.
        monkeypatch.setattr(producer, "converged_reference", forbidden_solver)
        monkeypatch.setattr(producer, "interaction_reference", selected_solver)
        assert (
            producer.main(
                [
                    "--converged",
                    str(other_index),
                    "--interaction",
                    str(index),
                    "--scene-cache",
                    str(scenes),
                    "--cache",
                    str(cache),
                    "--scenario",
                    "chosen",
                ]
            )
            == 0
        )
    assert calls == ["interaction"]
    with closing(case.source):
        assert_allclose(
            load_reference(cache, case.source, case.batch).density, expected.density
        )


@pytest.mark.parametrize("solver", ["--converged", "--interaction"])
def test_secular_producer_does_not_publish_full_loss_miss(
    tmp_path: Path, solver: str
) -> None:
    case = accuracy_case("idle_coherence")
    with closing(case.source):
        model = case.source.model((0,))
        jump = np.zeros_like(model.eigenvectors)
        jump[0, 1] = 0.2j
        full = replace(model, collapse_operators=(jump,))
        with closing(
            ModelSource(
                case.source.initial_density, lambda level: replace(full, level=level)
            )
        ) as source:
            record = FrozenRecord("full-loss", physical_input(source, case.batch))

    scenes, cache, index = (
        tmp_path / "scenes",
        tmp_path / "references",
        tmp_path / "scene.index.json",
    )
    write_frozen_index(index, (write_scene_record(scenes, record),))
    with pytest.raises(
        ValueError,
        match="full-loss.*reference solvers do not support explicit collapse operators",
    ):
        producer.main(
            [solver, str(index), "--scene-cache", str(scenes), "--cache", str(cache)]
        )
    assert not cache.exists()


@pytest.mark.parametrize("duplicate", [False, True])
def test_scenario_rejects_zero_or_multiple_index_matches_before_payloads_and_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate: bool
) -> None:
    case = accuracy_case("weak_plateau")
    with closing(case.source):
        record = FrozenRecord("chosen", physical_input(case.source, case.batch))
    indexes = [tmp_path / "first.index.json", tmp_path / "second.index.json"]
    for path in indexes:
        write_frozen_index(path, (FrozenIndexEntry.from_record(record),))
    calls: list[str] = []

    def forbidden_solver(source: ModelSource, batch: TimelineBatch) -> ReferenceResult:
        calls.append("solver")
        raise AssertionError("invalid selection must not run a solver")

    # No scene payload exists; both index selection and solver dispatch run normally.
    monkeypatch.setattr(producer, "converged_reference", forbidden_solver)
    monkeypatch.setattr(producer, "interaction_reference", forbidden_solver)
    scenario = "chosen" if duplicate else "unknown"
    with pytest.raises(
        ValueError, match="Ambiguous scenario" if duplicate else "Unknown scenario"
    ) as failure:
        producer.main(
            [
                "--converged",
                str(indexes[0]),
                "--interaction",
                str(indexes[1]),
                "--scene-cache",
                str(tmp_path / "no-scenes"),
                "--cache",
                str(tmp_path / "references"),
                "--scenario",
                scenario,
            ]
        )
    assert "chosen" in str(failure.value)
    if duplicate:
        assert all(str(path) in str(failure.value) for path in indexes)
    else:
        with pytest.raises(ValueError, match="Unknown scenario"):
            producer.prepare_references(
                indexes[0],
                tmp_path / "references",
                scene_cache=tmp_path / "no-scenes",
                solver=forbidden_solver,
                scenario=scenario,
            )
    assert calls == []


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
