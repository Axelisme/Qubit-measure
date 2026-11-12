"""Public Engine-only measurement and comparison contracts, without real timers."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from itertools import count
from pathlib import Path

import numpy as np
import pytest
from family_pareto import (
    EngineSession,
    FamilyResult,
    ModeResult,
    Sample,
    dominance,
    main,
    measure_family,
    measure_policy,
    run_table,
    timing_equivalents,
)
from family_pareto_inputs import Input
from zcu_tools.simulate.evolution import (
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    RoutingHistory,
    TimelineBatch,
)

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case, history_case
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    converged_reference,
)


def test_three_modes_time_five_samples_and_exclude_setup_warmup_clear() -> None:
    visits: list[list[int]] = []
    clock = count()
    density = np.eye(2, dtype=np.complex128)[None] / 2

    @contextmanager
    def session() -> Generator[EngineSession]:
        history = 0
        seen: list[int] = []
        visits.append(seen)

        def evolve() -> EvolutionResult:
            nonlocal history
            seen.append(history)
            history += 1
            return EvolutionResult(
                np.array([0]), density, PathDiagnostics(segments_by_kernel={"t4": 1})
            )

        def clear() -> None:
            nonlocal history
            history = 0

        yield EngineSession(evolve, clear)

    result = measure_family(
        session,
        ReferenceResult(density, 0.0),
        wall_clock=lambda: float(next(clock)),
        cpu_clock=lambda: float(next(clock)),
    )
    assert visits == [
        [0],
        [0],
        [0],
        [0],
        [0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6],
    ]
    assert [mode.mode for mode in result] == [
        "first_evolve",
        "warm_clear_history",
        "warm_retain_history",
    ]
    for mode in result:
        assert len(mode.samples) == 5
        assert mode.wall_median == 3.0
        assert mode.cpu_median == 1.0
        assert all(sample.paths == {"t4": 1} for sample in mode.samples)
        assert all(sample.density_error == 0 for sample in mode.samples)


def test_dominance_keeps_ten_percent_boundary_modes_clocks_and_exact_errors_separate() -> (
    None
):
    def row(name: str, error: float, wall: float, cpu: float) -> FamilyResult:
        return FamilyResult(
            name,
            error,
            None,
            [
                ModeResult(mode, [Sample(wall, cpu, error, {name: 1})] * 5, wall, cpu)
                for mode in (
                    "first_evolve",
                    "warm_clear_history",
                    "warm_retain_history",
                )
            ],
        )

    rows = [
        row("target", 0.002, 10, 10),
        row("boundary", 0.002, 9, 9),
        row("faster", 0.002, 8.99, 10.1),
        row("less_accurate", 0.002000000001, 1, 1),
        row("default", 0, 0.1, 0.1),
        FamilyResult("unavailable", None, "refused", []),
    ]
    result = dominance(rows)
    assert result["target"]["first_evolve/wall"] == ["faster"]
    assert result["target"]["first_evolve/cpu"] == []
    assert result["boundary"]["warm_clear_history/wall"] == []
    assert result["faster"]["warm_retain_history/cpu"] == ["boundary"]
    assert "default" not in result
    assert "unavailable" not in result
    equivalent = timing_equivalents(rows)
    assert equivalent["target"]["first_evolve/wall"] == ["boundary"]
    assert equivalent["target"]["first_evolve/cpu"] == ["boundary", "faster"]
    assert equivalent["boundary"]["warm_clear_history/wall"] == ["target", "faster"]


@pytest.mark.parametrize("compare", [dominance, timing_equivalents])
def test_comparisons_pair_history_modes_by_name(compare) -> None:
    target = FamilyResult(
        "target", 0.001, None,
        [ModeResult(mode, [Sample(time, time, 0.001, {})] * 5, time, time)
         for mode, time in (("first_evolve", 10), ("warm_clear_history", 20),
                            ("warm_retain_history", 40))],
    )
    other = replace(target, family="other", modes=list(reversed(target.modes)))
    result = compare([target, other])
    expected = [] if compare is dominance else ["other"]
    assert result["target"] == {
        "first_evolve/wall": expected, "first_evolve/cpu": expected,
        "warm_clear_history/wall": expected, "warm_clear_history/cpu": expected,
        "warm_retain_history/wall": expected, "warm_retain_history/cpu": expected,
    }


@pytest.mark.sim
def test_real_engine_pins_path_and_reports_domain_refusal_without_zero_values() -> None:
    item = Input("idle", lambda: accuracy_case("idle_coherence"), "tiny Engine seam")
    case = item.build()
    try:
        reference = converged_reference(case.source, case.batch)
    finally:
        case.source.close()
    accepted = measure_policy(item, "t4", reference)
    assert accepted.unavailable is None
    assert accepted.error is not None and accepted.error < 2e-3
    assert all(s.paths == {"t4": 2} for mode in accepted.modes for s in mode.samples)
    refused = measure_policy(item, "dop853", reference)
    assert (
        refused.unavailable
        == "no valid kernel for segment 0; domain idle: points [0, 1], segments [0, 1]"
    )
    assert refused.error is None
    assert refused.modes == []


@pytest.mark.sim
def test_cli_freezes_then_measures_and_preserves_output_on_manifest_or_write_failure(
    tmp_path: Path,
) -> None:
    inputs = [
        Input("idle", lambda: accuracy_case("idle_coherence"), "tiny Engine seam")
    ]
    manifest = tmp_path / "inputs.json"
    output = tmp_path / "result.json"
    summary = tmp_path / "summary.md"
    assert main(["--freeze-inputs", str(manifest)], inputs=inputs) == 0
    assert manifest.exists()
    argv = [
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    assert main(argv, inputs=inputs) == 0
    assert '"unavailable"' in output.read_text()
    assert "t4" in summary.read_text()
    complete = output.read_text()
    manifest.write_text("{}")
    assert main(argv, inputs=inputs) == 1
    assert output.read_text() == complete
    assert main(["--freeze-inputs", str(tmp_path)], inputs=inputs) == 1


@pytest.mark.parametrize(
    "message", ["numerical integration failed", "no valid kernel: malformed refusal"]
)
def test_policy_does_not_disguise_other_runtime_errors_as_unavailable(
    message: str,
) -> None:
    error = RuntimeError(message)
    item = Input("idle", lambda: accuracy_case("idle_coherence"), "synthetic failure")

    @contextmanager
    def broken_session(item: Input, family: str) -> Generator[EngineSession]:
        def fail() -> EvolutionResult:
            raise error

        yield EngineSession(fail, lambda: None)

    with pytest.raises(RuntimeError) as caught:
        measure_policy(
            item,
            "t4",
            ReferenceResult(np.eye(2, dtype=np.complex128)[None] / 2, 0),
            opener=broken_session,
        )
    assert caught.value is error


def test_policy_does_not_allow_an_unpinned_fallback_to_masquerade_as_success() -> None:
    density = np.eye(2, dtype=np.complex128)[None] / 2
    item = Input(
        "idle", lambda: accuracy_case("idle_coherence"), "synthetic wrong route"
    )

    @contextmanager
    def wrong_session(item: Input, family: str) -> Generator[EngineSession]:
        yield EngineSession(
            lambda: EvolutionResult(
                np.array([0]), density, PathDiagnostics(segments_by_kernel={"t6": 1})
            ),
            lambda: None,
        )

    with pytest.raises(RuntimeError, match="pinned t4 did not execute exclusively"):
        measure_policy(item, "t4", ReferenceResult(density, 0), opener=wrong_session)


def test_late_warmup_failure_stops_measurement_without_partial_success() -> None:
    density = np.eye(2, dtype=np.complex128)[None] / 2
    calls = 0
    error = RuntimeError("warmup failed")

    @contextmanager
    def session() -> Generator[EngineSession]:
        def evolve() -> EvolutionResult:
            nonlocal calls
            calls += 1
            if calls == 6:
                raise error
            return EvolutionResult(
                np.array([0]), density, PathDiagnostics(segments_by_kernel={"t4": 1})
            )

        yield EngineSession(evolve, lambda: None)

    with pytest.raises(RuntimeError) as caught:
        measure_family(session, ReferenceResult(density, 0))
    assert caught.value is error
    assert calls == 6


def test_table_shares_oracle_for_identical_physics_with_different_model_identities() -> (
    None
):
    calls = 0
    inputs = [
        Input("linear", lambda: accuracy_case("linear_shaped"), "original"),
        Input(
            "history",
            lambda: history_case(2, RoutingHistory()),
            "same physics, different model identity",
        ),
    ]

    def solve(source: ModelSource, batch: TimelineBatch) -> ReferenceResult:
        nonlocal calls
        calls += 1
        return ReferenceResult(
            np.tile(source.initial_density, (len(batch.points), 1, 1)), 0
        )

    def policy(item: Input, family: str, reference: ReferenceResult) -> FamilyResult:
        return FamilyResult(family, 0, None, [])

    rows = run_table(
        inputs, ("single_tone_gate",), reference_solver=solve, policy_runner=policy
    )
    assert len(rows) == 2
    assert calls == 1


def test_cleanup_error_is_never_an_availability_refusal_even_with_matching_message() -> (
    None
):
    density = np.eye(2, dtype=np.complex128)[None] / 2
    error = RuntimeError(
        "no valid kernel for segment 0; domain idle: points [0], segments [0]"
    )
    item = Input("idle", lambda: accuracy_case("idle_coherence"), "cleanup isolation")

    @contextmanager
    def cleanup_fails(item: Input, family: str) -> Generator[EngineSession]:
        yield EngineSession(
            lambda: EvolutionResult(
                np.array([0]), density, PathDiagnostics(segments_by_kernel={"t4": 1})
            ),
            lambda: None,
        )
        raise error

    with pytest.raises(RuntimeError) as caught:
        measure_policy(item, "t4", ReferenceResult(density, 0), opener=cleanup_fails)
    assert caught.value is error
