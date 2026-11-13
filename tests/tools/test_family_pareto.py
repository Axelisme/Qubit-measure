"""Public Engine-only measurement and comparison contracts, without real timers."""

import json
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
from itertools import count
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    EvolutionResult,
    FamilyRouting,
    ModelSource,
    NoValidKernel,
    PathDiagnostics,
    RoutingHistory,
    TimelineBatch,
)

from tests.simulate.evolution.contract.accuracy import ALL_FAMILIES
from tests.simulate.evolution.contract.accuracy_cases import accuracy_case, history_case
from tests.simulate.evolution.contract.guard_sweep_cost import measure_engine_cost
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    converged_reference,
)
from tools import family_pareto_report
from tools.family_pareto import (
    EngineSession,
    FamilyResult,
    InputResult,
    ModeResult,
    Sample,
    dominance,
    main,
    measure_family,
    measure_policy,
    render_summary,
    run_table,
    timing_equivalents,
)
from tools.family_pareto_inputs import Input, canonical_json, freeze_manifest


@pytest.mark.parametrize("measure_errors", [True, False])
def test_three_modes_time_five_samples_and_exclude_setup_warmup_clear(
    measure_errors: bool,
) -> None:
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
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"t4": 1},
                    duration_ns_by_kernel={
                        name: float(count) for name, count in ({"t4": 1}).items()
                    },
                ),
            )

        def clear() -> None:
            nonlocal history
            history = 0

        yield EngineSession(evolve, clear)

    result = measure_family(
        session,
        ReferenceResult(density, 0.0) if measure_errors else None,
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
        expected_error = 0 if measure_errors else None
        assert all(sample.density_error == expected_error for sample in mode.samples)


@pytest.mark.parametrize("seconds", [60.0, 61.0])
def test_slow_family_uses_two_evolves_in_the_same_history(seconds: float) -> None:
    density = np.eye(2, dtype=np.complex128)[None] / 2
    wall_ticks = count()
    visits: list[list[int]] = []

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
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"t4": 1},
                    duration_ns_by_kernel={
                        name: float(count) for name, count in ({"t4": 1}).items()
                    },
                ),
            )

        yield EngineSession(evolve, lambda: None)

    modes = measure_family(
        session,
        None,
        wall_clock=lambda: float(next(wall_ticks)) * seconds,
        cpu_clock=lambda: 0.0,
    )
    if seconds == 60:
        assert [len(mode.samples) for mode in modes] == [5] * 3
        assert [mode.wall_median for mode in modes] == [seconds] * 3
        assert [len(visit) for visit in visits] == [1] * 5 + [7] * 2
    else:
        assert visits == [[0, 1]]
        assert [len(mode.samples) for mode in modes] == [1, 0, 1]
        assert modes[1].wall_median is None and modes[1].cpu_median is None
        assert modes[1].unmeasured_reason == "slow family not measured"
        row = FamilyResult(
            "slow",
            0.0,
            None,
            [
                replace(
                    mode,
                    samples=[
                        replace(sample, density_error=0.0) for sample in mode.samples
                    ],
                )
                for mode in modes
            ],
        )
        fast = replace(
            row,
            family="fast",
            modes=[
                ModeResult(mode.mode, [Sample(1, 1, 0, {})], 1, 1) for mode in modes
            ],
        )
        for compare in (dominance, timing_equivalents):
            compared = compare([row, fast])
            assert "warm_clear_history/wall" not in compared["slow"]
            assert compared["fast"]["warm_clear_history/wall"] == []


def test_slow_policy_has_no_separate_probe_or_warmup() -> None:
    density = np.eye(2, dtype=np.complex128)[None] / 2
    visits: list[list[int]] = []
    ticks = count()

    @contextmanager
    def session(item: Input, family: str) -> Generator[EngineSession]:
        seen: list[int] = []
        visits.append(seen)

        def evolve() -> EvolutionResult:
            seen.append(len(seen))
            return EvolutionResult(
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"t4": 1},
                    duration_ns_by_kernel={
                        name: float(count) for name, count in ({"t4": 1}).items()
                    },
                ),
            )

        def clear() -> None:
            pytest.fail("slow family must not clear history")

        yield EngineSession(evolve, clear)

    result = measure_policy(
        Input("idle", lambda: accuracy_case("idle_coherence"), "slow contract"),
        "t4",
        ReferenceResult(density, 0),
        opener=session,
        wall_clock=lambda: float(next(ticks)) * 61,
        cpu_clock=lambda: 0.0,
    )
    assert visits == [[0, 1]]
    assert [len(mode.samples) for mode in result.modes] == [1, 0, 1]


@pytest.mark.parametrize("compare", [dominance, timing_equivalents])
def test_comparisons_omit_only_the_explicitly_unmeasured_slow_mode(
    compare: Callable[[list[FamilyResult]], dict[str, dict[str, list[str]]]],
) -> None:
    slow = FamilyResult(
        "slow",
        0,
        None,
        [
            ModeResult("first_evolve", [Sample(61, 61, 0, {})], 61, 61),
            ModeResult(
                "warm_clear_history", [], None, None, "slow family not measured"
            ),
            ModeResult("warm_retain_history", [Sample(1, 1, 0, {})], 1, 1),
        ],
    )
    fast = replace(
        slow,
        family="fast",
        modes=[
            ModeResult(mode.mode, [Sample(1, 1, 0, {})], 1, 1) for mode in slow.modes
        ],
    )
    result = compare([slow, fast])
    assert "warm_clear_history/wall" not in result["slow"]
    assert result["fast"]["warm_clear_history/wall"] == []


def test_dominance_relations_distinguish_equal_and_different_companion_groups() -> None:
    policies = [
        FamilyResult(
            "gate_cache",
            1e-4,
            None,
            [],
            whitelist=("gate_cache", "multitone_rwa", "t4"),
        ),
        FamilyResult(
            "period_table",
            1e-5,
            None,
            [],
            whitelist=("period_table", "t4", "multitone_rwa"),
        ),
        FamilyResult(
            "multitone_gate",
            1e-5,
            None,
            [],
            whitelist=("multitone_gate", "single_tone_gate", "t4"),
        ),
    ]
    row = InputResult(
        "overlap",
        "physical",
        "oracle",
        0,
        policies,
        {"gate_cache": {"first_evolve/wall": ["period_table", "multitone_gate"]}},
        {},
    )
    assert row.dominance_same_companions == {
        "gate_cache": {
            "first_evolve/wall": {"period_table": True, "multitone_gate": False}
        }
    }


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
        FamilyResult(
            "blocked", None, None, [], companion_blocked="idle companion refused"
        ),
    ]
    result = dominance(rows)
    assert result["target"]["first_evolve/wall"] == ["faster"]
    assert result["target"]["first_evolve/cpu"] == []
    assert result["boundary"]["warm_clear_history/wall"] == []
    assert result["faster"]["warm_retain_history/cpu"] == ["boundary"]
    assert "default" not in result
    assert "unavailable" not in result
    assert "blocked" not in result
    equivalent = timing_equivalents(rows)
    assert equivalent["target"]["first_evolve/wall"] == ["boundary"]
    assert equivalent["target"]["first_evolve/cpu"] == ["boundary", "faster"]
    assert equivalent["boundary"]["warm_clear_history/wall"] == ["target", "faster"]


@pytest.mark.parametrize("compare", [dominance, timing_equivalents])
def test_comparisons_pair_history_modes_by_name(
    compare: Callable[[list[FamilyResult]], dict[str, dict[str, list[str]]]],
) -> None:
    target = FamilyResult(
        "target",
        0.001,
        None,
        [
            ModeResult(mode, [Sample(time, time, 0.001, {})] * 5, time, time)
            for mode, time in (
                ("first_evolve", 10),
                ("warm_clear_history", 20),
                ("warm_retain_history", 40),
            )
        ],
    )
    other = replace(target, family="other", modes=list(reversed(target.modes)))
    result = compare([target, other])
    expected = [] if compare is dominance else ["other"]
    assert result["target"] == {
        "first_evolve/wall": expected,
        "first_evolve/cpu": expected,
        "warm_clear_history/wall": expected,
        "warm_clear_history/cpu": expected,
        "warm_retain_history/wall": expected,
        "warm_retain_history/cpu": expected,
    }


@pytest.mark.parametrize("compare", [dominance, timing_equivalents])
def test_comparisons_reject_missing_history_modes(
    compare: Callable[[list[FamilyResult]], dict[str, dict[str, list[str]]]],
) -> None:
    row = FamilyResult(
        "incomplete",
        0.001,
        None,
        [ModeResult("first_evolve", [Sample(1, 1, 0.001, {})] * 5, 1, 1)],
    )
    with pytest.raises(ValueError, match="history modes"):
        compare([row])


@pytest.mark.parametrize("compare", [dominance, timing_equivalents])
@pytest.mark.parametrize("missing", ["samples", "error", "family_error"])
def test_comparisons_reject_unmeasured_available_rows(
    compare: Callable[[list[FamilyResult]], dict[str, dict[str, list[str]]]],
    missing: str,
) -> None:
    row = FamilyResult(
        "unmeasured",
        None if missing == "family_error" else 0.001,
        None,
        [
            ModeResult(
                mode,
                []
                if missing == "samples"
                else [Sample(1, 1, None if missing == "error" else 0.001, {})] * 5,
                1,
                1,
            )
            for mode in ("first_evolve", "warm_clear_history", "warm_retain_history")
        ],
    )
    with pytest.raises(ValueError, match="density error"):
        compare([row])


def test_dominance_uses_only_errors_from_the_compared_history_mode() -> None:
    target = FamilyResult(
        "target",
        0.1,
        None,
        [
            ModeResult(mode, [Sample(10, 10, error, {})] * 5, 10, 10)
            for mode, error in (
                ("first_evolve", 0.001),
                ("warm_clear_history", 0.01),
                ("warm_retain_history", 0.01),
            )
        ],
    )
    other = FamilyResult(
        "other",
        0.2,
        None,
        [
            ModeResult(mode, [Sample(time, time, error, {})] * 5, time, time)
            for mode, error, time in (
                ("first_evolve", 0.002, 1),
                ("warm_clear_history", 0.01, 8.99),
                ("warm_retain_history", 0.01, 9),
            )
        ],
    )
    assert dominance([target, other])["target"] == {
        "first_evolve/wall": [],
        "first_evolve/cpu": [],
        "warm_clear_history/wall": ["other"],
        "warm_clear_history/cpu": ["other"],
        "warm_retain_history/wall": [],
        "warm_retain_history/cpu": [],
    }


@pytest.mark.sim
def test_exclusion_estimator_preserves_policies_modes_and_observed_paths() -> None:
    results = measure_engine_cost(
        lambda: replace(accuracy_case("weak_plateau"), families=("t6",))
    )
    assert [(r.policy, r.mode) for r in results] == [
        (policy, mode)
        for policy in ("pinned", "excluded")
        for mode in ("first_evolve", "warm_clear_history", "warm_retain_history")
    ]
    for result in results:
        assert (
            len(result.wall_seconds)
            == len(result.cpu_seconds)
            == len(result.paths)
            == 5
        )
        if result.policy == "pinned":
            assert all(path.get("t6", 0) == 2 for path in result.paths)
        else:
            assert all(path and "t6" not in path for path in result.paths)


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
    saved = json.loads(output.read_text())
    refused = next(
        policy
        for policy in saved["inputs"][0]["policies"]
        if policy["family"] == "dop853"
    )
    assert refused["refusal"] == {
        "domain": "idle",
        "points": [0, 1],
        "segments": [0, 1],
        "attempted": [],
        "family_reasons": {"dop853": "not_candidate"},
    }
    refused_row = next(
        line
        for line in summary.read_text().splitlines()
        if line.startswith("| dop853 |")
    )
    assert refused["unavailable"] in refused_row
    assert "not_candidate" in refused_row
    assert "t4" in summary.read_text()
    complete = output.read_text()
    manifest.write_text("{}")
    assert main(argv, inputs=inputs) == 1
    assert output.read_text() == complete
    assert main(["--freeze-inputs", str(tmp_path)], inputs=inputs) == 1


def test_cli_selects_families_without_changing_inputs_or_default_group(
    tmp_path: Path,
) -> None:
    inputs = [
        Input(
            "plateau",
            lambda: accuracy_case("weak_plateau"),
            "tiny Engine seam",
            category="real",
        )
    ]
    manifest = tmp_path / "inputs.json"
    output = tmp_path / "matrix.json"
    summary = tmp_path / "summary.md"
    assert main(["--freeze-inputs", str(manifest)], inputs=inputs) == 0
    frozen = manifest.read_bytes()
    argv = [
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    assert main(argv, inputs=inputs) == 0
    full = json.loads(output.read_text())
    assert main([*argv, "--families", "t6,floquet_period"], inputs=inputs) == 0
    partial = json.loads(output.read_text())
    assert partial["measured_families"] == ["t6", "floquet_period"]
    assert partial["manifest_sha256"] == full["manifest_sha256"]
    assert manifest.read_bytes() == frozen
    row = partial["inputs"][0]
    assert row["physical_sha256"] == full["inputs"][0]["physical_sha256"]
    assert row["oracle_sha256"] == full["inputs"][0]["oracle_sha256"]
    assert [policy["family"] for policy in row["policies"]] == [
        "t6",
        "floquet_period",
        "default",
    ]
    assert row["fixed_default_group"]["family"] == "fixed_default_group"
    assert (
        row["fixed_default_group"]["whitelist"]
        == full["inputs"][0]["fixed_default_group"]["whitelist"]
    )
    assert set(row["dominators"]) == {"t6", "floquet_period"}
    for comparisons in (row["dominators"], row["timing_equivalents"]):
        for family, cells in comparisons.items():
            for peers in cells.values():
                assert set(peers) <= {"t6", "floquet_period"} - {family}
    report = summary.read_text()
    assert "Partial measurement" in report.split("\n\n")[1]
    assert "unmeasured, not unavailable" in report
    assert "Measured pinned families: t6, floquet_period" in report
    conclusions = report.split("## Family conclusions\n", 1)[1].split(
        "## Complete measured matrix", 1
    )[0]
    assert [line for line in conclusions.splitlines() if line.startswith("### ")] == [
        "### t6",
        "### floquet_period",
    ]


@pytest.mark.parametrize("names", ["not_a_family", "t6,not_a_family", "", "t6,"])
def test_cli_rejects_unknown_families_with_legal_names(
    names: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert main(["--families", names], inputs=[]) == 1
    assert "Unknown families" in caplog.text
    assert all(family in caplog.text for family in ALL_FAMILIES)


@pytest.mark.parametrize("phase", ["--freeze-inputs", "--prepare-references"])
def test_cli_rejects_family_selection_outside_measurement(
    phase: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = tmp_path / "inputs.json"
    output = tmp_path / "output.json"
    assert main(["--freeze-inputs", str(manifest)], inputs=[]) == 0
    output.write_text("previous output")
    phase_args = [phase, str(output)] if phase == "--freeze-inputs" else [phase]
    assert (
        main(
            [
                "--families",
                "t6",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--reference-cache",
                str(tmp_path / "references"),
                *phase_args,
            ],
            inputs=[],
        )
        == 1
    )
    assert "--families cannot be combined" in caplog.text
    assert phase in caplog.text
    assert output.read_text() == "previous output"


@pytest.mark.parametrize("attempted", [(), ("single_tone_gate",)])
def test_policy_classifies_only_the_reported_failing_group(
    attempted: tuple[str, ...],
) -> None:
    message = (
        "no valid kernel for segment 3; domain shaped: points [2, 7], segments [3, 5]"
    )
    refusal = NoValidKernel(
        message, domain="shaped", points=(2, 7), segments=(3, 5), attempted=attempted
    )
    item = Input(
        "gates",
        lambda: accuracy_case("shaped_complex"),
        "structured refusal",
        companions=("t4",),
    )

    @contextmanager
    def refused_session(item: Input, family: str) -> Generator[EngineSession]:
        def evolve() -> EvolutionResult:
            raise refusal

        yield EngineSession(evolve, lambda: None)

    result = measure_policy(
        item,
        "single_tone_gate",
        ReferenceResult(np.eye(2, dtype=np.complex128)[None] / 2, 0),
        opener=refused_session,
    )
    assert result.unavailable == message
    assert result.error is None and result.modes == []
    detail = result.refusal
    assert detail is not None
    assert detail.domain == "shaped"
    assert detail.points == (2, 7) and detail.segments == (3, 5)
    assert detail.attempted == attempted
    assert detail.family_reasons == {
        "single_tone_gate": "guard_or_precondition" if attempted else "not_candidate",
        "t4": "not_candidate",
    }


@pytest.mark.parametrize(
    "message",
    [
        "numerical integration failed",
        "no valid kernel: malformed refusal",
        "no valid kernel for segment 0; domain shaped: points [0], segments [0]",
    ],
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


def test_policy_attributes_companion_refusal_without_declaring_target_unavailable() -> (
    None
):
    item = companion_input()
    refusal = NoValidKernel(
        "first idle group refused",
        domain="idle",
        points=(0,),
        segments=(1,),
        attempted=("t4",),
    )

    @contextmanager
    def session(item: Input, family: str) -> Generator[EngineSession]:
        def evolve() -> EvolutionResult:
            raise refusal

        yield EngineSession(evolve, lambda: None)

    result = measure_policy(
        item,
        "single_tone_rwa",
        ReferenceResult(np.eye(2, dtype=np.complex128)[None] / 2, 0),
        opener=session,
    )
    assert result.unavailable is None
    assert result.companion_blocked == str(refusal)
    assert result.expected_route == item.expected_routes["single_tone_rwa"]
    assert result.attribution == "target_and_companions"
    assert result.error is None and result.modes == []
    assert result.refusal is not None
    assert result.refusal.attempted == ("t4",)


def test_policy_does_not_allow_an_unpinned_fallback_to_masquerade_as_success() -> None:
    density = np.eye(2, dtype=np.complex128)[None] / 2
    item = Input(
        "idle", lambda: accuracy_case("idle_coherence"), "synthetic wrong route"
    )

    @contextmanager
    def wrong_session(item: Input, family: str) -> Generator[EngineSession]:
        yield EngineSession(
            lambda: EvolutionResult(
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"t6": 1},
                    duration_ns_by_kernel={
                        name: float(count) for name, count in ({"t6": 1}).items()
                    },
                ),
            ),
            lambda: None,
        )

    with pytest.raises(RuntimeError, match="pinned t4 did not execute exclusively"):
        measure_policy(item, "t4", ReferenceResult(density, 0), opener=wrong_session)


def test_policy_uses_target_companions_instead_of_default_and_freezes_them() -> None:
    assert companion_input().companions_by_target == {}
    item = Input(
        "idle",
        lambda: accuracy_case("idle_coherence"),
        "target override excludes the otherwise declared driven companion",
        companions=("dop853",),
        companions_by_target={"t4": ()},
    )
    with closing((case := item.build()).source):
        reference = converged_reference(case.source, case.batch)
    result = measure_policy(item, "t4", reference)
    assert result.whitelist == ("t4",)
    assert result.attribution == "target"
    assert all(
        set(sample.paths) == {"t4"} for mode in result.modes for sample in mode.samples
    )
    manifest = json.loads(canonical_json(freeze_manifest([item])))
    assert manifest["inputs"][0]["companions_by_target"] == {"t4": []}


def companion_input() -> Input:
    return Input(
        "companion",
        lambda: accuracy_case("linear_plateau"),
        "group route",
        companions=("t4",),
        expected_routes={
            "single_tone_rwa": {"single_tone_rwa": (1, 10.0), "t4": (1, 2.0)}
        },
    )


@pytest.mark.parametrize("mismatch", [None, "count", "duration"])
def test_companion_route_checks_counts_lengths_and_preserves_family_cpu(
    mismatch: str | None,
) -> None:
    item = companion_input()
    density = np.eye(2, dtype=np.complex128)[None] / 2
    count = 2 if mismatch == "count" else 1
    duration = 11.0 if mismatch == "duration" else 10.0
    reports = {
        "single_tone_rwa": FamilyRouting(
            rows=count, duration_ns=duration, prepare_cpu_s=0.3, apply_cpu_s=0.4
        ),
        "t4": FamilyRouting(
            rows=1, duration_ns=2.0, prepare_cpu_s=0.1, apply_cpu_s=0.2
        ),
    }

    @contextmanager
    def session(item: Input, family: str) -> Generator[EngineSession]:
        yield EngineSession(
            lambda: EvolutionResult(
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"single_tone_rwa": count, "t4": 1},
                    duration_ns_by_kernel={"single_tone_rwa": duration, "t4": 2.0},
                    routing=reports,
                ),
            ),
            lambda: None,
        )

    if mismatch is not None:
        with pytest.raises(RuntimeError, match="expected route"):
            measure_policy(
                item, "single_tone_rwa", ReferenceResult(density, 0), opener=session
            )
    else:
        result = measure_policy(
            item, "single_tone_rwa", ReferenceResult(density, 0), opener=session
        )
        sample = result.modes[0].samples[0]
        assert sample.routing == reports
        assert sample.family_cpu_s == pytest.approx({"single_tone_rwa": 0.7, "t4": 0.3})
        assert sample.family_cpu_shares == pytest.approx(
            {"single_tone_rwa": 0.7, "t4": 0.3}
        )
        assert result.expected_route == item.expected_routes["single_tone_rwa"]


@pytest.mark.parametrize(
    ("family", "attribution", "whitelist", "companion_role", "diagnostic_role"),
    [
        (
            "period_table",
            "target_and_companions",
            ("period_table", "t4"),
            "companion",
            "preparation diagnostic",
        ),
        (
            "dop853",
            "target_and_companions",
            ("dop853", "t4"),
            "companion",
            "preparation diagnostic",
        ),
        (
            "period_table",
            "target",
            ("period_table",),
            "preparation diagnostic",
            "preparation diagnostic",
        ),
        ("default", "default", None, "group member", "group member"),
        (
            "fixed_default_group",
            "fixed_family_group",
            ("period_table", "t4"),
            "group member",
            "group member",
        ),
    ],
)
def test_summary_preserves_diagnostic_cpu_without_calling_it_a_companion(
    family: str,
    attribution: str,
    whitelist: tuple[str, ...] | None,
    companion_role: str,
    diagnostic_role: str,
) -> None:
    target = family if attribution.startswith("target") else "period_table"
    reports = {
        target: FamilyRouting(
            rows=2, duration_ns=10, prepare_cpu_s=0.2, apply_cpu_s=0.3
        ),
        "t4": (
            FamilyRouting(prepare_cpu_s=0.2)
            if attribution == "target"
            else FamilyRouting(
                rows=1, duration_ns=2, prepare_cpu_s=0.1, apply_cpu_s=0.1
            )
        ),
        "single_tone_gate": FamilyRouting(rows=0, duration_ns=0, prepare_cpu_s=0.3),
    }
    sample = Sample(
        9,
        8,
        1e-4,
        {name: report.rows for name, report in reports.items() if report.rows},
        routing=reports,
    )
    policy = FamilyResult(
        family,
        1e-4,
        None,
        [ModeResult("first_evolve", [sample], 9, 8)],
        attribution=attribution,
        whitelist=whitelist,
    )
    row = InputResult("synthetic", "physical", "oracle", 0, [policy], {family: {}}, {})
    if attribution == "fixed_family_group":
        row = replace(
            row,
            policies=[
                replace(policy, family="default", attribution="default", whitelist=None)
            ],
            fixed_default_group=policy,
        )
    rendered = render_summary([row], families=(target,))
    cells = {
        fields[2]: fields
        for line in rendered.splitlines()
        if line.startswith(f"| {family} | first_evolve |")
        for fields in [[cell.strip() for cell in line.strip("|").split("|")]]
    }
    assert cells[target][9] == (
        "target" if attribution.startswith("target") else "group member"
    )
    assert cells["t4"][9] == companion_role
    diagnostic = cells["single_tone_gate"]
    assert diagnostic[9] == diagnostic_role
    assert diagnostic[3:6] == ["1", "0..0", "0..0"]
    assert [float(value) for value in diagnostic[6:9]] == pytest.approx([0.3, 0, 0.3])
    assert {name: float(fields[10]) for name, fields in cells.items()} == pytest.approx(
        {target: 0.5, "t4": 0.2, "single_tone_gate": 0.3}
    )


def test_saved_summary_restores_routing_reports_and_measured_family_selection() -> None:
    sample = Sample(
        9,
        8,
        1e-4,
        {"period_table": 2, "t4": 1},
        routing={
            "period_table": FamilyRouting(
                rows=2, duration_ns=10, prepare_cpu_s=0.2, apply_cpu_s=0.3
            ),
            "t4": FamilyRouting(
                rows=1, duration_ns=2, prepare_cpu_s=0.1, apply_cpu_s=0.1
            ),
            "single_tone_gate": FamilyRouting(prepare_cpu_s=0.3),
        },
    )
    policy = FamilyResult(
        "period_table",
        1e-4,
        None,
        [ModeResult("first_evolve", [sample], 9, 8)],
        expected_route={"period_table": (2, 10), "t4": (1, 2)},
        attribution="target_and_companions",
        whitelist=("period_table", "t4"),
    )
    row = InputResult(
        "synthetic", "physical", "oracle", 0, [policy], {"period_table": {}}, {}
    )
    saved = canonical_json(
        {
            "inputs": [
                {
                    **asdict(row),
                    "dominance_same_companions": row.dominance_same_companions,
                }
            ],
            "measured_families": ["period_table"],
        }
    )
    rendered = family_pareto_report.render_saved_summary(saved)
    # Compare dynamic table data, not prose or a fixed Markdown snapshot.
    direct = render_summary([row], families=("period_table",))
    assert [line for line in rendered.splitlines() if line.startswith("| ")] == [
        line for line in direct.splitlines() if line.startswith("| ")
    ]
    assert [line for line in rendered.splitlines() if line.startswith("### ")] == [
        "### period_table",
        "### synthetic",
    ]


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
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"t4": 1},
                    duration_ns_by_kernel={
                        name: float(count) for name, count in ({"t4": 1}).items()
                    },
                ),
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
        return FamilyResult(
            family,
            0,
            None,
            [
                ModeResult(mode, [Sample(1, 1, 0, {})] * 5, 1, 1)
                for mode in (
                    "first_evolve",
                    "warm_clear_history",
                    "warm_retain_history",
                )
            ],
        )

    rows = run_table(
        inputs, ("single_tone_gate",), reference_solver=solve, policy_runner=policy
    )
    assert len(rows) == 2
    assert calls == 1


def test_table_times_the_defaults_exact_family_union_separately() -> None:
    paths = {"single_tone_rwa": 1, "t4": 1}

    def solve(source: ModelSource, batch: TimelineBatch) -> ReferenceResult:
        return ReferenceResult(
            np.tile(source.initial_density, (len(batch.points), 1, 1)), 0
        )

    def policy(item: Input, family: str, reference: ReferenceResult) -> FamilyResult:
        return FamilyResult(
            family,
            0,
            None,
            [
                ModeResult(mode, [Sample(1, 1, 0, paths)], 1, 1)
                for mode in (
                    "first_evolve",
                    "warm_clear_history",
                    "warm_retain_history",
                )
            ],
        )

    (row,) = run_table(
        [replace(companion_input(), category="real")],
        ("single_tone_rwa",),
        reference_solver=solve,
        policy_runner=policy,
    )
    assert row.fixed_default_group is not None
    assert row.fixed_default_group.whitelist == ("single_tone_rwa", "t4")
    assert row.fixed_default_group.family == "fixed_default_group"
    assert set(row.dominators) == {"single_tone_rwa"}


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
                np.array([0]),
                density,
                PathDiagnostics(
                    segments_by_kernel={"t4": 1},
                    duration_ns_by_kernel={
                        name: float(count) for name, count in ({"t4": 1}).items()
                    },
                ),
            ),
            lambda: None,
        )
        raise error

    with pytest.raises(RuntimeError) as caught:
        measure_policy(item, "t4", ReferenceResult(density, 0), opener=cleanup_fails)
    assert caught.value is error
