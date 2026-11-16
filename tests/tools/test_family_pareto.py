"""Public Engine-only measurement and comparison contracts, without real timers."""

import json
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    EvolutionResult,
    FamilyRouting,
    NoValidKernel,
    PathDiagnostics,
    ReferenceResult,
    RoutingHistory,
    write_reference,
)

from tests.tools.family_pareto_support import (
    engine_reference,
    make_history_scene,
    make_scene,
    write_test_cache,
    write_test_index,
)
from tools import family_pareto, family_pareto_report
from tools.family_pareto import (
    EngineSession,
    main,
    measure_family,
    measure_policy,
    render_summary,
    run_table,
)
from tools.family_pareto import (
    tested_state as read_tested_state,
)
from tools.family_pareto_inputs import Input, Scene, canonical_json, freeze_records
from tools.family_pareto_results import (
    ACCURACY_LIMIT,
    ALL_FAMILIES,
    FamilyResult,
    InputResult,
    ModeResult,
    Refusal,
    Sample,
    dominance,
    timing_equivalents,
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
        ReferenceResult(density, 0.0),
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
        Input("idle", lambda: make_scene("idle_coherence"), "slow contract"),
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


def test_dominance_uses_final_qualification_and_excludes_directly_retained() -> None:
    target = FamilyResult(
        "target",
        1e-6,
        None,
        [
            ModeResult(mode, [Sample(10, 10, error, {})] * 5, 10, 10)
            for mode, error in (
                ("first_evolve", 0.001),
                ("warm_clear_history", 1e-6),
                ("warm_retain_history", 1e-6),
            )
        ],
    )
    other = FamilyResult(
        "other",
        2e-3,
        None,
        [
            ModeResult(mode, [Sample(time, time, error, {})] * 5, time, time)
            for mode, error, time in (
                ("first_evolve", 0.002, 1),
                ("warm_clear_history", 2e-3, 8.99),
                ("warm_retain_history", 2e-3, 9),
            )
        ],
    )
    excluded = [
        replace(other, family=name)
        for name in (
            "t4",
            "adiabatic_map",
            "flux_dop853",
            "slow_envelope",
            "t6",
            "dop853",
            "default",
            "fixed_default_group",
        )
    ]
    # One bad final result disqualifies the entire policy, even if a mode passed.
    over_limit = replace(other, family="over_limit", error=2e-3 + 1e-12)
    result = dominance([target, other, over_limit, *excluded])
    assert set(result) == {"target", "other"}
    assert result["target"] == {
        "first_evolve/wall": ["other"],
        "first_evolve/cpu": ["other"],
        "warm_clear_history/wall": ["other"],
        "warm_clear_history/cpu": ["other"],
        "warm_retain_history/wall": [],
        "warm_retain_history/cpu": [],
    }


@pytest.mark.parametrize(
    ("error", "reason", "blocked", "expected"),
    [
        (2e-3, None, None, "qualified"),
        (2.0001e-3, None, None, "over_limit"),
        (None, "guard_or_precondition", None, "guard_or_precondition"),
        (None, "not_candidate", None, "not_candidate"),
        (None, "guard_or_precondition", "companion refused", "companion_blocked"),
    ],
)
def test_family_status_owns_accuracy_and_companion_priority(
    error: float | None,
    reason: Literal["guard_or_precondition", "not_candidate"] | None,
    blocked: str | None,
    expected: str,
) -> None:
    refusal = (
        None
        if reason is None
        else Refusal("plateau", (0,), (0,), ("t6",), {"t6": reason})
    )
    policy = FamilyResult(
        "t6",
        error,
        None if reason is None else "refused",
        [],
        refusal=refusal,
        companion_blocked=blocked,
    )
    assert policy.status == expected


@pytest.mark.sim
def test_real_engine_pins_path_and_reports_domain_refusal_without_zero_values() -> None:
    item = Input("idle", lambda: make_scene("idle_coherence"), "tiny Engine seam")
    case = item.build()
    try:
        reference = engine_reference(case)
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
def test_cli_measures_prepared_inputs_and_preserves_output_on_manifest_failure(
    tmp_path: Path,
) -> None:
    inputs = [Input("idle", lambda: make_scene("idle_coherence"), "tiny Engine seam")]
    manifest = tmp_path / "inputs.json"
    output = tmp_path / "result.json"
    summary = tmp_path / "summary.md"
    write_test_index(inputs, manifest)
    cache = tmp_path / "cache"
    write_test_cache(inputs, cache)
    argv = [
        "--reference-cache",
        str(cache),
        "--index",
        str(manifest),
        "--scene-cache",
        str(tmp_path / "scenes"),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    assert main(argv) == 0
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
    assert main(argv) == 1
    assert output.read_text() == complete


def test_cli_scenario_selects_exactly_one_before_build_with_a_manifest(
    tmp_path: Path,
) -> None:
    def must_not_build() -> Scene:
        raise AssertionError("unselected input was built")

    inputs = [
        Input("unused", must_not_build, "must remain unbuilt"),
        Input("idle", lambda: make_scene("idle_coherence"), "tiny scenario"),
    ]
    cache = tmp_path / "cache"
    write_test_cache([inputs[1]], cache)
    manifest = tmp_path / "inputs.json"
    write_test_index([inputs[1]], manifest)
    matrix = tmp_path / "matrix.json"
    report = tmp_path / "report.md"
    assert (
        main(
            [
                "--scenario",
                "idle",
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--families",
                "t4",
                "--reference-cache",
                str(cache),
                "--output",
                str(matrix),
                "--summary",
                str(report),
                "--exclusive",
                "--partial-output",
                str(tmp_path / "journal.jsonl"),
            ],
        )
        == 0
    )
    saved = json.loads(matrix.read_text())
    assert [item["name"] for item in saved["records"]] == ["idle"]
    assert saved["measurement"]["reference_cache_only"] is True
    assert saved["exclusive"] is True
    assert datetime.fromisoformat(saved["started_at_utc"]).utcoffset() == timedelta(0)
    assert saved["inputs"][0]["policies"][0]["error"] < 2e-3


def test_cli_scenario_cache_miss_stops_without_solving_and_explains_rebuild(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    matrix = tmp_path / "matrix.json"
    cache = tmp_path / "empty-cache"
    manifest = tmp_path / "scenes.json.gz"
    write_test_index(
        [Input("idle", lambda: make_scene("idle_coherence"), "tiny")], manifest
    )
    assert (
        main(
            [
                "--scenario",
                "idle",
                "--families",
                "t4",
                "--reference-cache",
                str(cache),
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--output",
                str(matrix),
                "--summary",
                str(tmp_path / "report.md"),
                "--partial-output",
                str(tmp_path / "journal.jsonl"),
            ],
        )
        == 1
    )
    assert not matrix.exists()
    assert not list(cache.glob("*.npz"))
    assert "reference cache miss" in caplog.text
    assert "tests-side reference producer" in caplog.text


def test_cli_scenario_unknown_name_does_not_build_and_lists_legal_names(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = tmp_path / "scenes.json.gz"
    write_test_index(
        [Input("idle", lambda: make_scene("idle_coherence"), "tiny")], manifest
    )
    assert main(["--scenario", "unknown", "--index", str(manifest)]) == 1
    assert "Unknown scenario: unknown. Legal names: idle" in caplog.text


@pytest.mark.parametrize("selection", [[], ["--scenario", "idle"]])
def test_cli_rejects_ambiguous_scenario_before_build_and_lists_sources(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    selection: list[str],
) -> None:
    inputs = [Input("idle", lambda: make_scene("idle_coherence"), "tiny")]
    paths = [tmp_path / "first.json.gz", tmp_path / "second.json.gz"]
    for path in paths:
        write_test_index(inputs, path)
    calls: list[str] = []

    def engine(*args: object, **kwargs: object) -> None:
        calls.append("Engine")
        raise AssertionError("ambiguous selection must not start Engine")

    # CLI has one index option; substitute only default location discovery so
    # the real global selection rule reads two independent valid fixture indexes.
    monkeypatch.setattr(family_pareto, "scene_index_paths", lambda index: tuple(paths))
    monkeypatch.setattr(family_pareto, "Engine", engine)
    monkeypatch.setattr(family_pareto, "measure_policy", engine)
    assert main(selection) == 1
    assert "Ambiguous scenario: idle" in caplog.text
    assert all(str(path) in caplog.text for path in paths)
    assert calls == []


@pytest.mark.parametrize(
    "baseline_kind", ["fingerprint_mismatch", "malformed", "missing"]
)
def test_cli_summary_comparison_never_blocks_writing_current_measurement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], baseline_kind: str
) -> None:
    inputs = [Input("idle", lambda: make_scene("idle_coherence"), "tiny")]
    cache = tmp_path / "cache"
    write_test_cache(inputs, cache)
    manifest = tmp_path / "scenes.json.gz"
    write_test_index(inputs, manifest)
    destination = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    if baseline_kind == "fingerprint_mismatch":
        baseline.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "idle",
                    "manifest_sha256": "old-manifest",
                    "physical_sha256": "old-physical",
                    "oracle_sha256": "old-oracle",
                    "measured_families": ["t4"],
                    "policies": {},
                    "provenance": {
                        "head": "old",
                        "tree": "old",
                        "dirty": False,
                        "started_at_utc": "2026-11-12T00:00:00+00:00",
                        "exclusive": True,
                    },
                }
            )
        )
    elif baseline_kind == "malformed":
        baseline.write_text("{}")
    assert (
        main(
            [
                "--scenario",
                "idle",
                "--families",
                "t4",
                "--reference-cache",
                str(cache),
                "--output",
                str(tmp_path / "matrix.json"),
                "--summary",
                str(tmp_path / "report.md"),
                "--partial-output",
                str(tmp_path / "journal.jsonl"),
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--summary-output",
                str(destination),
                "--compare-summary",
                str(baseline),
            ],
        )
        == 0
    )
    saved = json.loads(destination.read_text())
    assert saved["measured_families"] == ["t4"]
    assert saved["policies"]["t4"]["status"] == "qualified"
    assert saved["provenance"]["exclusive"] is False
    assert saved["provenance"]["dirty"] == read_tested_state()["dirty"]
    assert destination.with_suffix(".md").exists()
    assert (
        "Not comparable"
        if baseline_kind == "fingerprint_mismatch"
        else "Comparison unavailable"
    ) in capsys.readouterr().out


def test_cli_selects_families_without_changing_inputs_or_default_group(
    tmp_path: Path,
) -> None:
    inputs = [
        Input(
            "plateau",
            lambda: make_scene("weak_plateau"),
            "tiny Engine seam",
            category="real",
        )
    ]
    manifest = tmp_path / "inputs.json"
    output = tmp_path / "matrix.json"
    summary = tmp_path / "summary.md"
    write_test_index(inputs, manifest)
    cache = tmp_path / "cache"
    write_test_cache(inputs, cache)
    frozen = manifest.read_bytes()
    argv = [
        "--reference-cache",
        str(cache),
        "--index",
        str(manifest),
        "--scene-cache",
        str(tmp_path / "scenes"),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ]
    assert main(argv) == 0
    full = json.loads(output.read_text())
    assert main([*argv, "--families", "t6,floquet_period"]) == 0
    partial = json.loads(output.read_text())
    assert partial["measured_families"] == ["t6", "floquet_period"]
    assert partial["records"] == full["records"]
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
    assert set(row["dominators"]) == {"floquet_period"}
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
    assert main(["--families", names]) == 1
    assert "Unknown families" in caplog.text
    assert all(family in caplog.text for family in ALL_FAMILIES)


def test_cli_rejects_family_selection_outside_measurement(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = tmp_path / "inputs.json"
    output = tmp_path / "output.json"
    write_test_index([], manifest)
    output.write_text("previous output")
    assert (
        main(
            [
                "--families",
                "t6",
                "--index",
                str(manifest),
                "--scene-cache",
                str(tmp_path / "scenes"),
                "--output",
                str(output),
                "--reference-cache",
                str(tmp_path / "references"),
                "--check-references",
            ],
        )
        == 1
    )
    assert "--families cannot be combined" in caplog.text
    assert "--check-references" in caplog.text
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
        lambda: make_scene("shaped_complex"),
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
    ("result_motion", "reference_motion", "reject"),
    [
        (0.0, 0.25, True),
        (1e-14, 0.25, True),
        (0.0, 0.0, False),
        (0.0, ACCURACY_LIMIT / 2, False),
        (2e-9, 0.25, False),
    ],
)
def test_policy_rejects_initial_density_only_when_reference_moved(
    result_motion: float,
    reference_motion: float,
    reject: bool,
) -> None:
    item = Input("shaped_no_op", lambda: make_scene("shaped_complex"), "no-op fake")
    with closing(item.build().source) as source:
        initial_density = source.initial_density
    density = np.repeat(initial_density[None], 2, axis=0)
    reference_density = density.copy()
    density[1, 0, 0] += result_motion
    reference_density[1, 0, 0] += reference_motion
    reference = ReferenceResult(reference_density, 0)

    @contextmanager
    def unchanged_session(item: Input, family: str) -> Generator[EngineSession]:
        yield EngineSession(
            lambda: EvolutionResult(
                np.array([0, 1]),
                density,
                PathDiagnostics(
                    segments_by_kernel={family: 1},
                    duration_ns_by_kernel={family: 1.0},
                ),
            ),
            lambda: None,
            initial_density=initial_density,
        )

    if reject:
        with pytest.raises(
            RuntimeError,
            match=(
                r"(?s)point 1: result moved .*reference moved 0.25"
                r".*shaped_no_op.*dop853"
            ),
        ):
            measure_policy(item, "dop853", reference, opener=unchanged_session)
    else:
        result = measure_policy(item, "dop853", reference, opener=unchanged_session)
        assert result.status == (
            "qualified" if reference_motion <= ACCURACY_LIMIT else "over_limit"
        )


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
    item = Input("idle", lambda: make_scene("idle_coherence"), "synthetic failure")

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
    item = Input("idle", lambda: make_scene("idle_coherence"), "synthetic wrong route")

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
        lambda: make_scene("idle_coherence"),
        "target override excludes the otherwise declared driven companion",
        companions=("dop853",),
        companions_by_target={"t4": ()},
    )
    with closing((case := item.build()).source):
        reference = engine_reference(case)
    result = measure_policy(item, "t4", reference)
    assert result.whitelist == ("t4",)
    assert result.attribution == "target"
    assert all(
        set(sample.paths) == {"t4"} for mode in result.modes for sample in mode.samples
    )
    (record,) = freeze_records([item])
    metadata = json.loads(canonical_json(record.metadata))
    assert metadata["companions_by_target"] == {"t4": []}


def companion_input() -> Input:
    return Input(
        "companion",
        lambda: make_scene("linear_plateau"),
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
        [
            ModeResult(mode, [sample], 9, 8)
            for mode in ("first_evolve", "warm_clear_history", "warm_retain_history")
        ],
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


def test_saved_report_recomputes_qualified_dominance() -> None:
    def policy(name: str, error: float, seconds: float) -> FamilyResult:
        return FamilyResult(
            name,
            error,
            None,
            [
                ModeResult(
                    mode, [Sample(seconds, seconds, error, {})], seconds, seconds
                )
                for mode in (
                    "first_evolve",
                    "warm_clear_history",
                    "warm_retain_history",
                )
            ],
        )

    policies = [
        policy("period_table", 1e-6, 10),
        policy("shaped_sambe", 2e-3, 1),
        policy("gate_cache", 0.01, 0.1),
        policy("dop853", 1e-8, 0.01),
    ]
    stored = {
        p.family: {
            f"{mode.mode}/{metric}": []
            for mode in p.modes
            for metric in ("wall", "cpu")
        }
        for p in policies
    }
    row = InputResult("synthetic", "physical", "oracle", 0, policies, stored, {})
    saved = canonical_json(
        {"inputs": [asdict(row)], "measured_families": [p.family for p in policies]}
    )
    rendered = family_pareto_report.render_saved_summary(saved)
    cells = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in rendered.splitlines()
        if line.startswith("| period_table |")
    ]
    assert len(cells) == 6
    assert {cell[7] for cell in cells} == {"shaped_sambe (same companions)"}


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


def test_table_shares_oracle_for_identical_physics_with_different_model_identities(
    tmp_path: Path,
) -> None:
    references: list[ReferenceResult] = []
    inputs = [
        Input("linear", lambda: make_scene("linear_shaped"), "original"),
        Input(
            "history",
            lambda: make_history_scene(2, RoutingHistory()),
            "same physics, different model identity",
        ),
    ]

    case = inputs[0].build()
    with closing(case.source):
        write_reference(
            tmp_path,
            case.source,
            case.batch,
            ReferenceResult(
                np.tile(case.source.initial_density, (len(case.batch.points), 1, 1)), 0
            ),
        )

    def policy(item: Input, family: str, reference: ReferenceResult) -> FamilyResult:
        references.append(reference)
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
        inputs, ("single_tone_gate",), reference_cache=tmp_path, policy_runner=policy
    )
    assert len(rows) == 2
    assert len(references) == 4
    assert all(reference is references[0] for reference in references)


def test_table_times_the_defaults_exact_family_union_separately(tmp_path: Path) -> None:
    paths = {"single_tone_rwa": 1, "t4": 1}
    inputs = [replace(companion_input(), category="real")]
    write_test_cache(inputs, tmp_path)

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
        inputs,
        ("single_tone_rwa",),
        reference_cache=tmp_path,
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
    item = Input("idle", lambda: make_scene("idle_coherence"), "cleanup isolation")

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
