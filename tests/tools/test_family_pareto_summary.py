"""Scenario summaries preserve policy outcomes without raw timing samples."""

from dataclasses import replace
from pathlib import Path

import pytest

from tools import family_pareto_summary as summaries
from tools.family_pareto_results import (
    FamilyResult,
    InputResult,
    ModeResult,
    Refusal,
    Sample,
)


def scenario() -> InputResult:
    """Return a tiny completed slow policy and separately timed default/group."""
    modes = [
        ModeResult("first_evolve", [Sample(61, 60, 1e-4, {"t6": 1})], 61, 60),
        ModeResult("warm_clear_history", [], None, None, "slow family not measured"),
        ModeResult("warm_retain_history", [Sample(2, 1, 2e-4, {"t6": 1})], 2, 1),
    ]
    target = FamilyResult(
        "t6",
        2e-4,
        None,
        modes,
        attribution="target_and_companions",
        whitelist=("t6", "t4"),
    )
    return InputResult(
        "tiny",
        "physical",
        "oracle",
        1e-9,
        [
            target,
            replace(target, family="default", attribution="default", whitelist=None),
        ],
        {},
        {},
        fixed_default_group=replace(
            target, family="fixed_default_group", attribution="fixed_family_group"
        ),
    )


def test_summary_refusal_has_explicit_unmeasured_modes_and_requires_public_facts() -> (
    None
):
    row = scenario()
    refusal = FamilyResult(
        "t6",
        None,
        "first group refused",
        [],
        refusal=Refusal(
            "plateau", (0,), (0,), ("t6",), {"t6": "guard_or_precondition"}
        ),
    )
    summary = summaries.build_summary(
        replace(row, policies=[refusal], fixed_default_group=None),
        manifest_sha256="manifest",
        measured_families=("t6",),
        provenance=summaries.Provenance(
            "head", "tree", False, "2026-11-13T00:00:00+00:00", True
        ),
    )
    policy = summary.policies["t6"]
    assert policy.status == "guard_or_precondition"
    assert policy.error is None
    assert set(policy.modes) == {
        "first_evolve",
        "warm_clear_history",
        "warm_retain_history",
    }
    assert all(
        mode.sample_count == 0
        and mode.wall_median is None
        and mode.cpu_median is None
        and mode.unmeasured_reason == "first group refused"
        for mode in policy.modes.values()
    )
    with pytest.raises(ValueError, match="missing public refusal"):
        summaries.build_summary(
            replace(
                row, policies=[replace(refusal, refusal=None)], fixed_default_group=None
            ),
            manifest_sha256="manifest",
            measured_families=("t6",),
            provenance=summary.provenance,
        )


def completed_summary() -> summaries.ScenarioSummary:
    """Build one current summary with explicit single-run provenance."""
    return summaries.build_summary(
        scenario(),
        manifest_sha256="manifest",
        measured_families=("t6",),
        provenance=summaries.Provenance(
            "head", "tree", True, "2026-11-13T01:02:03+00:00", False
        ),
    )


@pytest.mark.parametrize("mismatch", [False, True])
def test_comparison_limits_policy_intersection_and_reports_values_not_speed(
    mismatch: bool,
) -> None:
    baseline = completed_summary()
    baseline = replace(
        baseline,
        measured_families=("t6", "dyson"),
        policies={
            **baseline.policies,
            "dyson": baseline.policies["t6"],
        },
    )
    current = replace(
        completed_summary(),
        physical_sha256="changed" if mismatch else "physical",
        provenance=summaries.Provenance(
            "new-head", "new-tree", False, "2026-11-14T01:02:03+00:00", True
        ),
    )
    policy = current.policies["t6"]
    current = replace(
        current,
        policies={
            **current.policies,
            "t6": replace(
                policy,
                status="over_limit",
                error=3e-3,
                modes={
                    **policy.modes,
                    "first_evolve": replace(
                        policy.modes["first_evolve"],
                        wall_median=71,
                    ),
                },
            ),
        },
    )
    result = summaries.compare_summaries(baseline, current)
    assert result.comparable is not mismatch
    assert result.baseline_only == ("dyson",)
    assert result.current_only == ()
    assert result.metadata_changes["head"].before == "head"
    assert result.metadata_changes["head"].after == "new-head"
    assert result.metadata_changes["exclusive"].after is True
    if mismatch:
        assert result.fingerprint_differences == ("physical_sha256",)
        assert result.policies == {}
    else:
        assert set(result.policies) == {"t6", "default", "fixed_default_group"}
        assert result.policies["t6"]["status"].before == "qualified"
        assert result.policies["t6"]["status"].after == "over_limit"
        assert result.policies["t6"]["error"].before == 2e-4
        assert result.policies["t6"]["error"].after == 3e-3
        assert result.policies["t6"]["first_evolve.wall_median"].delta == 10
        assert result.policies["t6"]["warm_clear_history.wall_median"].delta is None
    rendered = summaries.render_comparison(result)
    assert "dyson" in rendered
    assert "new-head" in rendered


def test_summary_round_trip_preserves_unmeasured_modes_and_policy_ownership(
    tmp_path: Path,
) -> None:
    row = scenario()
    summary = summaries.build_summary(
        row,
        manifest_sha256="manifest",
        measured_families=("t6",),
        provenance=summaries.Provenance(
            "head", "tree", True, "2026-11-13T01:02:03+00:00", False
        ),
    )
    target = summary.policies["t6"]
    assert target.status == "qualified"
    assert target.error == 2e-4
    assert target.companions == ("t4",)
    assert target.whitelist == ("t6", "t4")
    assert target.attribution == "target_and_companions"
    assert target.modes["first_evolve"].sample_count == 1
    assert target.modes["first_evolve"].cpu_median == 60
    skipped = target.modes["warm_clear_history"]
    assert (skipped.sample_count, skipped.wall_median, skipped.cpu_median) == (
        0,
        None,
        None,
    )
    assert skipped.unmeasured_reason == "slow family not measured"
    assert summary.policies["default"].attribution == "default"
    assert summary.policies["fixed_default_group"].attribution == "fixed_family_group"
    destination = tmp_path / "nested" / "tiny.json"
    summaries.write_summary(summary, destination)
    assert summaries.read_summary(destination) == summary
    markdown = destination.with_suffix(".md").read_text()
    assert "qualified" in markdown
    assert "slow family not measured" in markdown
    assert "2026-11-13T01:02:03+00:00" in markdown
