"""Speed judgement uses declared workloads, published Results and measured telemetry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from zcu_tools.gui.form import FormErrors

from tools.experiment_wall_time import (
    Inventory,
    Run,
    RunObservation,
    WallTimeCase,
    main,
)


@dataclass(frozen=True)
class _PublishedRun:
    published_result: object

    def start(self) -> RunObservation:
        return RunObservation(0.0, "2026-11-23T00:00:00+00:00")

    def wait_until(
        self, _deadline: float, _interrupted: Callable[[], bool], *, idle: bool = False
    ) -> RunObservation:
        return RunObservation(200.0, "2026-11-23T00:03:20+00:00")

    def cancel(self) -> None:
        pytest.fail("an on-time Run should not be cancelled")

    def failure(self) -> None:
        return None

    def result(self) -> object:
        return self.published_result

    def is_idle(self) -> bool:
        return True


def _run_case(
    root: Path,
    scene: dict[str, object],
    result: object,
    records: list[dict[str, object]],
) -> Path:
    inputs, output = root / "inputs", root / "output"
    (inputs / "fixture").mkdir(parents=True)
    (inputs / "ftf-settings.yaml").write_text("settings")
    (inputs / "board-profile.yaml").write_text("profile")
    (inputs / "case-inputs.json").write_text(
        json.dumps(
            {
                "entry": "demo",
                "point": "initial",
                "scenes": [{"scene_id": "fake/run", "adapter": "fake/run", **scene}],
            }
        )
    )

    def measure(
        _case: WallTimeCase,
        _inputs: Path,
        telemetry: Path,
        record_attempt: Callable[[Run | FormErrors], None],
    ) -> None:
        (telemetry / "routing-test.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        record_attempt(_PublishedRun(result))

    assert (
        main(
            ["--output", str(output)],
            data_dir=inputs,
            discover=lambda: Inventory({"fake/run"}, measure),
            read_commit=lambda: "abc",
        )
        == 0
    )
    return output


def _acquire(core_s: float) -> dict[str, object]:
    return {
        "families": {},
        "points": 5,
        "rounds": 8,
        "wall_s": max(3.0, core_s + 1.0),
        "compile_wall_s": 0.0,
        "evolve_wall_s": core_s,
        "readout_wall_s": 0.0,
        "materialize_wall_s": 0.0,
    }


@pytest.mark.parametrize(
    ("core_s", "verdict"),
    [(180.0, "pass"), (180.0001, "over_target")],
    ids=["at-target", "just-over-target"],
)
def test_main_judges_shots_by_unrounded_total_core_not_per_point(
    tmp_path: Path, core_s: float, verdict: str
) -> None:
    output = _run_case(
        tmp_path, {"metric": {"kind": "shots"}}, None, [_acquire(core_s)]
    )
    [row] = json.loads((output / "report.json").read_text())["rows"]
    assert row["core_judgement"] == verdict
    assert row["point_count"] is None
    assert row["core_s_per_point"] is None


def test_main_judges_scan_core_per_result_grid_point_not_per_round(
    tmp_path: Path,
) -> None:
    output = _run_case(
        tmp_path,
        {
            "metric": {
                "kind": "scan",
                "axes": [{"field": "sweep"}, {"field": "coordinates", "dimension": 1}],
            },
        },
        # Two simultaneous channels share five coordinates. Channel is not a sweep.
        SimpleNamespace(sweep=np.arange(3), coordinates=np.zeros((2, 5))),
        [_acquire(1.875)],
    )

    [row] = json.loads((output / "report.json").read_text())["rows"]
    assert row["point_count"] == 15
    assert row["core_s_per_point"] == 0.125
    assert row["core_judgement"] == "over_target"
    assert "| 15 | 0.125 | over_target |" in (output / "summary.md").read_text()


@pytest.mark.parametrize(
    "acquired", [True, False], ids=["two-candidates", "no-acquires"]
)
def test_main_counts_acquired_evaluations_not_seeds_or_unmeasured_candidates(
    tmp_path: Path, *, acquired: bool
) -> None:
    counts = np.full((4, 2, 3, 3), np.nan)
    records = []
    if acquired:
        counts[0, :] = 1.0  # Two seeds for candidate 0, one seed for candidate 2.
        counts[2, 0] = 1.0
        records = [_acquire(0.0625), _acquire(0.0625), _acquire(0.0625)]
    output = _run_case(
        tmp_path,
        {
            "metric": {
                "kind": "evaluation",
                "peak_field": "peak",
                "counts_field": "sums",
            }
        },
        SimpleNamespace(peak=np.array([0.75, 1.1, 1.0, np.nan]), sums=counts),
        records,
    )

    [row] = json.loads((output / "report.json").read_text())["rows"]
    assert row["point_count"] == (2 if acquired else 0)
    assert row["core_s_per_point"] == (0.09375 if acquired else None)
    assert row["core_judgement"] == ("pass" if acquired else "undetermined")
    assert row["rejected_evaluations"] == 1
    assert row["unreached_evaluations"] == 1


@pytest.mark.parametrize("records", [2, 1], ids=["matched", "unmatched"])
def test_main_requires_one_acquire_per_entered_candidate_for_auto_search(
    tmp_path: Path, records: int
) -> None:
    output = _run_case(
        tmp_path,
        {"metric": {"kind": "evaluation", "single_acquire_candidate_field": "params"}},
        SimpleNamespace(
            params=np.array([[1.0, 2.0], [1.5, 3.0], [np.nan, np.nan]]),
            signals=np.full(
                3, np.nan
            ),  # An attempted evaluation can have an undefined score.
        ),
        [_acquire(0.125)] * records,
    )
    [row] = json.loads((output / "report.json").read_text())["rows"]
    assert row["core_s"] == 0.125 * records
    assert row["rejected_evaluations"] == 0
    assert row["unreached_evaluations"] == 1
    if records == 2:
        assert row["point_count"] == 2
        assert row["core_s_per_point"] == 0.125
        assert row["core_judgement"] == "over_target"
        assert row["workload_issue"] is None
    else:
        assert row["point_count"] is None
        assert row["core_s_per_point"] is None
        assert row["core_judgement"] == "undetermined"
        assert (
            row["workload_issue"]
            == "Entered evaluations (2) differ from observed acquires (1)"
        )


@pytest.mark.parametrize(
    "complete", [True, False], ids=["complete-core", "lower-bound-only"]
)
def test_main_extrapolates_complete_core_separately_from_measured_speed(
    tmp_path: Path, *, complete: bool
) -> None:
    records = [{**_acquire(0.75), "rounds": 2}]
    if not complete:
        records.append({"families": {}, "wall_s": 3.0})
    output = _run_case(
        tmp_path,
        {
            "metric": {
                "kind": "scan",
                "axes": [{"field": "outer"}, {"field": "inner"}],
            },
            "overrides": [
                {"path": ["flux", "n"], "value": 3},
                {"path": ["rounds"], "value": 2},
            ],
            "extrapolation": {
                "outer_axis": {"path": ["flux", "n"], "full_points": 9},
                "rounds": {"path": ["rounds"], "full_rounds": 8},
            },
        },
        SimpleNamespace(outer=np.arange(3), inner=np.arange(5)),
        records,
    )

    report_path = output / "report.json"
    report = report_path.read_text()
    [row] = json.loads(report)["rows"]
    assert row["point_count"] == 15
    assert row["core_s"] == (0.75 if complete else None)
    assert row["core_s_per_point"] == (0.05 if complete else None)
    assert row["core_judgement"] == ("pass" if complete else "undetermined")
    assert row["extrapolated_point_count"] == (45 if complete else None)
    assert row["extrapolated_core_s"] == (9.0 if complete else None)
    assert row["extrapolated_core_s_per_point"] == (0.2 if complete else None)
    assert row["extrapolated_core_judgement"] == ("over_target" if complete else None)
    summary = (output / "summary.md").read_text()
    assert "Measured core (s)" in summary
    assert "Extrapolated core (s)" in summary
    if complete:
        assert "| 45 | 9.000 | 0.200 | over_target |" in summary

    assert main(["--report-only", "--output", str(output)]) == 0
    assert report_path.read_text() == report
