"""Capture inspector reads synthetic manifests and JSONL through main(argv)."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from zcu_tools.simulate.evolution import (
    CaptureMetadata,
    FrozenRecord,
    write_frozen_manifest,
)

from tools import inspect_simulator_capture as inspector


@pytest.fixture
def metadata() -> CaptureMetadata:
    return {
        "capture_id": "evolve-a",
        "batch_index": 0,
        "batch_count": 1,
        "budget": 1e-6,
        "families": None,
        "source_identity": "synthetic-source",
        "segment_numeric_version": "recorded-version",
        "frozen_physical_version": 1,
        "complete": False,
        "routes": [
            {
                "point": 4,
                "segment": 2,
                "start_ns": 10,
                "duration_ns": 20,
                "round": "caller",
                "prefix_run": None,
                "family": None,
                "candidates": [{"family": "floquet", "guard": "deferred"}],
            },
            {
                "point": 4,
                "segment": 2,
                "start_ns": 10,
                "duration_ns": 20,
                "round": "worker",
                "prefix_run": None,
                "family": "direct",
                "candidates": [
                    {"family": "floquet", "guard": "rejected"},
                    {"family": "direct", "guard": "accepted"},
                    {"family": "rwa", "guard": "untested_after_short_circuit"},
                ],
            },
            {
                "point": 8,
                "segment": 3,
                "start_ns": 30,
                "duration_ns": 40,
                "round": "caller",
                "prefix_run": None,
                "family": None,
                "candidates": [{"family": "floquet", "guard": "deferred"}],
            },
        ],
        "sampled_rwa_admissions": [],
        "prefix_runs": [],
    }


def write_capture(path: Path, *items: CaptureMetadata) -> None:
    """Write solver-free synthetic records in the public manifest container."""
    write_frozen_manifest(
        path,
        [
            FrozenRecord(
                f"{item['capture_id']}-{item['batch_index']}",
                {"models": []},
                item,
            )
            for item in items
        ],
    )


def test_empty_manifest_joins_only_exact_capture_file_and_reports_acquire_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = tmp_path / "capture-empty.json.gz"
    write_frozen_manifest(capture, ())
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    rows = [
        {"capture_file": None, "wall_s": 999},
        {"wall_s": 888},
        {"capture_file": "other.json.gz", "wall_s": 777},
        {
            "capture_file": capture.name,
            "complete": False,
            "wall_s": 20,
            "compile_wall_s": 2,
            "evolve_wall_s": 3,
            "readout_wall_s": 5,
        },
    ]
    (telemetry / "routing-test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert inspector.main([str(capture), "--telemetry-directory", str(telemetry)]) == 0
    report = json.loads(capsys.readouterr().out)["captures"][0]
    assert report["file"] == capture.name
    assert report["batches"] == []
    assert report["status"]
    assert report["timing"] == {
        "complete": False,
        "wall_s": 20,
        "compile_wall_s": 2,
        "evolve_wall_s": 3,
        "readout_wall_s": 5,
        "remainder_s": 10,
    }


@pytest.mark.parametrize("matches", [0, 2])
def test_missing_or_ambiguous_telemetry_never_supplies_acquire_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], matches: int
) -> None:
    capture = tmp_path / "capture-empty.json.gz"
    write_frozen_manifest(capture, ())
    row = {
        "capture_file": capture.name,
        "complete": True,
        "wall_s": 20,
        "compile_wall_s": 2,
        "evolve_wall_s": 3,
        "readout_wall_s": 5,
    }
    for index in range(matches):
        (tmp_path / f"routing-{index}.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
    argv = [str(capture), "--telemetry-directory", str(tmp_path)]
    if matches:
        with pytest.raises(ValueError, match=capture.name):
            inspector.main(argv)
    else:
        assert inspector.main(argv) == 0
        report = json.loads(capsys.readouterr().out)["captures"][0]
        assert "timing" not in report
        assert report["telemetry_status"]


def test_report_preserves_rounds_candidates_and_only_marks_unresolved_rows_pending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], metadata: CaptureMetadata
) -> None:
    capture = tmp_path / "capture-routes.json.gz"
    write_capture(capture, metadata)
    before = capture.read_bytes()

    assert inspector.main([str(capture), "--telemetry-directory", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)["captures"][0]
    batch = report["batches"][0]
    assert (batch["capture_id"], batch["batch_index"], batch["complete"]) == (
        "evolve-a",
        0,
        False,
    )
    assert batch["routes"] == metadata["routes"]
    assert batch["pending"] == [{"point": 8, "segment": 3, "prefix_run": None}]
    assert batch["pending_status"]
    assert batch["reason"] and batch["error"]
    assert "timing" not in report
    assert capture.read_bytes() == before


def test_all_rejected_caller_is_not_pending_and_keeps_recorded_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], metadata: CaptureMetadata
) -> None:
    metadata["families"] = ["multitone_rwa"]
    metadata["routes"] = [
        {
            **metadata["routes"][0],
            "candidates": [{"family": "multitone_rwa", "guard": "rejected"}],
        }
    ]
    capture = tmp_path / "capture-rejected.json.gz"
    write_capture(capture, metadata)

    assert inspector.main([str(capture), "--telemetry-directory", str(tmp_path)]) == 0
    batch = json.loads(capsys.readouterr().out)["captures"][0]["batches"][0]
    assert batch["pending"] == []
    assert batch["pending_status"] is None
    assert batch["complete"] is False
    assert batch["routes"] == metadata["routes"]
    assert batch["reason"] == "未知"
    assert batch["error"] == "未提供"


def test_prefix_adopters_reuse_representative_across_batches_but_not_evolves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], metadata: CaptureMetadata
) -> None:
    metadata["batch_count"] = 2
    metadata["prefix_runs"] = [
        {
            "representative": {"point": 4, "segment": 2},
            "adopters": [
                {"batch": 0, "point": 4, "segment": 2},
                {"batch": 1, "point": 9, "segment": 3},
            ],
        }
    ]
    metadata["routes"] = [
        {
            **metadata["routes"][1],
            "round": "caller",
            "prefix_run": 0,
            "segment": 0,
        }
    ]
    adopter = deepcopy(metadata)
    adopter.update(batch_index=1, routes=[], prefix_runs=[])
    separate = deepcopy(adopter)
    separate["capture_id"] = "evolve-b"
    capture = tmp_path / "capture-prefix.json.gz"
    write_capture(capture, metadata, adopter, separate)

    assert inspector.main([str(capture), "--telemetry-directory", str(tmp_path)]) == 0
    batches = json.loads(capsys.readouterr().out)["captures"][0]["batches"]
    assert batches[0]["routes"] == metadata["routes"]
    for index, point, segment in [(0, 4, 2), (1, 9, 3)]:
        adoption = batches[index]["prefix_adoptions"][0]
        assert (adoption["point"], adoption["segment"]) == (point, segment)
        assert (adoption["representative_batch"], adoption["prefix_run"]) == (0, 1)
        assert adoption["status"]
    assert batches[1]["routes"] == []
    assert batches[2]["prefix_adoptions"] == []


@pytest.mark.parametrize(
    ("options", "differences"),
    [
        ([], []),
        (["--budget", "0.000001", "--families", "null"], []),
        (["--budget", "0.000002"], ["budget"]),
        (["--families", '["direct"]'], ["families"]),
        (["--families", "[]"], ["families"]),
        (["--source-identity", "another-source"], ["source_identity"]),
        (["--segment-numeric-version", "new-version"], ["segment_numeric_version"]),
        (["--frozen-physical-version", "2"], ["frozen_physical_version"]),
    ],
)
def test_condition_comparison_labels_differences_without_replacing_recorded_routes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    metadata: CaptureMetadata,
    options: list[str],
    differences: list[str],
) -> None:
    capture = tmp_path / "capture-conditions.json.gz"
    write_capture(capture, metadata)
    before = capture.read_bytes()
    assert (
        inspector.main(
            [
                str(capture),
                "--telemetry-directory",
                str(tmp_path),
                *options,
            ]
        )
        == 0
    )
    batch = json.loads(capsys.readouterr().out)["captures"][0]["batches"][0]
    assert batch["condition_differences"] == differences
    assert batch["evaluation_status"]
    assert batch["routes_status"]
    assert batch["budget"] == 1e-6
    assert batch["families"] is None
    assert batch["routes"] == metadata["routes"]
    assert capture.read_bytes() == before
