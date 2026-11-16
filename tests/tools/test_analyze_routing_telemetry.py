"""Routing analyzer's public JSONL-to-aggregate main(argv) contract."""

import json
from pathlib import Path

import pytest

from tools.analyze_routing_telemetry import main


@pytest.fixture
def family() -> dict[str, int | float | None]:
    return {
        "rows": 2,
        "prepare_cpu_s": 1.0,
        "apply_cpu_s": 3.0,
        "cache_hits": 1,
        "rejected_rows": 1,
        "estimated_cost": 2.0,
        "estimated_cpu_s": 2.0,
    }


def test_wall_groups_include_only_measured_rows_without_losing_family_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], family
) -> None:
    rows = [
        dict(
            complete=True,
            wall_s=10,
            compile_wall_s=1,
            evolve_wall_s=2,
            readout_wall_s=3,
        ),
        dict(
            complete=True,
            wall_s=20,
            compile_wall_s=2,
            evolve_wall_s=4,
            readout_wall_s=6,
        ),
        dict(complete=True, wall_s=100, compile_wall_s=1, readout_wall_s=1),
        dict(
            complete=False,
            wall_s=8,
            compile_wall_s=1,
            evolve_wall_s=2,
            readout_wall_s=1,
        ),
        dict(
            complete=False,
            wall_s=99,
            compile_wall_s=1,
            evolve_wall_s=1,
            readout_wall_s=None,
        ),
        dict(wall_s=6, compile_wall_s=1, evolve_wall_s=1, readout_wall_s=1),
        dict(
            complete=None, wall_s=7, compile_wall_s=1, evolve_wall_s=1, readout_wall_s=2
        ),
        {},
        dict(
            complete=None,
            wall_s=None,
            compile_wall_s=None,
            evolve_wall_s=None,
            readout_wall_s=None,
        ),
    ]
    path = tmp_path / "routing-mixed.jsonl"
    path.write_text(
        "".join(json.dumps({**row, "families": {"t4": family}}) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["records"] == 9
    assert report["skipped"] == 0
    assert report["wall_by_complete"] == {
        "normal_return": dict(
            included=2,
            excluded=1,
            wall_s=30,
            compile_wall_s=3,
            evolve_wall_s=6,
            readout_wall_s=9,
            remainder_s=12,
        ),
        "raised": dict(
            included=1,
            excluded=1,
            wall_s=8,
            compile_wall_s=1,
            evolve_wall_s=2,
            readout_wall_s=1,
            remainder_s=4,
        ),
        "unknown": dict(
            included=2,
            excluded=2,
            wall_s=13,
            compile_wall_s=2,
            evolve_wall_s=2,
            readout_wall_s=3,
            remainder_s=6,
        ),
    }
    assert (
        sum(
            group["included"] + group["excluded"]
            for group in report["wall_by_complete"].values()
        )
        == report["records"]
    )
    assert report["families"]["t4"]["rows"] == 18
    assert report["families"]["t4"]["prepare_cpu_s"] == 9
    assert report["families"]["t4"]["apply_cpu_s"] == 27


@pytest.mark.parametrize(
    ("field", "value"),
    [("complete", value) for value in ("true", "false", 0, 1, 2, 1.5, [], {})]
    + [
        (field, value)
        for field in ("wall_s", "compile_wall_s", "evolve_wall_s", "readout_wall_s")
        for value in ("1", True, False, [], {})
    ],
)
def test_wrong_json_types_skip_entire_record_with_location(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], family, field, value
) -> None:
    valid = dict(
        complete=True,
        wall_s=10,
        compile_wall_s=1.5,
        evolve_wall_s=2,
        readout_wall_s=3,
        families={"t4": family},
    )
    path = tmp_path / "routing-invalid.jsonl"
    path.write_text(
        json.dumps(valid) + "\n" + json.dumps({**valid, field: value}) + "\n",
        encoding="utf-8",
    )

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert f"{path}:2: skipping invalid record" in captured.err
    assert report["records"] == 1
    assert report["skipped"] == 1
    assert report["families"]["t4"]["rows"] == 2
    assert report["wall_by_complete"] == {
        "normal_return": dict(
            included=1,
            excluded=0,
            wall_s=10,
            compile_wall_s=1.5,
            evolve_wall_s=2,
            readout_wall_s=3,
            remainder_s=3.5,
        ),
    }


@pytest.mark.parametrize(
    "field", ("wall_s", "compile_wall_s", "evolve_wall_s", "readout_wall_s")
)
@pytest.mark.parametrize("missing", (True, False), ids=("missing", "null"))
def test_one_unmeasured_wall_field_excludes_times_but_keeps_family(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], family, field, missing
) -> None:
    row: dict[str, float | None] = dict(
        wall_s=10, compile_wall_s=1, evolve_wall_s=2, readout_wall_s=3
    )
    if missing:
        del row[field]
    else:
        row[field] = None
    path = tmp_path / "routing-unmeasured.jsonl"
    path.write_text(
        json.dumps({**row, "complete": False, "families": {"t4": family}}) + "\n",
        encoding="utf-8",
    )

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.err == ""
    assert report["records"] == 1
    assert report["skipped"] == 0
    assert report["families"]["t4"]["rows"] == 2
    assert report["wall_by_complete"] == {
        "raised": dict(
            included=0,
            excluded=1,
            wall_s=0,
            compile_wall_s=0,
            evolve_wall_s=0,
            readout_wall_s=0,
            remainder_s=0,
        ),
    }


def test_multiple_inputs_merge_family_calibration_and_skip_bad_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], family
) -> None:
    first = tmp_path / "single"
    second = tmp_path / "six"
    first.mkdir()
    second.mkdir()
    path = first / "routing-first.jsonl"
    path.write_text(
        json.dumps({"families": {"t4": family}}) + "\n{broken\n", encoding="utf-8"
    )
    (second / "routing-second.jsonl").write_text(
        json.dumps(
            {
                "families": {
                    "t4": {
                        **family,
                        "prepare_cpu_s": 10,
                        "apply_cpu_s": 2,
                        "estimated_cost": 4,
                        "estimated_cpu_s": None,
                    },
                    "dop853": {
                        **family,
                        "rows": 6,
                        "prepare_cpu_s": 2,
                        "apply_cpu_s": 4,
                        "estimated_cost": 0,
                        "estimated_cpu_s": None,
                    },
                }
            }
        )
        + '\n{"families": {"t4": {}}}\n',
        encoding="utf-8",
    )
    (second / "unrelated.jsonl").write_text("ignored", encoding="utf-8")

    assert main([str(first), str(second), str(path)]) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert f"{path}:2: skipping invalid record" in captured.err
    assert "routing-second.jsonl:2: skipping invalid record" in captured.err
    assert report["records"] == 2
    assert report["skipped"] == 2
    assert report["estimated_cost_unit"] == "relative scheduling units"
    assert report["wall_by_complete"] == {
        "unknown": dict(
            included=0,
            excluded=2,
            wall_s=0,
            compile_wall_s=0,
            evolve_wall_s=0,
            readout_wall_s=0,
            remainder_s=0,
        ),
    }
    assert report["families"]["t4"] == dict(
        rows=4,
        cache_hits=2,
        rejected_rows=2,
        prepare_cpu_s=11,
        apply_cpu_s=5,
        estimated_cost=6,
        estimated_cpu_s=2,
        covered_actual_cpu_s=4,
        estimated_records=1,
        usage_fraction=0.4,
        cpu_s_per_cost_unit=pytest.approx(8 / 3),
        cpu_relative_error=1,
    )
    assert report["families"]["dop853"]["usage_fraction"] == 0.6
    assert report["families"]["dop853"]["cpu_s_per_cost_unit"] is None
    assert report["families"]["dop853"]["cpu_relative_error"] is None
