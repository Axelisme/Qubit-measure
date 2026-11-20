"""The wall-time CLI reports Run facts without executing a real experiment."""

import json
from pathlib import Path

import pytest

from tools.experiment_wall_time import RunTiming, TracerCase, main


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Supply minimal hashed inputs for tests which replace the Run edge."""
    data_dir = tmp_path / "inputs"
    fixture = data_dir / "fixture" / "result" / "demo" / "points" / "initial"
    fixture.mkdir(parents=True)
    (fixture / "parameters.yaml").write_bytes(b"abc")
    (data_dir / "ftf-settings.yaml").write_bytes(b"settings")
    (data_dir / "board-profile.yaml").write_bytes(b"profile")
    (data_dir / "tracer.json").write_text(
        json.dumps(
            {
                "adapter": "fake/frequency",
                "guide": "lab/frequency/guide.md",
                "entry": "demo",
                "point": "initial",
            }
        ),
        encoding="utf-8",
    )
    return data_dir


def test_main_records_run_and_sums_acquires_from_its_telemetry(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    def measure(case: TracerCase, _data_dir: Path, telemetry: Path) -> RunTiming:
        assert case.adapter == "fake/frequency"
        assert telemetry == output / "telemetry"
        records = [
            {
                "families": {},
                "complete": True,
                "wall_s": 4.5,
                "compile_wall_s": 0.5,
                "evolve_wall_s": 2.0,
                "readout_wall_s": 1.0,
            },
            {
                "families": {},
                "complete": True,
                "wall_s": 2.0,
                "compile_wall_s": 0.5,
                "evolve_wall_s": 0.75,
                "readout_wall_s": 0.5,
            },
        ]
        for index, record in enumerate(records):
            (telemetry / f"routing-{index}.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
        return RunTiming(
            "2026-11-21T00:00:00+00:00",
            "2026-11-21T00:00:08+00:00",
            8.0,
        )

    commit = "0123456789abcdef0123456789abcdef01234567"
    assert (
        main(
            ["--output", str(output)],
            data_dir=data_dir,
            measure=measure,
            read_commit=lambda: commit,
        )
        == 0
    )

    report_path = output / "report.json"
    assert report_path.exists(), "successful Run must leave a JSON report"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["commit"] == commit
    input_hashes = report["inputs"]
    assert set(input_hashes) == {
        "ftf-settings.yaml",
        "board-profile.yaml",
        "tracer.json",
        "fixture/result/demo/points/initial/parameters.yaml",
    }
    assert input_hashes["fixture/result/demo/points/initial/parameters.yaml"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    [row] = report["rows"]
    assert row == {
        "adapter": "fake/frequency",
        "guide": "lab/frequency/guide.md",
        "started_at": "2026-11-21T00:00:00+00:00",
        "finished_at": "2026-11-21T00:00:08+00:00",
        "wall_s": 8.0,
        "acquires": 2,
        "compile_s": 1.0,
        "evolve_s": 2.75,
        "readout_s": 1.5,
        "remainder_s": 1.25,
        "status": "ok",
    }
    table = (output / "summary.md").read_text(encoding="utf-8")
    assert "| fake/frequency | 8.000 | 1.000 | 2.750 | 1.500 | 1.250 | 2 |" in table
    assert "`lab/frequency/guide.md`" in table


def test_main_keeps_run_wall_when_telemetry_is_missing(
    tmp_path: Path, data_dir: Path
) -> None:
    output = tmp_path / "output"

    def measure(_case: TracerCase, _data_dir: Path, _telemetry: Path) -> RunTiming:
        return RunTiming("2026-11-21T00:00:00+00:00", "2026-11-21T00:00:03+00:00", 3.0)

    assert main(["--output", str(output)], data_dir=data_dir, measure=measure) == 0

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    [row] = report["rows"]
    assert row["status"] == "missing_breakdown"
    assert row["wall_s"] == 3.0
    assert row["acquires"] == 0
    assert all(
        row[key] is None
        for key in ("compile_s", "evolve_s", "readout_s", "remainder_s")
    )
    assert "| fake/frequency | 3.000 | — | — | — | — | 0 | 分項缺 |" in (
        output / "summary.md"
    ).read_text(encoding="utf-8")
