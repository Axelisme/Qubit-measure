"""Public acquire telemetry JSONL contract shared by physical environments."""

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path


def test_writer_serializes_common_spans_completion_and_namespaced_extras(
    tmp_path: Path,
):
    try:
        from zcu_tools.simulators.telemetry import AcquireTelemetry, RoutingTelemetry
    except ImportError as error:
        raise AssertionError("Writer must accept a complete acquire summary") from error

    class CallerSummary(AcquireTelemetry):
        caller_context: str

    summary: CallerSummary = {
        "points": 2,
        "rounds": 3,
        "wall_s": 1.0,
        "compile_wall_s": 0.1,
        "evolve_wall_s": 0.5,
        "readout_wall_s": 0.2,
        "complete": False,
        "extras": {
            "ftf.projection_outcomes": [4, 2],
            "ftf.projection_pruned_weight": [0.0, 0.01],
        },
        "caller_context": "not part of the wire contract",
    }
    before = json.dumps(summary, sort_keys=True)
    writer = RoutingTelemetry(tmp_path)
    writer.write(summary, paths=())
    assert json.dumps(summary, sort_keys=True) == before
    records = [
        json.loads(line)
        for path in tmp_path.glob("routing-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert set(record) == {
        "points",
        "rounds",
        "wall_s",
        "compile_wall_s",
        "evolve_wall_s",
        "readout_wall_s",
        "complete",
        "extras",
        "timestamp",
        "frontend_cpu_s",
        "estimated_cost_unit",
        "families",
    }
    assert {
        key: record[key]
        for key in (
            "points",
            "rounds",
            "wall_s",
            "compile_wall_s",
            "evolve_wall_s",
            "readout_wall_s",
            "complete",
            "families",
            "extras",
        )
    } == {
        "points": 2,
        "rounds": 3,
        "wall_s": 1.0,
        "compile_wall_s": 0.1,
        "evolve_wall_s": 0.5,
        "readout_wall_s": 0.2,
        "complete": False,
        "families": {},
        "extras": {
            "ftf.projection_outcomes": [4, 2],
            "ftf.projection_pruned_weight": [0.0, 0.01],
        },
    }


def test_writer_removes_oldest_routing_files_and_warns_only_once(tmp_path: Path):
    try:
        from zcu_tools.simulators.telemetry import RoutingTelemetry
    except ImportError as error:
        raise AssertionError(
            "Simulators must provide a bounded best-effort routing writer"
        ) from error

    directory = tmp_path / "bounded"
    directory.mkdir()
    oldest = directory / "routing-20000101.jsonl"
    newer = directory / "routing-20000102.jsonl"
    oldest.write_text("x" * 400)
    newer.write_text("y" * 400)
    os.utime(oldest, (1, 1))
    os.utime(newer, (2, 2))
    unrelated = directory / "keep.txt"
    unrelated.write_text("owned by the caller")
    writer = RoutingTelemetry(directory, max_bytes=1000)
    writer.write(
        {
            "points": 2,
            "rounds": 1,
            "wall_s": 0.1,
            "compile_wall_s": 0.0,
            "evolve_wall_s": 0.0,
            "readout_wall_s": 0.0,
            "complete": True,
            "extras": {},
        },
        paths=(),
    )
    assert not oldest.exists()
    assert newer.exists()
    assert unrelated.read_text() == "owned by the caller"
    assert sum(p.stat().st_size for p in directory.glob("routing-*.jsonl")) <= 1000
    current = directory / f"routing-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    assert json.loads(current.read_text())["points"] == 2

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    failing = RoutingTelemetry(blocked)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            failing.write(
                {
                    "points": 1,
                    "rounds": 1,
                    "wall_s": 0.0,
                    "compile_wall_s": 0.0,
                    "evolve_wall_s": 0.0,
                    "readout_wall_s": 0.0,
                    "complete": False,
                    "extras": {},
                },
                paths=(),
            )
    assert len(caught) == 1
    assert "routing telemetry" in str(caught[0].message)
