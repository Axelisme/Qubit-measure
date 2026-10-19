"""One bounded, best-effort routing record per admitted FTF acquisition."""

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.experiment._ftf_templates import settings


def test_acquire_writes_one_line_for_all_rounds_and_opt_out_is_bitwise(tmp_path: Path):
    directory = tmp_path / "enabled"
    try:
        observed = ftf(
            settings().with_telemetry(enabled=True, directory=directory),
            min_round_duration_s=0,
        )
    except AttributeError as error:
        raise AssertionError(
            "FTF must provide configurable acquire telemetry"
        ) from error
    plain_directory = tmp_path / "disabled"
    plain = ftf(
        settings().with_telemetry(enabled=False, directory=plain_directory),
        min_round_duration_s=0,
    )
    try:
        program = ModularProgramV2(
            observed.soccfg,
            ProgramV2Cfg(reps=2, rounds=3),
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.1,
                    pre_delay=0.01,
                    waveform=ConstWaveformCfg(length=0.04),
                    ro_length=0.04,
                ).build("ro")
            ],
        )
        expected = program.acquire(observed.soc, progress=False)
        actual = program.acquire(plain.soc, progress=False)
        assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes()
        assert not plain_directory.exists()
        files = list(directory.glob("routing-*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["points"] == 1
        assert record["rounds"] == 3
        assert type(record["wall_s"]) is float
        assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None
        assert (
            "experiment" not in record
        )  # Numeric templates carry no program identifier.
        assert sum(f["rows"] for f in record["families"].values()) == 3
        assert sum(f["cache_hits"] for f in record["families"].values()) == 2
        assert record["estimated_cost_unit"] == "relative scheduling units"
        program.acquire(observed.soc, progress=False)
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(records) == 2
        assert sum(f["cache_hits"] for f in records[1]["families"].values()) == 3
        assert sum(f["actual_cpu_s"] for f in records[1]["families"].values()) == 0.0
    finally:
        observed.devices.close_all_devices()
        plain.devices.close_all_devices()


def test_writer_removes_oldest_routing_files_and_warns_only_once(tmp_path: Path):
    try:
        from zcu_tools.experiment.ftf.telemetry import RoutingTelemetry
    except ImportError as error:
        raise AssertionError(
            "FTF must provide a bounded best-effort routing writer"
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
    writer = RoutingTelemetry(directory, max_bytes=700)
    writer.write(points=2, wall_s=0.1, rounds=1, paths=())
    assert not oldest.exists()
    assert newer.exists()
    assert unrelated.read_text() == "owned by the caller"
    assert sum(p.stat().st_size for p in directory.glob("routing-*.jsonl")) <= 700
    current = directory / f"routing-{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    assert json.loads(current.read_text())["points"] == 2

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    failing = RoutingTelemetry(blocked)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            failing.write(points=1, wall_s=0.0, rounds=1, paths=())
    assert len(caught) == 1
    assert "routing telemetry" in str(caught[0].message)
