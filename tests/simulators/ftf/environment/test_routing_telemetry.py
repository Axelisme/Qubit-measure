"""One bounded, best-effort routing record per admitted FTF acquisition."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulators.ftf.environment import ftf

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.ftf.environment._templates import settings


@pytest.mark.sim
def test_acquire_writes_one_line_for_all_rounds_and_opt_out_is_bitwise(tmp_path: Path):
    directory = tmp_path / "enabled"
    try:
        observed = ftf(
            settings().with_telemetry(enabled=True, directory=directory),
            soccfg=profile_soccfg("ftf"),
            time_scale=0,
        )
    except AttributeError as error:
        raise AssertionError(
            "FTF must provide configurable acquire telemetry"
        ) from error
    plain_directory = tmp_path / "disabled"
    plain = ftf(
        settings().with_telemetry(enabled=False, directory=plain_directory),
        soccfg=profile_soccfg("ftf"),
        time_scale=0,
    )
    try:
        program = compile_modules(
            observed.soccfg,
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.1,
                    pre_delay=0.01,
                    trig_offset=0.02,
                    waveform=ConstWaveformCfg(length=0.04),
                    ro_length=0.04,
                ).build("ro")
            ],
            reps=2,
        )
        expected = program.acquire(
            observed.soc,
            progress=False,
            rounds=3,
        )
        actual = program.acquire(
            plain.soc,
            progress=False,
            rounds=3,
        )
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
        program.acquire(
            observed.soc,
            progress=False,
            rounds=3,
        )
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(records) == 2
        assert sum(f["cache_hits"] for f in records[1]["families"].values()) == 3
        assert sum(f["actual_cpu_s"] for f in records[1]["families"].values()) == 0.0
    finally:
        observed.devices.close_all_devices()
        plain.devices.close_all_devices()

