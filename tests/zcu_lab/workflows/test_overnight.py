"""Native overnight pairs through Engine, FakeDevice and simulator persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.device import FakeDevice
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
)
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.experiment.workflows._engine_fakes import RecordingBar, RecordingDevices
from zcu_lab import workflows
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoExp

from ._sim_inputs import Inputs, make_snapshot


@pytest.mark.uses_wall_clock
def test_overnight_commits_fixed_native_pairs_and_round_progress(
    inputs: Inputs, tmp_path: Path
) -> None:
    plan = workflows.OvernightPlan(
        flux_dev="flux",
        flux_unit="native",
        flux=inputs.flux_value,
        outer_rounds=3,
        start_interval_s=0.01,
        points=21,
        qub_pulse=inputs.t1.modules.pi_pulse,
        readout=inputs.readout.modules.readout,
        snapshot=make_snapshot(),
    )
    tun = workflows.OvernightTunables(reps=100, rounds=2)
    devices = RecordingDevices()
    driver = FakeDevice(fast_mode=True)
    devices.drivers["flux"] = driver
    paths = RunPaths(tmp_path / "metadata", tmp_path / "data")
    engine = Engine[None](
        EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            UtcClock(),
            soc=inputs.soc,
            soccfg=inputs.soccfg,
            devices=devices,
        )
    )
    engine.start(
        workflows.overnight,
        plan=plan,
        tunables=tun,
        paths=paths,
        identity=RunIdentity(
            "overnight",
            SoftwareProvenance(
                software_versions={},
                git_commit=None,
                git_dirty=None,
                qick_version=None,
                soc_fingerprint=None,
                hostname="offline",
            ),
        ),
    )
    before = datetime.now(UTC)
    try:
        status = engine.execute("overnight")
        assert status.lifecycle == "done", status.reason
        assert status.committed_seq == 3
        records = engine.records("overnight")
        assert len(records) == 3
        assert devices.captures == 6
        assert devices.values == [("flux", inputs.flux_value)] * 3
        assert any(p.total == 3 and p.completed == 3 for p in status.progress)
        for index, record in enumerate(records):
            data = record.encoded_record.value
            assert isinstance(data, dict)
            assert data["round_index"] == index
            assert data["flux"] == inputs.flux_value
            actual = data["actual_start"]
            scheduled = data["scheduled_start"]
            assert isinstance(actual, str) and isinstance(scheduled, str)
            assert before <= datetime.fromisoformat(actual) <= datetime.now(UTC)
            assert datetime.fromisoformat(scheduled) <= datetime.fromisoformat(actual)
            assert data["t1_us"] == pytest.approx(20.0, abs=2.0)
            assert data["t2_us"] == pytest.approx(10.0, abs=2.0)
            assert data["t1_reason"] is None and data["t2_reason"] is None
            assert len(record.run_files) == 2
            assert_native_pair(paths, record, plan, tun)
        journal = [
            json.loads(line)
            for line in (paths.metadata_root / "journal.jsonl").read_text().splitlines()
        ]
        commits = [event for event in journal if event["kind"] == "step_committed"]
        assert len(commits) == 3
        assert [event["run_files"] for event in commits] == [
            list(record.run_files) for record in records
        ]
    finally:
        driver.close()


def assert_native_pair(
    paths: RunPaths,
    record: CommittedRecord,
    plan: workflows.OvernightPlan,
    tun: workflows.OvernightTunables,
) -> None:
    """Read both native members and verify fixed cfg and acquisition metadata."""
    root = paths.data_root / record.iteration_dir
    t1, t1_metadata = load_run_with_metadata(
        root / record.run_files[0], spec=T1Exp().persistence_spec()
    )
    t2, t2_metadata = load_run_with_metadata(
        root / record.run_files[1], spec=T2EchoExp().persistence_spec()
    )
    assert "run_t1" in record.run_files[0]
    assert "run_t2echo" in record.run_files[1]
    assert t1_metadata.device_snapshot is not None
    assert t2_metadata.device_snapshot == t1_metadata.device_snapshot
    assert t1_metadata.snapshot == plan.snapshot
    assert t2_metadata.snapshot == plan.snapshot
    assert t1.cfg.reps == t2.cfg.reps == tun.reps
    assert t1.cfg.rounds == t2.cfg.rounds == tun.rounds
    assert t1.cfg.sweep.length.stop == 80.0
    assert t2.cfg.sweep.length.stop == 40.0
