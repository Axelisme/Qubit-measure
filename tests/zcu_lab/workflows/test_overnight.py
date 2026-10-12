"""Native overnight pairs through Engine, FakeDevice and simulator persistence."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.datafile import JsonObject, SoftwareProvenance
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
from zcu_tools.program.v2 import SweepCfg

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
    # Echo's low contrast needs shot-noise averaging, as in native recovery tests.
    tun = workflows.OvernightTunables(reps=2000, rounds=2)
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
    assert t1.cfg is not None and t2.cfg is not None
    assert isinstance(t1.cfg.sweep.length, SweepCfg)
    assert t1.cfg.reps == t2.cfg.reps == tun.reps
    assert t1.cfg.rounds == t2.cfg.rounds == tun.rounds
    assert t1.cfg.sweep.length.stop == 80.0
    assert t2.cfg.sweep.length.stop == 40.0


def start_overnight(
    inputs: Inputs,
    root: Path,
    devices: RecordingDevices,
    *,
    interval: float = 0.25,
) -> Engine[None]:
    """Start three real native pairs with fixed cfg and injected device timing."""
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
        plan=workflows.OvernightPlan(
            flux_dev="flux",
            flux_unit="native",
            flux=inputs.flux_value,
            outer_rounds=3,
            start_interval_s=interval,
            points=21,
            qub_pulse=inputs.t1.modules.pi_pulse,
            readout=inputs.readout.modules.readout,
            snapshot=make_snapshot(),
        ),
        tunables=workflows.OvernightTunables(reps=2000, rounds=2),
        paths=RunPaths(root / "metadata", root / "data"),
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
    return engine


def pair_time(record: CommittedRecord, field: str) -> datetime:
    """Read one declared aware UTC pair timestamp through the committed record."""
    value = record.encoded_record.value
    assert isinstance(value, dict)
    encoded = value[field]
    assert isinstance(encoded, str)
    return datetime.fromisoformat(encoded)


@pytest.mark.uses_wall_clock
def test_overrun_restarts_interval_from_actual_start_without_catchup(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    from zcu_tools.experiment.stop_signal import StopSignal

    devices = RecordingDevices()
    engine = start_overnight(inputs, tmp_path, devices, interval=1.0)

    def overrun_first_pair(signal: StopSignal) -> None:
        if len(devices.values) == 1:
            signal.event.wait(2.1)

    devices.on_set = overrun_first_pair
    status = engine.execute("overnight")
    assert status.lifecycle == "done", status.reason
    first, second, third = engine.records("overnight")
    assert (
        pair_time(second, "actual_start") - pair_time(first, "actual_start")
    ).total_seconds() >= 2.1
    # A fixed original-start grid would immediately run the third pair to catch up.
    assert (
        pair_time(third, "actual_start") - pair_time(second, "actual_start")
    ).total_seconds() >= 1.0
    assert (
        pair_time(third, "scheduled_start") - pair_time(second, "actual_start")
    ).total_seconds() >= 1.0


class ControlledDevices(RecordingDevices):
    """Allow public device captures to request run control between native members."""

    on_capture: Callable[[], None] | None = None

    def capture_snapshot(self) -> dict[str, JsonObject]:
        snapshot = super().capture_snapshot()
        if self.on_capture is not None:
            self.on_capture()
        return snapshot


@pytest.mark.uses_wall_clock
def test_pause_discards_incomplete_pair_and_resume_redoes_then_stop_keeps_commits(
    inputs: Inputs,
    tmp_path: Path,
) -> None:
    devices = ControlledDevices()
    engine = start_overnight(inputs, tmp_path, devices)

    def control_between_members() -> None:
        if devices.captures == 4:
            engine.pause("overnight")
        elif devices.captures == 7:
            engine.stop("overnight")

    devices.on_capture = control_between_members
    paused = engine.execute("overnight")
    assert paused.lifecycle == "paused"
    prior = engine.records("overnight")
    assert len(prior) == 1
    assert paused.committed_seq == 1
    assert any(
        p.name == "overnight rounds" and p.completed == 1 for p in paused.progress
    )
    # T1 of the cancelled second pair remains native evidence but is not a commit.
    discarded = tmp_path / "data" / "iter" / "000002"
    assert len(tuple(discarded.glob("runs/*run_t1.h5"))) == 1
    Event().wait(0.3)
    resumed_at = datetime.now(UTC)
    engine.resume("overnight")
    stopped = engine.execute("overnight")
    assert stopped.lifecycle == "stopped"
    assert stopped.committed_seq == 2
    records = engine.records("overnight")
    assert records[0] == prior[0]
    assert len(records) == 2
    assert records[1].call_seq == 3
    assert len(records[1].run_files) == 2
    assert pair_time(records[1], "scheduled_start") >= resumed_at
    assert pair_time(records[1], "actual_start") >= resumed_at
    assert any(
        p.name == "overnight rounds" and p.completed == 2 for p in stopped.progress
    )


@pytest.mark.uses_wall_clock
def test_point_names_each_native_pair_member(inputs: Inputs, tmp_path: Path) -> None:
    engine = start_overnight(inputs, tmp_path, RecordingDevices(), interval=0.001)
    status = engine.execute("overnight")
    assert status.lifecycle == "done", status.reason
    record = engine.records("overnight")[0]
    point = record.encoded_record.value
    assert isinstance(point, dict)
    assert (point["t1_file"], point["t2_file"]) == ("t1.h5", "t2echo.h5")
    for filename, experiment in (("t1.h5", T1Exp()), ("t2echo.h5", T2EchoExp())):
        native, metadata = load_run_with_metadata(tmp_path / "data" / record.iteration_dir / filename, spec=experiment.persistence_spec())
        assert native.cfg is not None
        assert metadata.snapshot == make_snapshot()
        assert metadata.device_snapshot is not None
