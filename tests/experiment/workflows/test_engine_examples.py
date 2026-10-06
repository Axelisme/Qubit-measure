"""Offline v0 workflow functions exercise the real Engine/Run/IO seam."""

from pathlib import Path

import pytest
from pydantic import BaseModel
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    DeviceSnapshot,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
    WorkflowStep,
)

from zcu_lab.workflows.v0 import (
    CalTunables,
    DemoContext,
    FluxPlan,
    OvernightPlan,
    OvernightTunables,
    T1Tunables,
    T2EchoTunables,
    t1_fluxdep,
    t1_overnight,
    t2echo_fluxdep,
)

from ._engine_fakes import (
    START,
    ManualClock,
    RecordingBar,
    RecordingDevices,
    RecordingPlots,
)


def execute_example[P: BaseModel, T: BaseModel, S, R](
    root: Path, step: WorkflowStep[P, T, S, R, DemoContext], plan: P, tunables: T
) -> tuple[tuple[CommittedRecord, ...], ManualClock, RecordingDevices]:
    clock, devices, plots = ManualClock(), RecordingDevices(), RecordingPlots()
    engine = Engine(
        EnginePorts(
            plots,
            RecordingBar,
            clock,
            devices=devices,
            context=DemoContext(5000.0, 0.05, 7000.0),
        )
    )
    engine.start(
        step,
        plan=plan,
        tunables=tunables,
        paths=RunPaths(root / "metadata", root / "data"),
        identity=RunIdentity("example", DeviceSnapshot(()), "offline"),
    )
    status = engine.execute("example")
    assert status.lifecycle == "done", status.reason
    records = engine.records("example")
    assert status.call_seq == len(records) + 1
    assert status.committed_seq == len(records)
    for record in records:
        for run_file in record.run_files:
            assert (root / "data" / record.iteration_dir / run_file).is_file()
    assert len(tuple((root / "data/iter").iterdir())) == len(records) + 1
    assert not tuple((root / "data").glob("iter/*/runs/.*.tmp.h5"))
    return records, clock, devices


def test_t1_fluxdep_commits_offline_fits_and_preserves_per_effect_files(
    tmp_path: Path,
) -> None:
    records, _, devices = execute_example(
        tmp_path,
        t1_fluxdep,
        FluxPlan(flux_dev="flux", fluxes=(0.1, 0.2), points=7),
        T1Tunables(),
    )
    assert [len(record.run_files) for record in records] == [4, 3]
    assert devices.values == [("flux", 0.1), ("flux", 0.2)]
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["reason"] is None
        value = point["t1"]
        assert isinstance(value, dict)
        assert value["value_us"] == pytest.approx(20.0)
    assert (tmp_path / "data/iter/000003/files/summary.png").is_file()


def test_t2echo_fluxdep_uses_the_same_core_with_its_own_typed_record(
    tmp_path: Path,
) -> None:
    records, _, _ = execute_example(
        tmp_path,
        t2echo_fluxdep,
        FluxPlan(flux_dev="flux", fluxes=(0.1, 0.2), points=7),
        T2EchoTunables(),
    )
    assert len(records) == 2
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        value = point["t2echo"]
        assert isinstance(value, dict)
        assert value["value_us"] == pytest.approx(30.0)


def test_overnight_waits_absolute_round_deadlines_and_reuses_calibration(
    tmp_path: Path,
) -> None:
    records, clock, devices = execute_example(
        tmp_path,
        t1_overnight,
        OvernightPlan(flux_dev="flux", flux=0.2, period_s=10, rounds=3, points=7),
        OvernightTunables(),
    )
    assert [len(record.run_files) for record in records] == [4, 1, 1]
    assert [(target - START).total_seconds() for target in clock.targets] == [
        10.0,
        20.0,
    ]
    assert devices.values == [("flux", 0.2)]


def test_fit_gate_rejection_commits_missing_values_and_continues(
    tmp_path: Path,
) -> None:
    records, _, _ = execute_example(
        tmp_path,
        t1_fluxdep,
        FluxPlan(flux_dev="flux", fluxes=(0.1, 0.2), points=7),
        T1Tunables(max_rel_err=0.01),
    )
    assert len(records) == 2
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["t1"] is None
        assert point["reason"] == "fit rejected"
        assert record.run_files[-1].endswith("run_t1.h5")


def test_overnight_retries_unaccepted_calibration_on_next_round(tmp_path: Path) -> None:
    records, _, _ = execute_example(
        tmp_path,
        t1_overnight,
        OvernightPlan(flux_dev="flux", flux=0.2, period_s=10, rounds=2, points=7),
        OvernightTunables(cal=CalTunables(min_snr=50)),
    )
    assert [len(record.run_files) for record in records] == [3, 3]
    for record in records:
        point = record.encoded_record.value
        assert isinstance(point, dict)
        assert point["t1"] is None
        assert point["reason"] == "no accepted frequency fit"
