"""Offline v0 workflow functions exercise the real Engine/Run/IO seam."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    CommittedRecord,
    DeviceSnapshot,
    Engine,
    EnginePorts,
    RunIdentity,
    RunPaths,
    WorkflowStep,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from zcu_lab.workflows.v0 import (
    CalTunables,
    DemoContext,
    FluxPlan,
    OvernightPlan,
    OvernightTunables,
    T2EchoTunables,
    t1_overnight,
    t2echo_fluxdep,
)

from ._engine_fakes import (
    START,
    ManualClock,
    RecordingBar,
    RecordingDevices,
)


def execute_example[P: BaseModel, T: BaseModel, S, R](
    root: Path, step: WorkflowStep[P, T, S, R, DemoContext], plan: P, tunables: T
) -> tuple[tuple[CommittedRecord, ...], ManualClock, RecordingDevices]:
    clock, devices, plots = (
        ManualClock(),
        RecordingDevices(),
        Plots(NonPresentingHost()),
    )
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
        identity=RunIdentity(
            "example",
            DeviceSnapshot({}),
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


@pytest.mark.parametrize("name", ["echo", "overnight"])
@pytest.mark.parametrize("unit", ["A", "V", "native", None, "mA"])
def test_v0_plan_requires_explicit_flux_unit_and_forwards_it(
    tmp_path: Path, name: str, unit: str | None
) -> None:
    data: dict[str, object] = {"flux_dev": "flux", "points": 7}
    if unit is not None:
        data["flux_unit"] = unit
    if name == "echo":
        data["fluxes"] = (0.1, 0.2)
        if unit not in ("A", "V", "native"):
            with pytest.raises(ValidationError, match="flux_unit"):
                FluxPlan.model_validate(data)
            return
        plan = FluxPlan.model_validate(data)
        _, _, devices = execute_example(tmp_path, t2echo_fluxdep, plan, T2EchoTunables())
        assert devices.values == [("flux", 0.1), ("flux", 0.2)]
        assert devices.units == [unit, unit]
    else:
        data.update(flux=0.2, rounds=1)
        if unit not in ("A", "V", "native"):
            with pytest.raises(ValidationError, match="flux_unit"):
                OvernightPlan.model_validate(data)
            return
        overnight = OvernightPlan.model_validate(data)
        _, _, devices = execute_example(
            tmp_path, t1_overnight, overnight, OvernightTunables()
        )
        assert devices.values == [("flux", 0.2)]
        assert devices.units == [unit]


@pytest.mark.parametrize("name", ["echo", "overnight"])
@pytest.mark.parametrize("port", ["devices", "context"])
def test_demo_workflows_check_ports_before_effects(
    tmp_path: Path, name: str, port: str
) -> None:
    clock, devices, plots = (
        ManualClock(),
        RecordingDevices(),
        Plots(NonPresentingHost()),
    )
    ports = EnginePorts(
        plots,
        RecordingBar,
        clock,
        devices=devices,
        context=DemoContext(5000.0, 0.05, 7000.0),
    )
    if port == "devices":
        ports = replace(ports, devices=None)
    else:
        ports = replace(ports, context=None)
    engine = Engine(ports)
    paths = RunPaths(tmp_path / "metadata", tmp_path / "data")
    identity = RunIdentity(
        "example",
        DeviceSnapshot({}),
        SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname="offline",
        ),
    )
    if name == "echo":
        engine.start(
            t2echo_fluxdep,
            plan=FluxPlan(flux_dev="flux", fluxes=(0.1,), points=7),
            tunables=T2EchoTunables(),
            paths=paths,
            identity=identity,
        )
    else:
        engine.start(
            t1_overnight,
            plan=OvernightPlan(
                flux_dev="flux", flux=0.2, period_s=10, rounds=1, points=7
            ),
            tunables=OvernightTunables(),
            paths=paths,
            identity=identity,
        )
    status = engine.execute("example")
    assert status.lifecycle == "failed"
    assert f"Missing workflow capability: {port}" in str(status.reason)
    assert devices.values == []
    assert clock.targets == []
    assert len(plots) == 0
    assert engine.records("example") == ()
    assert not tuple(paths.data_root.glob("iter/*/runs/*.h5"))


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
