"""Real Engine evidence with distinct calibration/final device snapshots."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from pydantic import BaseModel, JsonValue
from zcu_tools.datafile import (
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    save_run_data,
)
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.workflows import (
    Completed,
    Done,
    Engine,
    EnginePorts,
    InitEnv,
    Next,
    Run,
    RunIdentity,
    RunPaths,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests._native_support import native_metadata
from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    RecordingBar,
    RecordingDevices,
)
from zcu_lab.workflows.calibration import Calibrated


@dataclass(frozen=True)
class Capture:
    """Native callback result retains only the actual Run snapshot."""

    snapshot: dict[str, dict[str, JsonValue]]


@dataclass(frozen=True)
class CaptureCfg:
    index: int


def acquire(run: Run[CaptureCfg]) -> Capture:
    return Capture(dict(run.device_snapshot))


def save_capture(source: Completed[CaptureCfg, Capture], path: Path) -> None:
    save_run_data(
        path,
        ExperimentPayload(
            {
                DataVariable("data"): LabberPayload(
                    ("Signal", "a.u.", np.array([1 + 0j])),
                    [("Time", "s", np.array([0.0]))],
                )
            },
            LabberMetadata(),
            "single",
        ),
        replace(native_metadata(), device_snapshot=source.result.snapshot),
        cfg=CfgSnapshot(values={}, cfg_type="Capture", schema_version="1.0"),
    )


def sample_engine[P: BaseModel, R](
    plan: P, root: Path, record: Callable[[int], R], *, seed: Calibrated
) -> tuple[Engine[Calibrated], RunPaths, RecordingDevices]:
    """Commit two successful points and one rejected point with two env.run each."""
    devices = RecordingDevices()
    devices.drivers["flux"] = FakeDevice(fast_mode=True)
    devices.drivers["other"] = FakeDevice(fast_mode=True)
    devices.drivers["other"].setup(
        FakeDeviceInfo(address="none", value=999), progress=False
    )
    paths = RunPaths(root / "metadata", root / "data")
    engine = Engine(
        EnginePorts(
            Plots(NonPresentingHost()),
            RecordingBar,
            ManualClock(),
            context=seed,
            devices=devices,
        )
    )

    def init(_env: InitEnv[Calibrated], _plan: P) -> int:
        return 0

    @workflow("samples", plan=type(plan), tunables=BaseModel, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: P, tun: BaseModel, state: int
    ) -> Step[int, R]:
        del plan, tun
        if state == 3:
            return Done()
        devices.drivers["flux"].setup(
            FakeDeviceInfo(address="none", value=10 + state), progress=False
        )
        yield from env.run(acquire, CaptureCfg(state), save=save_capture)
        devices.drivers["flux"].setup(
            FakeDeviceInfo(address="none", value=40 + state), progress=False
        )
        yield from env.run(acquire, CaptureCfg(state), save=save_capture)
        return Next(record(state), state + 1)

    engine.start(
        step,
        plan=plan,
        tunables=BaseModel(),
        paths=paths,
        identity=RunIdentity("sample-run", native_metadata().provenance),
    )
    engine.execute("sample-run")
    devices.drivers["flux"].set_value(777)
    return engine, paths, devices
