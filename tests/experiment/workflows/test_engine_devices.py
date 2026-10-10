"""Engine Start/Resume retain detached native device provenance (D272)."""

from datetime import timedelta
from pathlib import Path

import pytest
from zcu_tools.datafile import JsonObject
from zcu_tools.device import DeviceInfo, FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.workflows import (
    DeviceSnapshot,
    Done,
    Engine,
    EnginePorts,
    Next,
    Run,
    Step,
    WorkflowEnv,
)

from ._engine_fakes import Cfg, Plan, Rig, State, Tunables, save_integer


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_native_start_and_resume_snapshots_are_detached_in_journal(rig: Rig) -> None:
    initial_info = FakeDeviceInfo(address="none", value=0.1)
    initial: dict[str, DeviceInfo] = {
        "z": initial_info,
        "a": FakeDeviceInfo(address="none"),
    }
    initial_json = {
        name: info.model_dump(mode="json") for name, info in initial.items()
    }

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        yield from env.wait_until(rig.clock.now() + timedelta(seconds=1))
        state.index = 1
        return Next(1, state)

    rig.start(step, devices=DeviceSnapshot(initial))
    initial_info.value = 0.9
    initial.clear()
    rig.clock.on_wait = lambda _signal: rig.engine.pause("run")
    assert rig.engine.execute("run").lifecycle == "paused"

    current_info = FakeDeviceInfo(address="none", value=0.2)
    current: dict[str, DeviceInfo] = {"flux": current_info}
    current_json = {
        name: info.model_dump(mode="json") for name, info in current.items()
    }
    rig.engine.resume("run", devices=DeviceSnapshot(current))
    current_info.value = 0.8
    current.clear()
    rig.clock.on_wait = None
    assert rig.engine.execute("run").lifecycle == "done"

    started = next(event for event in rig.events() if event["kind"] == "run_started")
    identity = started["identity"]
    assert isinstance(identity, dict)
    assert identity["devices"] == {"items": initial_json}
    resumed = next(event for event in rig.events() if event["kind"] == "resumed")
    assert resumed["devices"] == {"items": current_json}


class SnapshotDriver(FakeDevice):
    """Connected in-memory driver with observable reads and I/O failure."""

    info_model = FakeDeviceInfo

    def __init__(self) -> None:
        super().__init__(fast_mode=True)
        self.reads = 0
        self.error: OSError | None = None

    def get_info(self) -> FakeDeviceInfo:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return super().get_info()


def test_each_env_run_captures_all_drivers_before_callback_and_keeps_fixed_data(
    rig: Rig,
) -> None:
    selected, unused = SnapshotDriver(), SnapshotDriver()
    selected.set_value(0.1)
    unused.set_value(0.2)
    rig.devices.drivers.update(flux=selected, unused=unused)
    captured: list[dict[str, JsonObject]] = []

    def acquire(run: Run[Cfg]) -> int:
        assert (selected.reads, unused.reads) == (len(captured) + 1,) * 2
        before = dict(run.device_snapshot)
        selected.set_value(0.9)
        unused.set_value(0.8)
        assert run.device_snapshot == before
        captured.append(before)
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        for _ in range(2):
            yield from env.run(acquire, Cfg(dev={"flux": "flux"}), save=save_integer)
        return Done()

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"
    assert (selected.reads, unused.reads, rig.devices.captures) == (2, 2, 2)
    assert captured[0]["flux"]["value"] == 0.1
    assert captured[0]["unused"]["value"] == 0.2
    assert captured[1]["flux"]["value"] == 0.9
    assert captured[1]["unused"]["value"] == 0.8


def test_snapshot_read_failure_fails_run_before_callback_and_save(rig: Rig) -> None:
    broken = SnapshotDriver()
    broken.error = OSError("unused driver read failed")
    rig.devices.drivers["unused"] = broken
    callbacks: list[str] = []

    def acquire(_run: Run[Cfg]) -> int:
        callbacks.append("acquire")
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        yield from env.run(acquire, Cfg(), save=save_integer)
        return Done()

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert status.reason == "unused driver read failed"
    assert callbacks == []
    assert broken.reads == 1
    assert list(rig.paths.data_root.rglob("*.h5")) == []
    ended = next(event for event in rig.events() if event["kind"] == "run_ended")
    assert "OSError" in str(ended)


@pytest.mark.parametrize("missing", [True, False], ids=["missing", "explicit-empty"])
def test_env_run_distinguishes_missing_devices_from_explicit_empty_port(
    rig: Rig, missing: bool
) -> None:
    rig.engine = Engine(
        EnginePorts(
            rig.plots,
            rig.progress,
            rig.clock,
            context=7,
            devices=None if missing else rig.devices,
        )
    )
    callbacks: list[str] = []

    def acquire(run: Run[Cfg]) -> int:
        callbacks.append("acquire")
        assert run.device_snapshot == {}
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        yield from env.run(acquire, Cfg(), save=save_integer)
        return Done()

    rig.start(step)
    started = rig.engine.status()
    assert started is not None and started.lifecycle == "running"
    status = rig.engine.execute("run")
    if missing:
        assert status.lifecycle == "failed"
        assert status.reason is not None and "devices" in status.reason
        assert callbacks == []
    else:
        assert status.lifecycle == "done"
        assert callbacks == ["acquire"]
        assert rig.devices.captures == 1
