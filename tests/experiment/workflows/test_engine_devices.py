"""Engine device effects use native models, driver validation, and provenance."""

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.device import (
    DeviceInfo,
    DeviceManager,
    FakeDevice,
    FakeDeviceInfo,
    YOKOGS200Info,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import (
    Completed,
    DeviceSnapshot,
    Done,
    Next,
    Run,
    Step,
    WorkflowEnv,
)

from ._engine_fakes import Cfg, Plan, Rig, State, Tunables, save_integer


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


class ModelCfg(ExpCfgModel):
    value: int = 1


@pytest.mark.parametrize("setup", [True, False])
@pytest.mark.parametrize("make_cfg", [Cfg, ModelCfg], ids=["dataclass", "native-model"])
def test_cfg_device_setup_precedes_experiment_and_can_be_explicitly_skipped(
    rig: Rig, setup: bool, make_cfg: Callable[..., Cfg | ModelCfg]
) -> None:
    settings: dict[str, DeviceInfo] = {
        "z": FakeDeviceInfo(address="none", output="on", value=0.1),
        "a": FakeDeviceInfo(address="none", output="on", value=0.2),
    }
    device = FakeDevice(fast_mode=True)
    manager = DeviceManager()
    # Aliases make final value expose native mapping order without mocking setup.
    manager.register_device("z", device)
    manager.register_device("a", device)
    rig.devices.manager = manager
    cfg = make_cfg(dev=settings)
    saved: list[Completed[Cfg | ModelCfg, int]] = []

    def acquire(run: Run[Cfg | ModelCfg]) -> int:
        assert rig.devices.settings == ([settings] if setup else [])
        assert device.get_value() == (0.2 if setup else 0.0)
        assert device.get_output() == ("on" if setup else "off")
        assert run.device_setup == settings
        return 1

    def save(source: Completed[Cfg | ModelCfg, int], path: Path) -> None:
        saved.append(source)
        path.write_text(str(source.result), encoding="utf-8")

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        if state.index:
            return Done()
        yield from env.run(acquire, cfg, save=save, setup_devices=setup)
        state.index = 1
        return Next(1, state)

    rig.start(step)
    assert rig.engine.execute("run").lifecycle == "done"
    assert saved[0].cfg.dev == settings
    assert cfg.dev == settings


@pytest.mark.parametrize("control", ["pause", "stop"])
def test_control_between_native_device_setups_never_acquires_or_saves(
    rig: Rig, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    first, second = FakeDevice(fast_mode=True), FakeDevice(fast_mode=True)
    manager = DeviceManager()
    manager.register_device("first", first)
    manager.register_device("second", second)
    rig.devices.manager = manager
    original_setup = first.setup
    signals: list[Event | None] = []

    def setup(
        info: FakeDeviceInfo, *, progress: bool = True, stop_event: Event | None = None
    ) -> None:
        signals.append(stop_event)
        original_setup(info, progress=progress, stop_event=stop_event)
        if control == "pause":
            rig.engine.pause("run")
        else:
            rig.engine.stop("run")

    monkeypatch.setattr(first, "setup", setup)

    def acquire(_run: Run[Cfg]) -> int:
        raise AssertionError("Cancellation during setup must not acquire")

    def save(_source: Completed[Cfg, int], _path: Path) -> None:
        raise AssertionError("Cancellation during setup must not save")

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State
    ) -> Step[State, int]:
        yield from env.run(
            acquire,
            Cfg(
                dev={
                    "first": FakeDeviceInfo(address="none", value=0.1),
                    "second": FakeDeviceInfo(address="none", value=0.2),
                }
            ),
            save=save,
        )
        raise AssertionError("Cancelled step must not receive an outcome")

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("paused" if control == "pause" else "stopped")
    assert first.get_value() == 0.1
    assert second.get_value() == 0.0
    assert len(signals) == 1
    assert signals[0] is not None and signals[0].is_set()
    assert rig.engine.records("run") == ()
    assert "step_discarded" in rig.kinds()
    assert "experiment_failed" not in rig.kinds()
    assert not tuple(rig.paths.data_root.glob("iter/*/runs/*"))


@pytest.mark.parametrize(
    "info",
    [
        FakeDeviceInfo(address="wrong-address", value=0.1),
        YOKOGS200Info(address="none", value=0.1),
    ],
    ids=["address", "driver-model"],
)
def test_native_driver_validation_fails_the_engine_without_acquiring(
    rig: Rig, info: DeviceInfo
) -> None:
    device = FakeDevice(fast_mode=True)
    manager = DeviceManager()
    manager.register_device("flux", device)
    rig.devices.manager = manager

    def acquire(_run: Run[Cfg]) -> int:
        raise AssertionError("Driver validation failure must not acquire")

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.run(acquire, Cfg(dev={"flux": info}), save=save_integer)
        return Next(1, state)

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "Trying to setup device" in str(status.reason)
    event = next(event for event in rig.events() if event["kind"] == "step_failed")
    error = event["error"]
    assert isinstance(error, dict)
    assert error["type"] == "builtins.RuntimeError"
    assert "Trying to setup device" in str(error["message"])
    assert device.get_value() == 0.0
    assert "experiment_failed" not in rig.kinds()
    assert rig.engine.records("run") == ()
    assert not tuple(rig.paths.data_root.glob("iter/*/runs/*"))


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


def test_device_setup_error_is_not_recoverable_schedule_failure(rig: Rig) -> None:
    settings = {"flux": FakeDeviceInfo(address="none", value=0.1)}
    observed: list[int] = []

    def setup(_signal: StopSignal) -> None:
        rig.engine.pause("run")
        raise OSError("setup failed")

    def acquire(_run: Run[Cfg]) -> int:
        observed.append(1)
        return 1

    def step(
        env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, state: State
    ) -> Step[State, int]:
        yield from env.run(acquire, Cfg(dev=settings), save=save_integer)
        return Next(1, state)

    rig.devices.on_setup = setup
    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "failed"
    assert "setup failed" in str(status.reason)
    assert observed == []
    assert "experiment_failed" not in rig.kinds()
    assert "step_discarded" not in rig.kinds()
