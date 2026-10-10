"""Engine Start/Resume retain detached native device provenance (D272)."""

from datetime import timedelta
from pathlib import Path

import pytest
from zcu_tools.device import DeviceInfo, FakeDeviceInfo
from zcu_tools.experiment.workflows import DeviceSnapshot, Done, Next, Step, WorkflowEnv

from ._engine_fakes import Plan, Rig, State, Tunables


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
