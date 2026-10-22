"""Device capture, core RunContext and explicit setup through async Engine."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.experiment.workflows import Engine, EnginePorts, Live, WorkflowEnv

from ._engine_fakes import Cfg, Plan, Point, Rig, Tunables, completed


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    rig = Rig(tmp_path, native=True)
    yield rig
    rig.close()


class SnapshotDriver(FakeDevice):
    """Anonymous memory driver with observable reads and I/O failures."""

    info_model = FakeDeviceInfo

    def __new__(cls) -> Self:
        return super().__new__(cls)

    def __init__(self) -> None:
        super().__init__(fast_mode=True)
        self.reads = 0
        self.error: OSError | None = None

    def get_info(self) -> FakeDeviceInfo:
        self.reads += 1
        if self.error is not None:
            raise self.error
        return super().get_info()


def test_each_env_run_reads_all_drivers_before_core_callback(rig: Rig) -> None:
    selected, unused = SnapshotDriver(), SnapshotDriver()
    selected.set_value(0.1)
    unused.set_value(0.2)
    rig.devices.drivers.update(flux=selected, unused=unused)
    callbacks: list[int] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        assert (selected.reads, unused.reads) == (len(callbacks) + 1,) * 2
        assert context.devices == {"flux": selected, "unused": unused}
        selected.set_value(0.9)
        unused.set_value(0.8)
        callbacks.append(1)
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        for _ in range(2):
            completed(await env.run(acquire, Cfg(dev={"flux": "flux"})))
        await env.commit(Point(value=2))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert (selected.reads, unused.reads) == (2, 2)
    assert callbacks == [1, 1]
    document = json.loads(
        (rig.paths.data_root / "commits/000001/record.json").read_text()
    )
    operations = document["operations"]
    journal_operations = [
        event["operation"]
        for event in rig.events()
        if event["kind"] == "operation_finished"
    ]
    assert journal_operations == operations
    assert [operation["kind"] for operation in operations] == ["run", "run"]
    assert operations[0]["device_snapshot"]["flux"]["value"] == 0.1
    assert operations[0]["device_snapshot"]["unused"]["value"] == 0.2
    assert operations[1]["device_snapshot"]["flux"]["value"] == 0.9
    assert operations[1]["device_snapshot"]["unused"]["value"] == 0.8


def test_snapshot_read_failure_fails_before_callback_and_save(rig: Rig) -> None:
    broken = SnapshotDriver()
    broken.error = OSError("unused driver read failed")
    rig.devices.drivers["unused"] = broken
    callbacks: list[str] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        callbacks.append("acquire")
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        await env.run(acquire, Cfg())

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "failed"
    assert final.reason == "OSError: unused driver read failed"
    assert callbacks == []
    assert broken.reads == 1
    assert list(rig.paths.data_root.rglob("*.h5")) == []


@pytest.mark.parametrize("missing", [True, False], ids=["missing", "explicit-empty"])
def test_env_run_distinguishes_missing_devices_from_empty_port(
    rig: Rig, missing: bool
) -> None:
    board = white_noise()
    rig.engine = Engine(
        EnginePorts(
            rig.plots,
            rig.progress,
            rig.clock,
            context=7,
            devices=None if missing else rig.devices,
            soc=board.soc,
            soccfg=board.soccfg,
        )
    )
    callbacks: list[str] = []

    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        callbacks.append("acquire")
        assert context.devices == {}
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        await env.run(acquire, Cfg())

    rig.start(script)
    final = rig.engine.execute("run")
    if missing:
        assert final.lifecycle == "failed"
        assert "Missing workflow capability: devices" in str(final.reason)
        assert callbacks == []
    else:
        assert final.lifecycle == "done", final.reason
        assert callbacks == ["acquire"]


@pytest.mark.parametrize("uses_device", [False, True])
def test_role_names_are_resolved_only_when_core_callback_uses_device(
    rig: Rig,
    uses_device: bool,
) -> None:
    def acquire(cfg: Cfg, *, context: RunContext) -> int:
        if uses_device:
            _ = context.devices[cfg.dev["flux"]]
        return 1

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        await env.run(acquire, Cfg(dev={"flux": "unavailable"}))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == ("failed" if uses_device else "done"), final.reason
    if uses_device:
        assert final.reason == "KeyError: 'unavailable'"
    assert rig.devices.values == []


def test_live_cfg_getter_is_detached_and_setup_preserves_complete_model(
    rig: Rig,
) -> None:
    driver = SnapshotDriver()
    driver.set_value(0.1)
    rig.devices.drivers["flux"] = driver
    cfg = FakeDeviceInfo(address=driver.address, value=0.25, rampstep=0.05, output="on")

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        actual = env.get_device_cfg("flux")
        assert isinstance(actual, FakeDeviceInfo)
        assert actual.value == 0.1
        actual.value = 99
        assert env.get_device_cfg("flux").model_dump()["value"] == 0.1
        completed(await env.setup_device("flux", cfg))

    rig.start(script)
    final = rig.engine.execute("run")
    assert final.lifecycle == "done", final.reason
    assert rig.devices.values == [("flux", 0.25)]
    assert rig.devices.configs == [cfg]
