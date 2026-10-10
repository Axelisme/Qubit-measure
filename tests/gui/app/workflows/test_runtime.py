"""Session driver access under the workflow worker's retained RUN lease."""

from collections.abc import Callable
from pathlib import Path
from threading import Event

import pytest
from pydantic import Field, ValidationError
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import Done, Engine, EnginePorts, Run, Step, WorkflowEnv
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.session.services.device import ConnectDeviceRequest, DeviceService

from tests.experiment.workflows._engine_fakes import Cfg, Plan, Rig, State, Tunables, save_integer

from ._start_fakes import StartHarness


class ExtendedInfo(FakeDeviceInfo):
    """Driver-owned settings beyond the framework's knowledge."""

    labels: list[str] = Field(default_factory=lambda: ["live"])


class RecordingDriver(FakeDevice):
    """Real native setup with observable cfg, event, reads and injected I/O cause."""

    info_model = ExtendedInfo

    def __init__(self) -> None:
        super().__init__(fast_mode=True)
        self.info = ExtendedInfo(address=self.address, rampstep=1.0)
        self.reads = 0
        self.calls: list[tuple[FakeDeviceInfo, Event | None]] = []
        self.read_error: OSError | None = None
        self.setup_error: OSError | None = None
        self.on_setup: Callable[[], None] | None = None

    def get_info(self) -> ExtendedInfo:
        self.reads += 1
        if self.read_error is not None:
            raise self.read_error
        self.info.value = self.value
        self.info.output = self.output
        return self.info

    def setup(self, cfg: FakeDeviceInfo, *, progress: bool = True, stop_event: Event | None = None) -> None:
        self.calls.append((cfg, stop_event))
        if self.setup_error is not None:
            raise self.setup_error
        if self.on_setup is not None:
            self.on_setup()
        super().setup(cfg, progress=False, stop_event=stop_event)
        if isinstance(cfg, ExtendedInfo):
            self.info = cfg.model_copy(deep=True)


@pytest.fixture
def connected() -> tuple[StartHarness, DeviceService, RecordingDriver, SessionDevices]:
    harness = StartHarness()
    driver = RecordingDriver()
    service = DeviceService(harness.bus, harness.state, harness.gate, harness.background,
                            harness.runner, harness.handles, driver_factory=lambda type_name, address: driver)
    service.start_connect_device(ConnectDeviceRequest("FakeDevice", "flux", driver.address))
    harness.complete_segment()
    # The connect observation belongs to the cache; later reads may reuse a model.
    driver.info = driver.info.model_copy(deep=True)
    driver.reads = 0
    return harness, service, driver, SessionDevices(service)


def test_live_getter_returns_detached_actual_subclass_without_updating_cache(connected) -> None:
    _, service, driver, port = connected
    driver.set_value(0.12)
    cfg = port.get_device_cfg("flux")
    assert isinstance(cfg, ExtendedInfo)
    assert cfg.value == 0.12 and cfg.value_unit == "native"
    cfg.labels.append("caller")
    cfg.value = 0.7
    driver.set_value(0.2)
    fresh = port.get_device_cfg("flux")
    assert isinstance(fresh, ExtendedInfo)
    assert fresh.labels == ["live"] and fresh.value == 0.2
    assert driver.reads == 2
    assert service.capture_native_device_info()["flux"].model_dump()["value"] == 0.0


@pytest.mark.parametrize("update", [{"value": "invalid"}, {"value_unit": "V"}, {"output": "invalid"}])
def test_complete_setup_revalidates_unchecked_model_copy_before_driver_io(connected, update: dict[str, object]) -> None:
    _, _, driver, port = connected
    invalid = driver.info.model_copy(update=update)
    with pytest.raises(ValidationError):
        port.setup_device("flux", invalid, StopSignal())
    assert driver.calls == [] and driver.reads == 0
    assert driver.value == 0.0


def test_complete_setup_preserves_all_fields_subclass_and_signal_event(connected) -> None:
    _, service, driver, port = connected
    cfg = ExtendedInfo(address=driver.address, value=0.2, rampstep=0.05, output="on", labels=["configured"])
    signal = StopSignal()
    port.setup_device("flux", cfg, signal)
    applied, event = driver.calls[0]
    assert isinstance(applied, ExtendedInfo)
    assert applied.model_dump() == cfg.model_dump()
    assert event is signal.event
    assert driver.value == 0.2 and driver.output == "on"
    assert service.capture_native_device_info()["flux"].model_dump()["value"] == 0.0


def test_cancelled_native_setup_does_not_apply_value(connected) -> None:
    _, _, driver, port = connected
    signal = StopSignal()
    signal.set()
    port.setup_device("flux", driver.info.model_copy(update={"value": 0.5}), signal)
    assert driver.calls[0][1] is signal.event
    assert driver.value == 0.0


@pytest.mark.parametrize("operation", ["get", "setup"])
def test_unknown_name_is_rejected_at_use_and_names_requested_device(connected, operation: str) -> None:
    _, _, driver, port = connected
    with pytest.raises(ValueError, match="missing"):
        if operation == "get":
            port.get_device_cfg("missing")
        else:
            port.setup_device("missing", driver.info, StopSignal())
    assert driver.calls == [] and driver.reads == 0


def test_wrong_address_and_io_cause_propagate_without_retry(connected) -> None:
    _, _, driver, port = connected
    with pytest.raises(RuntimeError, match="address"):
        port.setup_device("flux", driver.info.model_copy(update={"address": "wrong"}), StopSignal())
    assert driver.value == 0.0
    cause = OSError("native write failed")
    driver.setup_error = cause
    with pytest.raises(OSError) as caught:
        port.setup_device("flux", driver.info, StopSignal())
    assert caught.value is cause and len(driver.calls) == 2


def test_worker_captures_each_run_after_setup_before_callback(connected, tmp_path: Path) -> None:
    _, _, driver, port = connected
    rig = Rig(tmp_path)
    rig.engine = Engine(EnginePorts(rig.plots, rig.progress, rig.clock, context=7, devices=port))
    seen: list[float] = []

    def acquire(run: Run[Cfg]) -> int:
        assert driver.reads == len(seen) + 1
        value = run.device_snapshot["flux"]["value"]
        assert isinstance(value, float)
        seen.append(value)
        return 1

    def step(env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State) -> Step[State, int]:
        for value in (0.1, 0.3):
            yield from env.setup_device("flux", driver.info.model_copy(update={"value": value}))
            yield from env.run(acquire, Cfg(), save=save_integer)
        return Done()

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == "done", status.reason
    assert seen == [0.1, 0.3]


@pytest.mark.parametrize("empty", [False, True], ids=["read-failure", "zero-devices"])
def test_capture_failure_stops_before_callback_but_zero_devices_is_explicit(connected, tmp_path: Path, empty: bool) -> None:
    harness, _, driver, port = connected
    if empty:
        port = SessionDevices(harness.services.device)
    else:
        driver.read_error = OSError("capture failed")
    rig = Rig(tmp_path)
    rig.engine = Engine(EnginePorts(rig.plots, rig.progress, rig.clock, context=7, devices=port))
    entered: list[object] = []

    def acquire(run: Run[Cfg]) -> int:
        entered.append(run.device_snapshot)
        return 1

    def step(env: WorkflowEnv[int], _plan: Plan, _tun: Tunables, _state: State) -> Step[State, int]:
        yield from env.run(acquire, Cfg(), save=save_integer)
        return Done()

    rig.start(step)
    status = rig.engine.execute("run")
    assert status.lifecycle == ("done" if empty else "failed"), status.reason
    assert entered == ([{}] if empty else [])
    if not empty:
        assert status.reason == "OSError: capture failed"
        assert driver.reads == 1
