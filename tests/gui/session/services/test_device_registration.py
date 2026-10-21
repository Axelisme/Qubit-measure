"""Runtime registration and all-driver teardown through the session owner."""

from dataclasses import dataclass

import pytest
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.adapters.thread_pool_background import (
    ThreadPoolBackgroundExecutor,
)
from zcu_tools.gui.session.events import DeviceChangedPayload
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationInfrastructure,
    OperationRunner,
)
from zcu_tools.gui.session.ports import (
    DeviceMemoryInfo,
    OperationConflictError,
    OperationKind,
)
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import DeviceStatus, SessionState
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._progress_fakes import DirectProgressTransport


@dataclass
class DeviceSession:
    service: DeviceService
    registry: DeviceManager
    state: SessionState
    gate: RunBlocksHardwareGate
    facts: list[DeviceChangedPayload]


@pytest.fixture
def session():
    bus = BaseEventBus()
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    handles = OperationHandles()
    background = ThreadPoolBackgroundExecutor(ManualOwnerScheduler())
    progress = ProgressService(DirectProgressTransport())
    registry = DeviceManager()
    service = DeviceService(
        bus,
        state,
        OperationInfrastructure(
            gate=gate,
            background=background,
            runner=OperationRunner(gate, handles, progress, background, bus),
            handles=handles,
        ),
        device_registry=registry,
    )
    facts: list[DeviceChangedPayload] = []
    bus.subscribe(DeviceChangedPayload, facts.append)
    try:
        yield DeviceSession(service, registry, state, gate, facts)
    finally:
        assert background.quiesce(timeout=1.0)
        registry.close_all_devices()


def test_register_publishes_live_cache_and_replaces_remembered_row(
    session: DeviceSession,
):
    svc = session.service
    svc.register_remembered_devices([DeviceMemoryInfo("YOKOGS200", "bias", "old")])
    session.facts.clear()
    driver = FakeDevice(fast_mode=True)
    driver.set_value(1.25)

    svc.register_device("bias", driver)

    row = svc.get_device_snapshot("bias")
    assert row is not None
    assert row.status is DeviceStatus.CONNECTED
    assert row.type_name == "FakeDevice"
    assert row.address == "none"
    remembered = session.state.get_device("bias")
    assert remembered is not None and remembered.remember is True
    assert row.info == driver.get_info()
    assert session.registry.get_all_devices() == {"bias": driver}
    assert [fact.name for fact in session.facts] == ["bias"]
    assert svc.get_cached_device_value("bias") == 1.25
    with pytest.raises(FailedPreconditionError, match="already"):
        svc.register_device("bias", FakeDevice(fast_mode=True))
    assert session.registry.get_device("bias") is driver


@pytest.mark.parametrize("kind", ["run", OperationKind.DEVICE_CONNECT])
def test_register_rejects_active_hardware_work_without_publishing(
    session: DeviceSession, kind: str
):
    session.gate.register(
        42, kind, owner_id="bias", origin_kind="user", note="busy", resource_id="bias"
    )
    try:
        with pytest.raises(OperationConflictError):
            session.service.register_device("bias", FakeDevice(fast_mode=True))
        assert session.service.get_device_snapshot("bias") is None
        assert session.registry.get_all_devices() == {}
        assert session.facts == []
    finally:
        session.gate.release(42)


class UnreadableDevice(FakeDevice):
    info_model = FakeDeviceInfo

    def get_info(self) -> FakeDeviceInfo:
        raise RuntimeError("unreadable candidate")


def test_failed_registration_keeps_prior_row_and_caller_ownership(
    session: DeviceSession,
):
    svc = session.service
    svc.register_remembered_devices([DeviceMemoryInfo("YOKOGS200", "bias", "old")])
    before = svc.get_device_snapshot("bias")
    version = session.state.version.get("device:bias")
    session.facts.clear()
    with pytest.raises(RuntimeError, match="unreadable candidate"):
        svc.register_device("bias", UnreadableDevice(fast_mode=True))
    assert svc.get_device_snapshot("bias") == before
    assert session.state.version.get("device:bias") == version
    assert session.registry.get_all_devices() == {}
    assert session.facts == []


class FailingCloseDevice(FakeDevice):
    info_model = FakeDeviceInfo

    fail_close = True
    close_attempts = 0

    def close(self) -> None:
        self.close_attempts += 1
        if self.fail_close:
            raise RuntimeError("cannot close bias")
        super().close()


def test_close_all_attempts_every_driver_and_publishes_partial_failure(
    session: DeviceSession,
):
    bad = FailingCloseDevice()
    good = FakeDevice(fast_mode=True)
    session.registry.register_device("bad", bad)
    session.registry.register_device("good", good)
    session.service.publish_initial_devices()
    session.facts.clear()
    try:
        with pytest.raises(ExceptionGroup, match="device") as error:
            session.service.close_all_devices()
        assert "cannot close bias" in str(error.value.exceptions[0])
        assert bad.close_attempts == 1
        assert session.registry.get_all_devices() == {"bad": bad}
        rows = {row.name: row for row in session.service.list_device_snapshots()}
        assert rows["good"].status is DeviceStatus.MEMORY_ONLY
        assert rows["good"].error is None
        assert rows["bad"].status is DeviceStatus.CONNECTED
        assert "cannot close bias" in (rows["bad"].error or "")
        assert {fact.name for fact in session.facts} == {"bad", "good"}
        bad.fail_close = False
        session.service.close_all_devices()
        assert bad.close_attempts == 2
        assert session.registry.get_all_devices() == {}
    finally:
        bad.fail_close = False
