"""Disconnected-device rename through the owner and its published projections."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.device.fake import FakeDevice
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.events import DeviceChangedPayload
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import DeviceMemoryInfo
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import ConnectDeviceRequest, DeviceService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.services.project_settings import ProjectSettingsService
from zcu_tools.gui.session.services.value_sources import ValueSourceBinder
from zcu_tools.gui.session.state import DeviceStatus, SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.value_lookup import MissingValue, ValueRegistry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._qt_waits import qt_until
from tests.gui.session.services._device_fakes import FakeDeviceRegistry
from tests.gui.session.services._device_waits import DeviceWorkers


@dataclass
class RenameSession:
    """Real owners with a synthetic driver and controlled completion delivery."""

    device: DeviceService
    state: SessionState
    bus: BaseEventBus
    values: ValueRegistry
    binder: ValueSourceBinder
    settings: ProjectSettingsService
    workers: DeviceWorkers


@pytest.fixture
def rename_session(qapp, tmp_path: Path) -> Iterator[RenameSession]:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    handles = OperationHandles()
    workers = DeviceWorkers()
    background = workers.background()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(gate, handles, progress, background, bus)
    device = DeviceService(
        bus,
        state,
        gate,
        background,
        runner,
        handles,
        driver_factory=lambda type_name, address: FakeDevice(fast_mode=True),
        device_registry=FakeDeviceRegistry(),
    )
    device.register_remembered_devices(
        [
            DeviceMemoryInfo(type_name="FakeDevice", name="old", address="addr"),
            DeviceMemoryInfo(type_name="FakeDevice", name="occupied", address="other"),
        ]
    )
    values = ValueRegistry()
    binder = ValueSourceBinder(state=state, bus=bus, registry=values)
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    settings = ProjectSettingsService(
        context,
        device,
        state,
        result_root=tmp_path / "result",
        database_root=tmp_path / "Database",
    )
    try:
        yield RenameSession(device, state, bus, values, binder, settings, workers)
    finally:
        workers.close()


def _rename(session: RenameSession, old_name: str, new_name: str) -> None:
    # Missing public capability must be an assertion failure in the red run.
    assert hasattr(session.device, "rename_device"), "DeviceService lacks rename_device"
    session.device.rename_device(old_name, new_name)


@pytest.mark.parametrize(
    "new_name", ["new", "with space", "a.b", "a/b", "$flux", "class", " 新名 "]
)
def test_rename_publishes_new_name_to_live_values_and_preferences(
    rename_session: RenameSession,
    new_name: str,
) -> None:
    session = rename_session
    old = session.device.get_device_snapshot("old")
    observed: list[tuple[str, ...]] = []
    session.bus.subscribe(
        DeviceChangedPayload,
        lambda _payload: observed.append(
            tuple(entry.name for entry in session.device.list_devices())
        ),
    )

    _rename(session, "old", new_name)

    assert session.device.get_device_snapshot("old") is None
    renamed = session.device.get_device_snapshot(new_name)
    assert renamed is not None and old is not None
    assert (renamed.name, renamed.type_name, renamed.address, renamed.status) == (
        new_name,
        old.type_name,
        old.address,
        DeviceStatus.MEMORY_ONLY,
    )
    assert renamed.info == old.info and renamed.error == old.error
    assert observed == [tuple(sorted((new_name, "occupied")))]
    with pytest.raises(MissingValue):
        session.values.get_as("device.old.name", str)
    assert session.values.get_as(f"device.{new_name}.name", str) == new_name
    assert session.values.get_as(f"device.{new_name}.address", str) == "addr"
    assert session.binder.snapshot().values[f"device.{new_name}.name"] == new_name
    remembered = session.settings.capture_settings(left_panel_width=560).devices
    assert {(entry.name, entry.address) for entry in remembered} == {
        (new_name, "addr"),
        ("occupied", "other"),
    }


@pytest.mark.parametrize("new_name", ["", " \t\n", "occupied"])
def test_rejected_rename_leaves_all_published_projections_unchanged(
    rename_session: RenameSession,
    new_name: str,
) -> None:
    session = rename_session
    before = session.device.list_device_snapshots()
    values_before = session.binder.snapshot()
    settings_before = session.settings.capture_settings(left_panel_width=560)
    changed: list[DeviceChangedPayload] = []
    session.bus.subscribe(DeviceChangedPayload, changed.append)

    with pytest.raises(InvalidInputError):
        _rename(session, "old", new_name)

    assert session.device.list_device_snapshots() == before
    assert session.binder.snapshot() == values_before
    assert session.settings.capture_settings(left_panel_width=560) == settings_before
    assert changed == []


def test_same_name_rename_is_a_noop(rename_session: RenameSession) -> None:
    session = rename_session
    before = session.binder.snapshot()
    changed: list[DeviceChangedPayload] = []
    session.bus.subscribe(DeviceChangedPayload, changed.append)

    _rename(session, "old", "old")

    assert changed == []
    assert session.binder.snapshot() == before
    assert {entry.name for entry in session.device.list_devices()} == {
        "occupied",
        "old",
    }


@pytest.mark.parametrize("new_name", ["new", "old"])
@pytest.mark.parametrize(
    "complete_connect", [False, True], ids=["inflight", "connected"]
)
def test_rename_requires_disconnected_idle_device_even_for_same_name(
    rename_session: RenameSession,
    new_name: str,
    complete_connect: bool,
) -> None:
    session = rename_session
    session.device.start_connect_device(
        ConnectDeviceRequest(
            type_name="YOKOGS200",
            name="old",
            address="addr",
        )
    )
    if complete_connect:
        qt_until(
            lambda: not session.device.get_active_device_operations(),
            label="synthetic device connect terminal",
        )
        connected = session.device.get_device_snapshot("old")
        assert connected is not None and connected.status is DeviceStatus.CONNECTED
    before = session.device.list_device_snapshots()
    values_before = session.binder.snapshot()

    with pytest.raises(FailedPreconditionError, match="[Dd]isconnect|[Ii]n.flight"):
        _rename(session, "old", new_name)

    assert session.device.list_device_snapshots() == before
    assert session.binder.snapshot() == values_before


def test_rename_unknown_device_is_rejected(rename_session: RenameSession) -> None:
    with pytest.raises(FailedPreconditionError, match="missing"):
        _rename(rename_session, "missing", "new")
