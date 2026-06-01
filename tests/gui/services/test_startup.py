from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import EventBus, GuiEvent
from zcu_tools.gui.services.startup import (
    StartupConnectionRequest,
    StartupProjectRequest,
    StartupService,
)
from zcu_tools.gui.services.startup_persistence import (
    STARTUP_VERSION,
    PersistedDeviceEntry,
    PersistedStartup,
)
from zcu_tools.gui.state import DeviceState, DeviceStatus, State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_service() -> tuple[StartupService, MagicMock, MagicMock, MagicMock]:
    context = MagicMock()
    devices = MagicMock()
    persistence = MagicMock()
    state = State(MagicMock())
    bus = EventBus()
    svc = StartupService(context, devices, persistence, state, bus)
    return svc, context, devices, persistence


def test_apply_project_passes_paths_through_without_rescoping() -> None:
    """apply_project does NOT scope under chip/qub — the caller (derive_project_
    paths) already did. It must pass result_dir/database_path through verbatim,
    else the chip/qub segment would be doubled."""
    svc, context, _, persistence = _make_service()
    req = StartupProjectRequest(
        "chip", "qubit", "res", "/tmp/result/chip/qubit", "/tmp/db/chip/qubit"
    )

    svc.apply_project(req)

    args = context.set_startup_context.call_args.args
    assert isinstance(args[0], MetaDict)
    assert isinstance(args[1], ModuleLibrary)
    assert args[2:] == (
        "chip",
        "qubit",
        "res",
        "/tmp/result/chip/qubit",
        "/tmp/db/chip/qubit",
    )
    context.setup_project.assert_called_once_with("/tmp/result/chip/qubit")
    persistence.update_project.assert_called_once_with(
        chip_name="chip",
        qub_name="qubit",
        res_name="res",
        result_dir="/tmp/result/chip/qubit",
        database_path="/tmp/db/chip/qubit",
    )


def test_apply_project_empty_result_dir_skips_setup_project() -> None:
    """An empty result_dir (DRAFT context) does not trigger setup_project."""
    svc, context, _, _ = _make_service()
    req = StartupProjectRequest("chip", "qubit", "res", "", "")

    svc.apply_project(req)

    context.setup_project.assert_not_called()


def test_derive_project_paths_scopes_under_chip_qubit() -> None:
    from zcu_tools.gui.services.startup import derive_project_paths

    result_dir, database_path = derive_project_paths("Q5_2D", "Q1", "/root")
    assert result_dir == "/root/result/Q5_2D/Q1"
    assert database_path == "/root/Database/Q5_2D/Q1"


def test_restore_devices_registers_memory_only_entries() -> None:
    svc, _, devices, persistence = _make_service()
    persistence.load.return_value = PersistedStartup(
        version=STARTUP_VERSION,
        chip_name="",
        qub_name="",
        res_name="",
        result_dir="",
        database_path="",
        ip="host",
        port=8887,
        devices=[PersistedDeviceEntry("FakeDevice", "flux", "addr")],
    )

    svc.restore_devices()

    (entries,) = devices.register_remembered_devices.call_args.args
    assert entries[0].name == "flux"
    assert entries[0].address == "addr"


def _dev(name: str, *, remember: bool) -> DeviceState:
    return DeviceState(
        name=name,
        type_name="FakeDevice",
        address=f"{name}-addr",
        status=DeviceStatus.CONNECTED,
        remember=remember,
    )


def _empty_persisted() -> PersistedStartup:
    return PersistedStartup(
        version=STARTUP_VERSION,
        chip_name="",
        qub_name="",
        res_name="",
        result_dir="",
        database_path="",
        ip="host",
        port=8887,
        devices=[],
    )


def test_device_changed_projects_remembered_set_onto_persistence() -> None:
    context = MagicMock()
    devices = MagicMock()
    persistence = MagicMock()
    persistence.get_current.return_value = _empty_persisted()
    state = State(MagicMock())
    bus = EventBus()
    StartupService(context, devices, persistence, state, bus)

    # Two remembered + one not-remembered → only the remembered ones persist.
    state.put_device(_dev("flux", remember=True))
    state.put_device(_dev("probe", remember=True))
    state.put_device(_dev("scratch", remember=False))
    from zcu_tools.gui.event_bus import DeviceChangedPayload

    bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload(name="flux"))

    (entries,) = persistence.replace_devices.call_args.args
    assert sorted(e.name for e in entries) == ["flux", "probe"]


def test_device_changed_diff_guard_skips_when_remembered_set_unchanged() -> None:
    context = MagicMock()
    devices = MagicMock()
    persistence = MagicMock()
    persistence.get_current.return_value = PersistedStartup(
        version=STARTUP_VERSION,
        chip_name="",
        qub_name="",
        res_name="",
        result_dir="",
        database_path="",
        ip="host",
        port=8887,
        devices=[PersistedDeviceEntry("FakeDevice", "flux", "flux-addr")],
    )
    state = State(MagicMock())
    bus = EventBus()
    StartupService(context, devices, persistence, state, bus)
    state.put_device(_dev("flux", remember=True))

    from zcu_tools.gui.event_bus import DeviceChangedPayload

    # The remembered set already matches what's persisted → no disk write.
    bus.emit(GuiEvent.DEVICE_CHANGED, DeviceChangedPayload(name="flux"))

    persistence.replace_devices.assert_not_called()


def test_connection_request_validates_port() -> None:
    with pytest.raises(ValueError, match="port"):
        StartupConnectionRequest(ip="host", port=0)
