from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.persistence_types import (
    PersistedDeviceEntry,
    PersistedStartup,
)
from zcu_tools.gui.app.main.services.startup import (
    StartupConnectionRequest,
    StartupProjectRequest,
    StartupService,
)
from zcu_tools.gui.app.main.state import DeviceState, DeviceStatus, State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_service() -> tuple[StartupService, MagicMock, MagicMock, State]:
    context = MagicMock()
    devices = MagicMock()
    state = State(MagicMock())
    svc = StartupService(context, devices, state)
    return svc, context, devices, state


def test_apply_project_passes_paths_through_and_records_prefs() -> None:
    """apply_project does NOT scope under chip/qub — the caller already did. It
    passes paths through verbatim and records them as the remembered prefs (in
    State; persisted at close)."""
    svc, context, _, state = _make_service()
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
    # Recorded as prefs (not written to disk here).
    prefs = state.startup_prefs
    assert prefs.chip_name == "chip"
    assert prefs.qub_name == "qubit"
    assert prefs.result_dir == "/tmp/result/chip/qubit"
    assert prefs.database_path == "/tmp/db/chip/qubit"


def test_apply_project_empty_result_dir_skips_setup_project() -> None:
    svc, context, _, _ = _make_service()
    req = StartupProjectRequest("chip", "qubit", "res", "", "")

    svc.apply_project(req)

    context.setup_project.assert_not_called()


def test_remember_connection_records_prefs() -> None:
    svc, _, _, state = _make_service()
    svc.remember_connection(StartupConnectionRequest(ip="10.0.0.2", port=1234))
    assert state.startup_prefs.ip == "10.0.0.2"
    assert state.startup_prefs.port == 1234


def test_derive_project_paths_scopes_under_chip_qubit() -> None:
    from datetime import datetime

    from zcu_tools.gui.app.main.services.startup import derive_project_paths

    result_dir, database_path = derive_project_paths("Q5_2D", "Q1", "/root")
    assert result_dir == "/root/result/Q5_2D/Q1"
    today = datetime.today()
    yy, mm, dd = today.strftime("%Y-%m-%d").split("-")
    assert database_path == f"/root/Database/Q5_2D/Q1/{yy}/{mm}/Data_{mm}{dd}"


def _dev(name: str, *, remember: bool) -> DeviceState:
    return DeviceState(
        name=name,
        type_name="FakeDevice",
        address=f"{name}-addr",
        status=DeviceStatus.CONNECTED,
        remember=remember,
    )


def test_capture_startup_projects_remembered_devices_and_prefs() -> None:
    """capture_startup re-projects the remember=True device set from State and
    composes it with the prefs + the given left-panel width into a memento."""
    svc, _, _, state = _make_service()
    svc.remember_connection(StartupConnectionRequest(ip="host", port=8887))
    state.put_device(_dev("flux", remember=True))
    state.put_device(_dev("probe", remember=True))
    state.put_device(_dev("scratch", remember=False))

    memento = svc.capture_startup(left_panel_width=321)

    assert isinstance(memento, PersistedStartup)
    assert memento.ip == "host"
    assert memento.left_panel_width == 321
    assert sorted(e.name for e in memento.devices) == ["flux", "probe"]


def test_restore_startup_seeds_prefs_and_registers_devices() -> None:
    svc, _, devices, state = _make_service()
    data = PersistedStartup(
        chip_name="chip",
        qub_name="qub",
        res_name="res",
        ip="host",
        port=9999,
        devices=(
            PersistedDeviceEntry(type_name="FakeDevice", name="flux", address="a"),
        ),
        left_panel_width=222,
    )

    svc.restore_startup(data)

    # Prefs seeded (so the setup dialog prefills) — project NOT auto-applied.
    assert state.startup_prefs.chip_name == "chip"
    assert state.startup_prefs.port == 9999
    assert state.startup_prefs.left_panel_width == 222
    (entries,) = devices.register_remembered_devices.call_args.args
    assert entries[0].name == "flux"


def test_connection_request_validates_port() -> None:
    with pytest.raises(ValueError, match="port"):
        StartupConnectionRequest(ip="host", port=0)
