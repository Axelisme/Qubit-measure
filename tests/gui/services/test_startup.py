from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.device import ConnectDeviceRequest
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
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_service() -> tuple[StartupService, MagicMock, MagicMock, MagicMock]:
    context = MagicMock()
    devices = MagicMock()
    persistence = MagicMock()
    return StartupService(context, devices, persistence), context, devices, persistence


def test_apply_project_constructs_draft_dependencies_and_persists() -> None:
    svc, context, _, persistence = _make_service()
    req = StartupProjectRequest("chip", "qubit", "res", "/tmp/result", "/tmp/db")

    svc.apply_project(req)

    args = context.set_startup_context.call_args.args
    assert isinstance(args[0], MetaDict)
    assert isinstance(args[1], ModuleLibrary)
    assert args[2:] == ("chip", "qubit", "res", "/tmp/result", "/tmp/db")
    context.setup_project.assert_called_once_with("/tmp/result")
    persistence.update_project.assert_called_once_with(
        chip_name="chip",
        qub_name="qubit",
        res_name="res",
        result_dir="/tmp/result",
        database_path="/tmp/db",
    )


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


def test_remember_device_rejects_non_persisted_command() -> None:
    svc, _, _, _ = _make_service()

    with pytest.raises(ValueError, match="remember=False"):
        svc.remember_device(
            ConnectDeviceRequest(
                type_name="FakeDevice",
                name="flux",
                address="addr",
                remember=False,
            )
        )


def test_connection_request_validates_port() -> None:
    with pytest.raises(ValueError, match="port"):
        StartupConnectionRequest(ip="host", port=0)
