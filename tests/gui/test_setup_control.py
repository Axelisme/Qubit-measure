"""SetupControlFacet delegation contract."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.gui.session.services.connection import ConnectMockRequest
from zcu_tools.gui.session.services.startup import (
    StartupConnectionRequest,
    StartupProjectRequest,
)
from zcu_tools.gui.session.setup_control import SetupControlFacet


def _facet() -> tuple[SetupControlFacet, MagicMock, MagicMock, MagicMock, MagicMock]:
    bus = MagicMock()
    startup = MagicMock()
    context = MagicMock()
    connection = MagicMock()
    device = MagicMock()
    return (
        SetupControlFacet(
            bus=cast(Any, bus),
            startup=cast(Any, startup),
            context=cast(Any, context),
            connection=cast(Any, connection),
            device=cast(Any, device),
        ),
        startup,
        context,
        connection,
        device,
    )


def test_setup_control_delegates_startup_context_connection_and_device_calls() -> None:
    facet, startup, context, connection, device = _facet()
    startup.get_persisted.return_value = "prefs"
    startup.list_result_scopes.return_value = ("scope",)
    context.get_context_labels.return_value = ["base"]
    context.get_active_context_label.return_value = "base"
    connection.start_connect.return_value = 7
    connection.get_soccfg.return_value = "cfg"
    device.list_devices.return_value = ["dev"]
    device.get_device_unit.return_value = "V"

    req = StartupProjectRequest("chip", "qub", "res")
    conn_req = StartupConnectionRequest(ip="127.0.0.1", port=8887)
    connect_req = ConnectMockRequest()

    assert facet.get_persisted_startup() == "prefs"
    assert facet.list_result_scopes() == ("scope",)
    assert facet.apply_startup_project(req) is True
    facet.use_context("base")
    facet.new_context(bind_device="flux", clone_from="base")
    assert facet.get_context_labels() == ["base"]
    assert facet.get_active_context_label() == "base"
    assert facet.start_connect(connect_req) == 7
    facet.remember_startup_connection(conn_req)
    assert facet.get_soccfg() == "cfg"
    assert facet.list_devices() == ["dev"]
    assert facet.get_device_unit("flux") == "V"

    startup.get_persisted.assert_called_once_with()
    startup.list_result_scopes.assert_called_once_with()
    startup.apply_project.assert_called_once_with(req)
    context.use_context.assert_called_once_with("base")
    context.new_context.assert_called_once_with(
        bind_device="flux",
        clone_from="base",
    )
    connection.start_connect.assert_called_once_with(connect_req)
    startup.remember_connection.assert_called_once_with(conn_req)
    connection.get_soccfg.assert_called_once_with()
    device.list_devices.assert_called_once_with()
    device.get_device_unit.assert_called_once_with("flux")


def test_setup_control_rebinds_connection_outcome_signals() -> None:
    facet, _startup, _context, connection, _device = _facet()
    on_finished = MagicMock()
    on_failed = MagicMock()

    facet.bind_connection_outcome(on_finished, on_failed)

    connection.connection_finished.disconnect.assert_called_once_with()
    connection.connection_failed.disconnect.assert_called_once_with()
    connection.connection_finished.connect.assert_called_once_with(on_finished)
    connection.connection_failed.connect.assert_called_once_with(on_failed)
