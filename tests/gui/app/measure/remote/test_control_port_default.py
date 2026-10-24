"""Main RemoteControlAdapter startup rollback on endpoint failures."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.gui.app.measure.remote import RemoteControlAdapter
from zcu_tools.gui.remote.rpc_endpoint import ControlOptions, NdjsonRpcEndpoint
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler


def test_remote_control_adapter_start_rolls_back_bind_error(qapp) -> None:
    """Bind failure propagates and releases the diagnostic listener."""
    ctrl = MagicMock()
    adapter = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
        render_view=MagicMock(),
    )
    with (
        patch.object(NdjsonRpcEndpoint, "start", side_effect=RuntimeError("bind")),
        patch.object(NdjsonRpcEndpoint, "stop") as endpoint_stop,
        pytest.raises(RuntimeError, match="bind"),
    ):
        adapter.start()
    endpoint_stop.assert_not_called()
    ctrl.add_diagnostic_sink.assert_called_once_with(adapter)
    ctrl.remove_diagnostic_sink.assert_called_once_with(adapter)


def test_remote_control_adapter_start_rolls_back_advertise_error(qapp) -> None:
    """Discovery failure stops the opened endpoint and diagnostic listener."""
    ctrl = MagicMock()
    adapter = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0, app_slug="measure"),
        owner_scheduler=QtOwnerScheduler(),
        render_view=MagicMock(),
    )
    with (
        patch.object(NdjsonRpcEndpoint, "start", return_value=12345),
        patch.object(NdjsonRpcEndpoint, "stop") as endpoint_stop,
        patch(
            "zcu_tools.gui.remote.control_service.write_session",
            side_effect=RuntimeError("discovery write failed"),
        ),
        pytest.raises(RuntimeError, match="discovery write failed"),
    ):
        adapter.start()
    endpoint_stop.assert_called_once_with()
    ctrl.add_diagnostic_sink.assert_called_once_with(adapter)
    ctrl.remove_diagnostic_sink.assert_called_once_with(adapter)
