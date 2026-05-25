"""Tests for DeviceService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.device import DeviceService
from zcu_tools.gui.state import ExpContext, State


def _make_svc(running: bool = False) -> DeviceService:
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    if running:
        state.running_tab_id = "some_tab"
    return DeviceService(state, EventBus())


def test_device_service_success():
    svc = _make_svc()
    device = MagicMock()

    with patch("zcu_tools.device.GlobalDeviceManager.register_device") as mock_reg:
        svc.register_device("dev1", device)
        mock_reg.assert_called_with("dev1", device)

    with patch("zcu_tools.device.GlobalDeviceManager.drop_device") as mock_drop:
        svc.drop_device("dev1")
        mock_drop.assert_called_with("dev1")

    with patch("zcu_tools.device.GlobalDeviceManager.get_all_devices", return_value={}):
        result = svc.list_devices()
        assert isinstance(result, dict)

    with patch("zcu_tools.device.GlobalDeviceManager.get_device") as mock_get:
        mock_get.return_value = device
        svc.set_device_value("dev1", 1.0)
        device.set_value.assert_called_with(1.0)

        svc.get_device_value("dev1")
        device.get_value.assert_called_once()

    with patch("zcu_tools.device.GlobalDeviceManager.get_device", return_value=device):
        svc.get_device_info("dev1")
        device.get_info.assert_called()


def test_device_service_blocks_when_running():
    svc = _make_svc(running=True)
    device = MagicMock()

    with pytest.raises(RuntimeError, match="Cannot register device"):
        svc.register_device("dev1", device)

    with pytest.raises(RuntimeError, match="Cannot drop device"):
        svc.drop_device("dev1")

    with pytest.raises(RuntimeError, match="Cannot set device value"):
        svc.set_device_value("dev1", 1.0)

    with pytest.raises(RuntimeError, match="Cannot setup device"):
        svc.setup_device("dev1", {})

    # Non-mutating methods should still work when running
    with patch("zcu_tools.device.GlobalDeviceManager.get_all_devices", return_value={}):
        svc.list_devices()

    with patch("zcu_tools.device.GlobalDeviceManager.get_device", return_value=device):
        svc.get_device_info("dev1")
        device.get_info.assert_called_once()
