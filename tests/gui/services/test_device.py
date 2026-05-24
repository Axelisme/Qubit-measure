"""Tests for DeviceService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.services.device import DeviceService
from zcu_tools.gui.state import ExpContext, State


def test_device_service_success():
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    dm = MagicMock()
    svc = DeviceService(state, dm)
    device = MagicMock()

    svc.register_device("dev1", device)
    dm.register_device.assert_called_with("dev1", device)

    svc.drop_device("dev1")
    dm.drop_device.assert_called_with("dev1")

    svc.list_devices()
    dm.list_devices.assert_called_once()

    svc.set_device_value("dev1", {"value": 1.0})
    dm.set_device_value.assert_called_with("dev1", {"value": 1.0})

    svc.get_device_value("dev1")
    dm.get_device_value.assert_called_with("dev1")

    svc.get_device_info("dev1")
    dm.get_device_info.assert_called_with("dev1")

    svc.setup_device("dev1", {"addr": "abc"})
    dm.setup_device.assert_called_with("dev1", {"addr": "abc"}, None)


def test_device_service_blocks_when_running():
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    state.running_tab_id = "some_tab"

    dm = MagicMock()
    svc = DeviceService(state, dm)
    device = MagicMock()

    with pytest.raises(RuntimeError, match="Cannot register device"):
        svc.register_device("dev1", device)

    with pytest.raises(RuntimeError, match="Cannot drop device"):
        svc.drop_device("dev1")

    with pytest.raises(RuntimeError, match="Cannot set device value"):
        svc.set_device_value("dev1", 1.0)

    with pytest.raises(RuntimeError, match="Cannot setup device"):
        svc.setup_device("dev1", {})

    # Non-mutating methods should still work
    svc.list_devices()
    dm.list_devices.assert_called_once()

    svc.get_device_info("dev1")
    dm.get_device_info.assert_called_once()
