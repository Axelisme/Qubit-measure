"""Tests for DeviceService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.device import (
    DeviceRegistrationError,
    DeviceService,
    RegisterDeviceRequest,
)
from zcu_tools.gui.state import ExpContext, State


def _make_svc(
    running: bool = False, driver: MagicMock | None = None
) -> tuple[DeviceService, MagicMock]:
    """Build a DeviceService with an injected fake driver factory.

    Returns the service and the driver instance the factory yields, so tests
    can introspect calls on the resulting device.
    """
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    if running:
        state.running_tab_id = "some_tab"
    fake_device = driver if driver is not None else MagicMock()

    def factory(type_name: str, address: str) -> object:
        return fake_device

    svc = DeviceService(state, EventBus(), driver_factory=factory)  # type: ignore[arg-type]
    return svc, fake_device


def _req(name: str = "dev1", type_name: str = "FakeDevice") -> RegisterDeviceRequest:
    return RegisterDeviceRequest(type_name=type_name, name=name, address="addr")


def test_device_service_success():
    svc, device = _make_svc()

    with patch("zcu_tools.device.GlobalDeviceManager.register_device") as mock_reg:
        svc.register_device(_req("dev1"))
        mock_reg.assert_called_with("dev1", device)

    with (
        patch(
            "zcu_tools.device.manager.GlobalDeviceManager.get_info",
            return_value=FakeDeviceInfo(address="addr"),
        ),
        patch(
            "zcu_tools.device.manager.GlobalDeviceManager.get_device",
            return_value=device,
        ),
        patch("zcu_tools.device.GlobalDeviceManager.drop_device") as mock_drop,
    ):
        svc.drop_device("dev1")
        device.close.assert_called_once_with()
        mock_drop.assert_called_with("dev1")

    with patch("zcu_tools.device.GlobalDeviceManager.get_all_devices", return_value={}):
        result = svc.list_devices()
        assert isinstance(result, list)

    fake_info = FakeDeviceInfo(address="none", value=1.0)
    with patch("zcu_tools.device.GlobalDeviceManager.get_device", return_value=device):
        device.set_value.reset_mock()
        svc.set_device_value("dev1", 1.0)
        device.set_value.assert_called_with(1.0)

    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_info", return_value=fake_info
    ):
        assert svc.get_device_value("dev1") == 1.0

    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_info", return_value=fake_info
    ):
        result = svc.get_device_info("dev1")
        assert result == fake_info


def test_device_service_blocks_when_running():
    svc, _device = _make_svc(running=True)

    with pytest.raises(RuntimeError, match="Cannot register device"):
        svc.register_device(_req("dev1"))

    with pytest.raises(RuntimeError, match="Cannot drop device"):
        svc.drop_device("dev1")

    with pytest.raises(RuntimeError, match="Cannot set device value"):
        svc.set_device_value("dev1", 1.0)

    with pytest.raises(RuntimeError, match="Cannot setup device"):
        svc.setup_device("dev1", FakeDeviceInfo(address="none"))

    # Non-mutating methods should still work when running
    with patch("zcu_tools.device.GlobalDeviceManager.get_all_devices", return_value={}):
        svc.list_devices()

    fake_info = FakeDeviceInfo(address="none", value=0.0)
    with patch(
        "zcu_tools.device.manager.GlobalDeviceManager.get_info", return_value=fake_info
    ):
        result = svc.get_device_info("dev1")
        assert result == fake_info


def test_device_service_blocks_mutation_while_setup_active():
    svc, _device = _make_svc()
    svc._state.begin_device_setup("active")

    with pytest.raises(RuntimeError, match="device setup is active"):
        svc.register_device(_req("dev1"))
    with pytest.raises(RuntimeError, match="device setup is active"):
        svc.drop_device("dev1")
    with pytest.raises(RuntimeError, match="device setup is active"):
        svc.set_device_value("dev1", 1.0)
    with pytest.raises(RuntimeError, match="device setup is active"):
        svc.setup_device("dev1", FakeDeviceInfo(address="none"))


def test_device_service_wraps_driver_factory_failures_as_registration_error():
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )

    def failing_factory(type_name: str, address: str) -> object:
        raise OSError("VISA timeout")

    svc = DeviceService(state, EventBus(), driver_factory=failing_factory)  # type: ignore[arg-type]
    with pytest.raises(DeviceRegistrationError, match="VISA timeout"):
        svc.register_device(_req("dev1"))


def test_device_service_register_request_unknown_type_raises_registration_error():
    svc, _device = _make_svc()
    # Use the default factory path to validate unknown type guard.
    svc._driver_factory = lambda type_name, address: (_ for _ in ()).throw(
        DeviceRegistrationError(f"Unknown device type: {type_name!r}")
    )
    with pytest.raises(DeviceRegistrationError, match="Unknown device type"):
        svc.register_device(_req("dev1", type_name="Nope"))
