"""Tests for DeviceService registry snapshot API (list_device_names / get_device_unit)."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.device.fake import FakeDevice
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.device import DeviceService, RegisterDeviceRequest
from zcu_tools.gui.state import ExpContext, State


def _make_svc(driver: object | None = None) -> DeviceService:
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    fake = driver if driver is not None else FakeDevice()

    def factory(type_name: str, address: str) -> object:
        return fake

    return DeviceService(state, EventBus(), driver_factory=factory)  # type: ignore[arg-type]


def test_list_device_names_returns_sorted_keys():
    svc = _make_svc()
    svc.register_device(
        RegisterDeviceRequest(type_name="FakeDevice", name="z", address="")
    )
    svc.register_device(
        RegisterDeviceRequest(type_name="FakeDevice", name="a", address="")
    )
    names = svc.list_device_names()
    # Both names should be present and sorted.
    assert "a" in names
    assert "z" in names
    assert names == sorted(names)
    svc.drop_device("z")
    svc.drop_device("a")


def test_get_device_unit_unknown_returns_none():
    svc = _make_svc()
    assert svc.get_device_unit("missing") == "none"


def test_get_device_unit_for_fake_device_returns_none():
    svc = _make_svc()
    svc.register_device(
        RegisterDeviceRequest(type_name="FakeDevice", name="flux", address="")
    )
    assert svc.get_device_unit("flux") == "none"
    svc.drop_device("flux")


def test_get_device_value_for_new_context_unknown_returns_none():
    svc = _make_svc()
    assert svc.get_device_value_for_new_context("missing") is None


def test_get_device_value_for_new_context_returns_float():
    fake = FakeDevice()
    fake.set_value(2.5)
    svc = _make_svc(driver=fake)
    svc.register_device(
        RegisterDeviceRequest(type_name="FakeDevice", name="flux", address="")
    )
    assert svc.get_device_value_for_new_context("flux") == 2.5
    svc.drop_device("flux")
