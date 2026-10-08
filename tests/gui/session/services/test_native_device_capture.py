"""Cached native device info is captured by the session owner without I/O."""

import pytest
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.gui.session.state import DeviceState, DeviceStatus

from tests.gui.app.workflows._start_fakes import StartHarness


def test_capture_uses_cached_native_models_not_live_drivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = StartHarness()
    service = harness.services.device
    service.start_connect_device(ConnectDeviceRequest("FakeDevice", "flux", "none"))
    harness.background.execute()
    harness.background.deliver()
    assert harness.state.get_device("flux") is not None
    harness.state.set_device_info("flux", FakeDeviceInfo(address="none", value=0.12))
    driver = service.get_connected_devices()["flux"]

    def forbid_read() -> FakeDeviceInfo:
        raise AssertionError("cached capture read the driver")

    monkeypatch.setattr(driver, "get_info", forbid_read)
    captured = service.capture_native_device_info()
    assert tuple(captured) == ("flux",)
    assert isinstance(captured["flux"], FakeDeviceInfo)
    assert captured["flux"].model_dump()["value"] == 0.12
    assert harness.handles.live_count() == 0
    assert not harness.gate.snapshot()


def test_remembered_devices_are_not_startup_provenance() -> None:
    harness = StartHarness()
    harness.state.put_device(
        DeviceState(
            name="remembered",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.MEMORY_ONLY,
            remember=True,
        )
    )
    assert harness.services.device.capture_native_device_info() == {}


@pytest.mark.parametrize(
    "info",
    [
        None,
        BaseDeviceInfo(address="none", type="unsupported"),
    ],
)
def test_live_device_without_native_cached_info_fails_instead_of_disappearing(
    info: BaseDeviceInfo | None,
) -> None:
    harness = StartHarness()
    harness.state.put_device(
        DeviceState(
            name="flux",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.CONNECTED,
            remember=False,
            info=info,
        )
    )
    with pytest.raises(ValueError, match=".+"):
        harness.services.device.capture_native_device_info()
