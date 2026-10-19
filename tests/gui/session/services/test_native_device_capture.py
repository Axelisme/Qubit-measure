"""Cached native device info is captured by the session owner without I/O."""

import pytest
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.gui.session.state import DeviceState, DeviceStatus

from tests.gui.app.workflows._start_fakes import StartHarness


def test_capture_uses_cached_native_models_not_live_drivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = StartHarness()
    service = harness.services.device
    service.register_device("flux", FakeDevice(fast_mode=True))
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


def test_live_capture_reads_all_connected_drivers_instead_of_cached_values() -> None:
    harness = StartHarness()
    service = harness.services.device
    for name, value in (("flux", 0.12), ("readout", 0.34)):
        service.register_device(name, FakeDevice(fast_mode=True))
        driver = service.get_connected_devices()[name]
        assert isinstance(driver, FakeDevice)
        driver.value = value
        harness.state.set_device_info(name, FakeDeviceInfo(address=name, value=-1.0))
    harness.state.put_device(
        DeviceState(
            name="remembered",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.MEMORY_ONLY,
            remember=True,
        )
    )

    captured = service.capture_live_native_device_info()

    assert tuple(captured) == ("flux", "readout")
    assert captured["flux"].model_dump()["value"] == 0.12
    assert captured["readout"].model_dump()["value"] == 0.34
    snapshot = service.get_device_snapshot("flux")
    assert snapshot is not None and snapshot.info == captured["flux"]
    assert harness.handles.live_count() == 0
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("error", [ValueError("bad response"), OSError("read failed")])
def test_live_capture_names_failed_device_without_cached_fallback(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    harness = StartHarness()
    service = harness.services.device
    service.register_device("flux", FakeDevice(fast_mode=True))
    harness.state.set_device_info("flux", FakeDeviceInfo(address="none", value=0.12))
    driver = service.get_connected_devices()["flux"]

    def reject_read() -> FakeDeviceInfo:
        raise error

    monkeypatch.setattr(driver, "get_info", reject_read)

    with pytest.raises(RuntimeError, match="flux"):
        service.capture_live_native_device_info()

    assert service.get_cached_device_value("flux") == 0.12
    assert harness.handles.live_count() == 0
    assert not harness.gate.snapshot()


def test_live_capture_rejects_non_native_info_with_device_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = StartHarness()
    service = harness.services.device
    service.register_device("flux", FakeDevice(fast_mode=True))
    driver = service.get_connected_devices()["flux"]

    def unsupported_read() -> BaseDeviceInfo:
        return BaseDeviceInfo(address="none", type="unsupported")

    monkeypatch.setattr(driver, "get_info", unsupported_read)

    with pytest.raises(RuntimeError, match="flux"):
        service.capture_live_native_device_info()


def test_live_capture_without_connected_devices_returns_empty_snapshot() -> None:
    assert StartHarness().services.device.capture_live_native_device_info() == {}
