"""Native driver effects retain validation and cooperative cancellation."""

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.workflows.runtime import SessionDevices
from zcu_tools.gui.session.services.device import ConnectDeviceRequest

from ._start_fakes import StartHarness


def connected() -> StartHarness:
    harness = StartHarness()
    harness.services.device.start_connect_device(
        ConnectDeviceRequest("FakeDevice", "flux", "none")
    )
    harness.background.execute()
    harness.background.deliver()
    driver = harness.services.device.get_connected_devices()["flux"]
    driver.setup(FakeDeviceInfo(address="none", rampstep=1.0), progress=False)
    return harness


def test_effect_changes_registered_driver_but_not_startup_cache() -> None:
    harness = connected()
    service = harness.services.device
    port = SessionDevices(service)
    port.set_value("flux", 0.12, StopSignal())
    driver = service.get_connected_devices()["flux"]
    assert driver.get_info().model_dump()["value"] == 0.12
    assert service.capture_native_device_info()["flux"].model_dump()["value"] == 0.0
    port.setup(
        {"flux": FakeDeviceInfo(address="none", value=0.2, rampstep=1.0)}, StopSignal()
    )
    assert driver.get_info().model_dump()["value"] == 0.2


def test_cancelled_native_setup_does_not_apply_value() -> None:
    harness = connected()
    port = SessionDevices(harness.services.device)
    signal = StopSignal()
    signal.set()
    port.set_value("flux", 0.5, signal)
    driver = harness.services.device.get_connected_devices()["flux"]
    assert driver.get_info().model_dump()["value"] == 0.0


def test_native_name_and_address_errors_propagate_without_partial_batch_setup() -> None:
    harness = connected()
    port = SessionDevices(harness.services.device)
    with pytest.raises(ValueError, match="not found"):
        port.setup(
            {
                "flux": FakeDeviceInfo(address="none", value=0.5),
                "unknown": FakeDeviceInfo(address="none"),
            },
            StopSignal(),
        )
    driver = harness.services.device.get_connected_devices()["flux"]
    assert driver.get_info().model_dump()["value"] == 0.0
    with pytest.raises(RuntimeError, match="address"):
        port.setup({"flux": FakeDeviceInfo(address="wrong", value=0.5)}, StopSignal())
    assert driver.get_info().model_dump()["value"] == 0.0
