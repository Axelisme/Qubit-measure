"""Address identity and bounded process-lifetime memory-source retention."""

import pytest
from zcu_tools.device import FakeDevice


def test_explicit_address_reuses_state_but_anonymous_devices_are_independent():
    device = FakeDevice(address="retained", fast_mode=True, value_unit="A")
    device.output_on()
    device.set_value(0.25)
    device.close()
    reused = FakeDevice(address="retained")
    assert reused is device
    assert reused.get_info().value_unit == "A"
    assert reused.get_info().output == "on"
    assert reused.get_value() == 0.25
    assert FakeDevice(address="independent") is not device
    assert FakeDevice() is not FakeDevice()


def test_full_instance_container_rejects_new_addresses_but_preserves_existing():
    first = FakeDevice(address="capacity-0", fast_mode=True)
    first.set_value(0.25)
    for index in range(1, 1024):
        FakeDevice(address=f"capacity-{index}")
    with pytest.raises(RuntimeError, match="(?i)capacity.*1024"):
        FakeDevice(address="over-capacity")
    assert FakeDevice(address="capacity-0") is first
    assert first.get_value() == 0.25
