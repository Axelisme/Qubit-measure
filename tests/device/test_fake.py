"""Address identity, fixed units and bounded memory-source retention."""

from typing import Literal

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo


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


@pytest.mark.parametrize("unit", ["V", "A"])
def test_fixed_coordinate_survives_setup_and_readback(unit: Literal["V", "A"]) -> None:
    device = FakeDevice(fast_mode=True, value_unit=unit)
    try:
        device.setup(
            FakeDeviceInfo(
                address="none", value_unit=unit, value=0.12, rampstep=0.04, output="on"
            ),
            progress=False,
        )
        info = device.get_info()
        assert info.value_unit == unit
        assert info.value == pytest.approx(0.12)
        assert info.rampstep == 0.04
        device.set_value(-0.08)
        assert device.get_value() == -0.08
        assert device.get_info().value_unit == unit
    finally:
        device.close()


def test_setup_rejects_coordinate_change_before_writing() -> None:
    device = FakeDevice(fast_mode=True, value_unit="V")
    try:
        device.set_value(0.25)
        before = device.get_info()
        with pytest.raises(ValueError, match="unit"):
            device.setup(
                FakeDeviceInfo(address="none", value_unit="A", value=0.9, output="on"),
                progress=False,
            )
        assert device.get_info() == before
    finally:
        device.close()


def test_full_instance_container_rejects_new_addresses_but_preserves_existing():
    first = FakeDevice(address="capacity-0", fast_mode=True)
    first.set_value(0.25)
    for index in range(1, 1024):
        FakeDevice(address=f"capacity-{index}")
    with pytest.raises(RuntimeError, match="(?i)capacity.*1024"):
        FakeDevice(address="over-capacity")
    assert FakeDevice(address="capacity-0") is first
    assert first.get_value() == 0.25
