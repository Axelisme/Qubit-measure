"""Fixed memory-device coordinates survive setup and selected flux writes."""

from typing import Literal

import pytest
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.utils.device import check_flux_unit, set_flux


@pytest.mark.parametrize("unit", ["V", "A"])
def test_fixed_fake_coordinate_roundtrip_and_flux(unit: Literal["V", "A"]) -> None:
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
        check_flux_unit(info, unit)
        set_flux(device, -0.08, unit=unit, stop=StopSignal())
        assert device.get_value() == -0.08
        assert device.get_info().value_unit == unit
    finally:
        device.close()


def test_setup_cannot_change_fake_coordinate_or_write_before_rejection() -> None:
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
        with pytest.raises(ValueError, match="unit"):
            set_flux(device, 1.0, unit="native", stop=StopSignal())
        assert device.get_info() == before
    finally:
        device.close()
