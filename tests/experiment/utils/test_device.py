"""Selected-device operations preserve units, cancellation and driver errors."""

from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.device import YOKOGS200, FakeDevice, RohdeSchwarzSGS100A
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.mg3692 import AnritsuMG3692Info
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.experiment import utils as public_helpers
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.utils import device as helpers

from tests.experiment.utils._device_transports import DeviceTransport


# New operations are accessed at runtime so missing helpers give behavioral red,
# rather than preventing collection on the test-only commit.
def _instrument(mode: str = "CURR") -> tuple[YOKOGS200, DeviceTransport]:
    transport = DeviceTransport(mode)
    manager = MagicMock()
    manager.open_resource.return_value = transport
    return YOKOGS200("SIMULATED::INSTR", manager), transport


def test_flux_fake_native_changes_only_selected_value() -> None:
    selected, other = FakeDevice(), FakeDevice()
    before = selected.get_info()
    helpers.set_flux(selected, 0.25, unit="native", stop=StopSignal())
    assert selected.get_info() == before.with_updates(value=0.25)
    assert other.get_value() == 0.0


@pytest.mark.parametrize("unit,mode,value", [("A", "CURR", 2e-6), ("V", "VOLT", 2e-3)])
def test_flux_yoko_uses_explicit_units_and_preserves_configuration(
    unit: Literal["A", "V"], mode: str, value: float
) -> None:
    device, transport = _instrument(mode)
    before = device.get_info()
    helpers.set_flux(device, value, unit=unit, stop=StopSignal())
    assert device.get_info() == before.with_updates(value=value)
    assert transport.writes
    assert all(
        command.startswith(":SOURce:LEVel:AUTO ") for command in transport.writes
    )


@pytest.mark.parametrize(
    "unit,mode", [("A", "VOLT"), ("V", "CURR"), ("native", "CURR")]
)
def test_flux_rejects_unit_or_mode_before_writing(
    unit: Literal["A", "V", "native"], mode: str
) -> None:
    device, transport = _instrument(mode)
    before = device.get_info()
    with pytest.raises(ValueError, match="."):
        helpers.set_flux(device, 1e-6, unit=unit, stop=StopSignal())
    assert transport.writes == []
    assert device.get_info() == before
    fake = FakeDevice()
    with pytest.raises(ValueError, match="."):
        helpers.set_flux(fake, 1.0, unit="A", stop=StopSignal())
    assert fake.get_value() == 0.0


def test_flux_yoko_honors_run_stop_during_ramp() -> None:
    device, transport = _instrument()
    stop = StopSignal()
    transport.stop = stop.event
    helpers.set_flux(device, 4e-6, unit="A", stop=stop, progress=False)
    assert stop.is_set()
    assert device.get_current() < 4e-6
    assert len(transport.writes) == 1
    transport.writes.clear()
    helpers.set_flux(device, 4e-6, unit="A", stop=stop)
    assert transport.writes == []


def test_rf_helpers_use_hz_dbm_and_single_fake_value() -> None:
    transport = DeviceTransport()
    manager = MagicMock()
    manager.open_resource.return_value = transport
    rf = RohdeSchwarzSGS100A("SIMULATED::INSTR", manager)
    fake = FakeDevice()
    for device in (rf, fake):
        helpers.set_rf_frequency(device, 6e9)
        if isinstance(device, FakeDevice):
            assert device.get_info().value == 6e9
        else:
            assert device.get_frequency() == 6e9
        helpers.set_rf_power(device, -30.0)
        if isinstance(device, FakeDevice):
            assert device.get_info().value == -30.0
        else:
            assert device.get_power() == -30.0
    assert transport.frequency == 6e9
    assert transport.power == -30.0


def test_helpers_preserve_driver_error() -> None:
    device, transport = _instrument()
    error = OSError("transport unavailable")
    transport.failure = error
    with pytest.raises(OSError, match="transport unavailable") as caught:
        helpers.set_flux(device, 1e-6, unit="A", stop=StopSignal())
    assert caught.value is error
    with pytest.raises(RuntimeError, match="Current"):
        helpers.set_flux(device, 0.021, unit="A", stop=StopSignal())
    assert transport.writes == []


@pytest.mark.parametrize(
    "operation", ["set_flux", "set_rf_frequency", "set_rf_power", "set_output"]
)
def test_helpers_reject_unsupported_device(operation: str) -> None:
    helper = getattr(helpers, operation)
    if operation == "set_flux":
        args = (object(), 1.0)
        kwargs = {"unit": "native", "stop": StopSignal()}
    elif operation == "set_output":
        args = (object(), "on")
        kwargs = {}
    else:
        args = (object(), 1.0)
        kwargs = {}
    with pytest.raises(TypeError):
        helper(*args, **kwargs)


def test_output_helper_changes_only_output() -> None:
    device = FakeDevice()
    device.set_value(1.25)
    before = device.get_info()
    helpers.set_output(device, "on")
    assert device.get_info() == before.with_updates(output="on")
    helpers.set_output(device, "off")
    assert device.get_info() == before


def _flux_configuration(
    model: Literal["yoko", "fake"], mode: Literal["current", "voltage"]
) -> BaseDeviceInfo:
    """Build an Info input without opening or querying a driver."""
    if model == "fake":
        return FakeDeviceInfo(address="unused", value=0.001)
    return YOKOGS200Info(address="unused", mode=mode, value=0.001)


@pytest.mark.parametrize(
    "model, mode, unit",
    [
        pytest.param("yoko", "current", "A", id="yoko-current-A"),
        pytest.param("yoko", "voltage", "V", id="yoko-voltage-V"),
        pytest.param("fake", "current", "native", id="fake-native"),
    ],
)
def test_check_flux_unit_accepts_matching_configuration(
    model: Literal["yoko", "fake"], mode: Literal["current", "voltage"], unit: str
) -> None:
    cfg = _flux_configuration(model, mode)
    before = cfg.model_copy(deep=True)
    assert public_helpers.check_flux_unit(cfg, unit) is None
    assert cfg == before


@pytest.mark.parametrize(
    "model, mode, unit",
    [
        pytest.param("yoko", "current", "V", id="yoko-current-V"),
        pytest.param("yoko", "current", "native", id="yoko-current-native"),
        pytest.param("yoko", "voltage", "A", id="yoko-voltage-A"),
        pytest.param("yoko", "voltage", "native", id="yoko-voltage-native"),
        pytest.param("fake", "current", "A", id="fake-A"),
        pytest.param("fake", "current", "V", id="fake-V"),
    ],
)
def test_check_flux_unit_rejects_mismatched_configuration(
    model: Literal["yoko", "fake"], mode: Literal["current", "voltage"], unit: str
) -> None:
    cfg = _flux_configuration(model, mode)
    before = cfg.model_copy(deep=True)
    with pytest.raises(ValueError, match="unit"):
        public_helpers.check_flux_unit(cfg, unit)
    assert cfg == before


@pytest.mark.parametrize(
    "model, mode",
    [
        pytest.param("yoko", "current", id="yoko-current"),
        pytest.param("yoko", "voltage", id="yoko-voltage"),
        pytest.param("fake", "current", id="fake"),
    ],
)
def test_check_flux_unit_rejects_unknown_unit(
    model: Literal["yoko", "fake"], mode: Literal["current", "voltage"]
) -> None:
    cfg = _flux_configuration(model, mode)
    before = cfg.model_copy(deep=True)
    with pytest.raises(ValueError, match="unit"):
        public_helpers.check_flux_unit(cfg, "mA")
    assert cfg == before


@pytest.mark.parametrize("model", ["sgs100a", "mg3692", "base-info"])
def test_check_flux_unit_rejects_non_flux_configuration(model: str) -> None:
    cfg: BaseDeviceInfo
    if model == "sgs100a":
        cfg = RohdeSchwarzSGS100AInfo(address="unused")
    elif model == "mg3692":
        cfg = AnritsuMG3692Info(address="unused")
    else:
        cfg = BaseDeviceInfo(address="unused", type="OtherDevice")
    before = cfg.model_copy(deep=True)
    with pytest.raises(TypeError) as error:
        public_helpers.check_flux_unit(cfg, "A")
    assert type(cfg).__name__ in str(error.value)
    assert "flux" in str(error.value).lower()
    assert "support" in str(error.value).lower()
    assert cfg == before
