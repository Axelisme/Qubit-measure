"""Explicit device-setup contract shared by experiment callers."""

from typing import Literal
from unittest.mock import MagicMock

import pytest
from zcu_tools.device import YOKOGS200, FakeDevice, FakeDeviceInfo, RohdeSchwarzSGS100A
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.utils import device as helpers
from zcu_tools.experiment.utils.device import setup_devices

from tests.experiment.utils._device_transports import DeviceTransport


def _cfg(*names: str) -> ExpCfgModel:
    return ExpCfgModel(
        dev={
            name: FakeDeviceInfo(address="none", output="on", value=0.5)
            for name in names
        }
    )


def test_setup_resolves_entire_mapping_before_any_device_is_modified():
    first = FakeDevice(fast_mode=True)
    stop = StopSignal()
    stop.set()
    with pytest.raises(ValueError, match="missing"):
        setup_devices(_cfg("first", "missing"), {"first": first}, cancel_signal=stop)
    assert first.get_output() == "off"
    assert first.get_value() == 0.0


def test_setup_uses_only_supplied_devices_and_honors_explicit_cancel():
    first = FakeDevice(fast_mode=True)
    unrelated = FakeDevice(fast_mode=True)
    stop = StopSignal()
    stop.set()
    setup_devices(
        _cfg("first"), {"first": first, "other": unrelated}, cancel_signal=stop
    )
    assert first.get_output() == "off"

    stop.clear_stop()
    setup_devices(
        _cfg("first"), {"first": first, "other": unrelated}, cancel_signal=stop
    )
    assert first.get_value() == pytest.approx(0.5)
    assert unrelated.get_value() == 0.0
    assert unrelated.get_output() == "off"


def test_setup_propagates_cancel_to_driver_and_stops_before_next_device(monkeypatch):
    first = FakeDevice(fast_mode=True)
    second = FakeDevice(fast_mode=True)
    stop = StopSignal()
    original = first.setup

    def setup_and_stop(info, *, progress, stop_event):
        assert stop_event is stop.event
        original(info, progress=progress, stop_event=stop_event)
        stop.set()

    monkeypatch.setattr(first, "setup", setup_and_stop)
    setup_devices(
        _cfg("first", "second"), {"first": first, "second": second}, cancel_signal=stop
    )
    assert first.get_value() == pytest.approx(0.5)
    assert second.get_output() == "off"


def test_setup_without_device_cfg_does_not_require_a_registry():
    setup_devices(ExpCfgModel(), {})


def test_setup_without_a_signal_configures_requested_devices():
    first = FakeDevice(fast_mode=True)
    second = FakeDevice(fast_mode=True)
    setup_devices(_cfg("first", "second"), {"first": first, "second": second})
    assert first.get_value() == pytest.approx(0.5)
    assert second.get_value() == pytest.approx(0.5)
    assert first.get_output() == second.get_output() == "on"


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
