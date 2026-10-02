"""Explicit device-setup contract shared by experiment callers."""

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.utils.device import setup_devices


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
