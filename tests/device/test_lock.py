"""Public device concurrency, ramp and operation logging contracts."""

from __future__ import annotations

import logging
import threading

import pytest
import zcu_tools.device.fake as fake_module
from _pytest.logging import LogCaptureFixture
from zcu_tools.device import DeviceBusyError, FakeDevice, FakeDeviceInfo

from ._threads import DeviceThreads


def test_fake_device_rampstep_is_actual_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FakeDevice rampstep matches YOKO semantics: it is the actual ramp step."""
    sleeps: list[float] = []
    monkeypatch.setattr(
        fake_module.time,
        "sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )
    dev = FakeDevice(fast_mode=False)
    cfg = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.25)

    dev.setup(cfg, progress=False)

    assert dev.get_value() == 1.0
    assert len(sleeps) == 4


def test_setup_ramp_allows_concurrent_get_info(
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev = FakeDevice(fast_mode=False)
    cfg = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.25)
    entered, release = paused_ramp
    worker = workers.thread(
        release=release, target=lambda: dev.setup(cfg, progress=False)
    )

    worker.start()
    try:
        workers.arrived(entered)
        assert 0.0 < dev.get_info().value < cfg.value
    finally:
        release.set()
        workers.join(worker)

    assert dev.get_value() == cfg.value


def test_concurrent_setup_raises_device_busy_error(
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev = FakeDevice(fast_mode=False)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.25)
    entered, release = paused_ramp
    worker = workers.thread(
        release=release, target=lambda: dev.setup(cfg, progress=False)
    )

    worker.start()
    try:
        workers.arrived(entered)
        with pytest.raises(DeviceBusyError):
            dev.setup(cfg.with_updates(value=-0.5), progress=False)
    finally:
        release.set()
        workers.join(worker)

    assert dev.get_value() == cfg.value


def test_nested_calls_do_not_deadlock() -> None:
    """Public setup must allow its nested output setter on the same thread."""
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.25, rampstep=0.01)

    dev.setup(cfg, progress=False)

    info = dev.get_info()
    assert info.value == 0.25
    assert info.output == "on"


def test_is_busy_true_during_ramp(
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev = FakeDevice(fast_mode=False)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.25)
    entered, release = paused_ramp
    worker = workers.thread(
        release=release, target=lambda: dev.setup(cfg, progress=False)
    )

    worker.start()
    try:
        workers.arrived(entered)
        assert dev.is_busy() is True
    finally:
        release.set()
        workers.join(worker)

    assert dev.get_value() == cfg.value
    assert dev.is_busy() is False


def test_is_busy_false_when_idle() -> None:
    dev = FakeDevice(fast_mode=True)
    assert dev.is_busy() is False


def test_is_busy_false_for_lock_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    dev = FakeDevice(fast_mode=False)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.1, rampstep=0.1)
    owner_readings: list[bool] = []
    monkeypatch.setattr(
        fake_module.time, "sleep", lambda _seconds: owner_readings.append(dev.is_busy())
    )

    dev.setup(cfg, progress=False)

    assert dev.get_value() == cfg.value
    assert owner_readings == [False]


def test_device_operation_logs_outer_operation(caplog: LogCaptureFixture) -> None:
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.1, rampstep=0.01)

    with caplog.at_level(logging.INFO, logger="zcu_tools.device.fake"):
        dev.setup(cfg, progress=False)

    messages = [record.getMessage() for record in caplog.records]
    assert "device operation started: FakeDevice.setup" in messages
    assert any(
        message.startswith("device operation finished: FakeDevice.setup")
        for message in messages
    )
