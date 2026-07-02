"""Concurrency tests for per-instance device operation and I/O locks.

These exercise BaseDevice's ``_op_lock`` through ``FakeDevice``, which is the
only device that runs without a real VISA session. The ``_io_lock`` check here is
limited to verifying that the base lock is a plain mutex, not a reentrant lock.
"""

from __future__ import annotations

import logging
import threading

import pytest
import zcu_tools.device.fake as fake_module
from _pytest.logging import LogCaptureFixture
from zcu_tools.device import DeviceBusyError, FakeDevice, FakeDeviceInfo


def _make_slow_ramp_device() -> FakeDevice:
    """FakeDevice with real (non-fast) ramp sleeps and a tiny rampstep, so a
    setup() ramp spans many observable steps — long enough for a second thread to
    attempt interleaving while the ramp holds op_lock."""
    dev = FakeDevice(fast_mode=False)
    # Small rampstep => many linspace steps => the ramp loop holds op_lock long
    # enough that get_info() should observe mid-ramp values through io_lock.
    dev._rampstep = 1e-2
    return dev


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


def test_setup_ramp_allows_concurrent_get_info() -> None:
    """While one thread runs a long setup() ramp, a second thread's get_info()
    should be able to read through the short io_lock and observe mid-ramp state."""
    dev = _make_slow_ramp_device()
    target = 1.0
    cfg = FakeDeviceInfo(
        address="none", output="on", value=target, rampstep=dev._rampstep
    )

    # Hook _setup to signal once the writer definitely holds op_lock, then start
    # reading: this pins the intended schedule (ramp first, reads during the
    # in-progress ramp) deterministically.
    ramp_started = threading.Event()
    observed: list[float] = []
    writer_done = threading.Event()

    original_setup = dev._setup

    def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
        ramp_started.set()
        original_setup(*args, **kwargs)

    dev._setup = hooked_setup  # type: ignore[method-assign]

    def writer() -> None:
        try:
            dev.setup(cfg, progress=False)
        finally:
            dev._setup = original_setup  # type: ignore[method-assign]
            writer_done.set()

    def reader() -> None:
        ramp_started.wait()
        while not writer_done.is_set():
            observed.append(dev.get_info().value)

    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)
    t_writer.start()
    t_reader.start()
    t_writer.join()
    t_reader.join()

    assert dev.get_value() == target
    intermediates = [v for v in observed if v not in (0.0, target)]
    assert intermediates, (
        "get_info() should be able to read during a long ramp and observe "
        "intermediate values"
    )


def test_concurrent_setup_raises_device_busy_error() -> None:
    """When a second thread calls setup() while the first is still ramping, it
    must raise DeviceBusyError immediately (fail-fast), not queue behind the ramp.
    A Barrier + Event guarantees the second call lands during the first ramp."""
    dev = _make_slow_ramp_device()
    cfg_a = FakeDeviceInfo(
        address="none", output="on", value=0.5, rampstep=dev._rampstep
    )
    cfg_b = FakeDeviceInfo(
        address="none", output="on", value=-0.5, rampstep=dev._rampstep
    )

    # ramp_started fires once the first thread has acquired op_lock and begun the
    # ramp, guaranteeing the second call is concurrent with an in-progress operation.
    ramp_started = threading.Event()
    errors: list[BaseException] = []

    def first_setup() -> None:
        # Inject a hook: signal after op_lock is acquired (i.e. inside _setup's
        # ramp loop) via a tiny subclass trick — we wrap _setup to set the event.
        original_setup = dev._setup

        def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
            ramp_started.set()
            original_setup(*args, **kwargs)

        dev._setup = hooked_setup  # type: ignore[method-assign]
        try:
            dev.setup(cfg_a, progress=False)
        finally:
            dev._setup = original_setup  # type: ignore[method-assign]

    def second_setup() -> None:
        ramp_started.wait()  # wait until the first thread definitely holds op_lock
        try:
            dev.setup(cfg_b, progress=False)
        except DeviceBusyError as exc:
            errors.append(exc)

    t1 = threading.Thread(target=first_setup)
    t2 = threading.Thread(target=second_setup)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 1, (
        f"Expected exactly one DeviceBusyError from the second setup(); got {errors}"
    )
    assert isinstance(errors[0], DeviceBusyError)


def test_nested_calls_do_not_deadlock() -> None:
    """Single-thread reentrancy regression: setup() -> _setup() ->
    set_output()/get_value() nest on the same thread; an RLock (not a plain Lock)
    must let this complete without self-deadlock."""
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.25, rampstep=0.01)

    dev.setup(cfg, progress=False)  # would hang here if op_lock were a plain Lock

    info = dev.get_info()
    assert info.value == 0.25
    assert info.output == "on"


def test_is_busy_true_during_ramp() -> None:
    """is_busy() must return True while another thread holds op_lock (mid-ramp),
    and False once idle. An Event (not sleep) synchronises the probe."""
    dev = _make_slow_ramp_device()
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=dev._rampstep)

    ramp_started = threading.Event()
    ramp_release = threading.Event()
    busy_readings: list[bool] = []

    original_setup = dev._setup

    def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
        ramp_started.set()
        original_setup(*args, **kwargs)
        # Keep op_lock a little longer so the probe can fire while still locked.
        # We can't extend artificially here — the probe must rely on the ramp itself.

    dev._setup = hooked_setup  # type: ignore[method-assign]

    def worker() -> None:
        try:
            dev.setup(cfg, progress=False)
        finally:
            dev._setup = original_setup  # type: ignore[method-assign]
            ramp_release.set()

    t = threading.Thread(target=worker)
    t.start()

    ramp_started.wait()
    # At this point the worker holds op_lock; is_busy() must be True.
    busy_readings.append(dev.is_busy())

    ramp_release.wait()
    t.join()
    # After setup completes, op_lock is free; is_busy() must be False.
    busy_readings.append(dev.is_busy())

    assert busy_readings[0] is True, "is_busy() should be True while ramp holds op_lock"
    assert busy_readings[1] is False, "is_busy() should be False after ramp completes"


def test_is_busy_false_when_idle() -> None:
    """is_busy() on an idle (unlocked) device must return False."""
    dev = FakeDevice(fast_mode=True)
    assert dev.is_busy() is False


def test_is_busy_false_for_lock_owner() -> None:
    """The thread that currently holds op_lock (i.e. is inside setup()) must see
    False from is_busy(), because RLock reentry by the owner succeeds."""
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.1, rampstep=0.01)

    owner_reading: list[bool] = []
    original_setup = dev._setup

    def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Called from within setup() which already holds op_lock.
        owner_reading.append(dev.is_busy())
        original_setup(*args, **kwargs)

    dev._setup = hooked_setup  # type: ignore[method-assign]
    try:
        dev.setup(cfg, progress=False)
    finally:
        dev._setup = original_setup  # type: ignore[method-assign]

    assert owner_reading == [False], (
        "is_busy() called by the op_lock owner should return False (RLock reentry)"
    )


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


def test_io_lock_is_plain_mutex() -> None:
    dev = FakeDevice(fast_mode=True)

    acquired_first = dev._io_lock.acquire(blocking=False)
    acquired_second = dev._io_lock.acquire(blocking=False)
    if acquired_second:
        dev._io_lock.release()
    if acquired_first:
        dev._io_lock.release()

    assert acquired_first is True
    assert acquired_second is False
