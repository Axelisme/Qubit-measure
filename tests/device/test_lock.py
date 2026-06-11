"""Concurrency tests for the per-instance device execution lock.

These exercise BaseDevice's ``_lock`` (the RLock added so that every public
operation on one device instance is serialized) through ``FakeDevice``, which is
the only device that runs without a real VISA session. There is no mock-VISA
infrastructure in the test suite, so the "two-thread write/query interleave"
assertion from the plan (item 3) is folded into the FakeDevice race test below
rather than building a large fake pyvisa session just to re-prove the same lock.
"""

from __future__ import annotations

import threading

from zcu_tools.device import DeviceBusyError, FakeDevice, FakeDeviceInfo


def _make_slow_ramp_device() -> FakeDevice:
    """FakeDevice with real (non-fast) ramp sleeps and a tiny rampstep, so a
    setup() ramp spans many observable steps — long enough for a second thread to
    attempt interleaving while the ramp holds the lock."""
    dev = FakeDevice(fast_mode=False)
    # Small rampstep => many linspace steps => the ramp loop holds the lock long
    # enough that an unlocked design would let get_info() observe a mid-ramp value.
    dev._rampstep = 1e-3
    return dev


def test_setup_ramp_not_interleaved_by_concurrent_get_info() -> None:
    """While one thread runs a long setup() ramp, a second thread's get_info()
    must observe either the pre-ramp value or the final value, never a mid-ramp
    intermediate. A barrier (not sleep) guarantees the reader fires during the
    ramp window, so the test is deterministic rather than timing-lucky."""
    dev = _make_slow_ramp_device()
    target = 1.0
    cfg = FakeDeviceInfo(
        address="none", output="on", value=target, rampstep=dev._rampstep
    )

    # With fail-fast setup(), the reader must not race for the lock before the
    # writer acquires it (the writer would raise DeviceBusyError-free but the
    # reader could win the lock first and serialize ahead of the ramp). Hook
    # _setup to signal once the writer definitely holds the lock, then start
    # reading: this pins the intended schedule (ramp first, reads contend with
    # the in-progress ramp) deterministically.
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
        # Read continuously for the whole ramp window (until the writer finishes),
        # not a fixed count — a fixed count could complete before the ramp even
        # starts and miss the race entirely. With the lock held by setup() for the
        # full ramp, every read must be 0.0 (pre-ramp) or target (post-ramp), never
        # an intermediate linspace point. (Confirmed failing when the lock is
        # removed: the reader then observes thousands of intermediates.)
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
    assert not intermediates, (
        f"get_info()/get_value() observed mid-ramp values {intermediates}; "
        "the per-instance lock failed to serialize the ramp."
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

    # ramp_started fires once the first thread has acquired the lock and begun the
    # ramp, guaranteeing the second call is concurrent with an in-progress operation.
    ramp_started = threading.Event()
    errors: list[BaseException] = []

    def first_setup() -> None:
        # Inject a hook: signal after the lock is acquired (i.e. inside _setup's
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
        ramp_started.wait()  # wait until the first thread definitely holds the lock
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

    dev.setup(cfg, progress=False)  # would hang here if _lock were a plain Lock

    info = dev.get_info()
    assert info.value == 0.25
    assert info.output == "on"


def test_is_busy_true_during_ramp() -> None:
    """is_busy() must return True while another thread holds the lock (mid-ramp),
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
        # Keep the lock a little longer so the probe can fire while still locked.
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
    # At this point the worker holds the lock; is_busy() must be True.
    busy_readings.append(dev.is_busy())

    ramp_release.wait()
    t.join()
    # After setup completes, the lock is free; is_busy() must be False.
    busy_readings.append(dev.is_busy())

    assert busy_readings[0] is True, (
        "is_busy() should be True while ramp holds the lock"
    )
    assert busy_readings[1] is False, "is_busy() should be False after ramp completes"


def test_is_busy_false_when_idle() -> None:
    """is_busy() on an idle (unlocked) device must return False."""
    dev = FakeDevice(fast_mode=True)
    assert dev.is_busy() is False


def test_is_busy_false_for_lock_owner() -> None:
    """The thread that currently holds the lock (i.e. is inside setup()) must see
    False from is_busy(), because RLock reentry by the owner succeeds."""
    dev = FakeDevice(fast_mode=True)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.1, rampstep=0.01)

    owner_reading: list[bool] = []
    original_setup = dev._setup

    def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Called from within setup() which already holds the lock.
        owner_reading.append(dev.is_busy())
        original_setup(*args, **kwargs)

    dev._setup = hooked_setup  # type: ignore[method-assign]
    try:
        dev.setup(cfg, progress=False)
    finally:
        dev._setup = original_setup  # type: ignore[method-assign]

    assert owner_reading == [False], (
        "is_busy() called by the lock owner should return False (RLock reentry)"
    )
