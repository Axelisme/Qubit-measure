"""Concurrency tests for GlobalDeviceManager's registry-scoped lock.

The manager _lock must protect only the registry dict (register/drop/get).
Long I/O operations — setup() ramps, get_info() SCPI reads — must run outside
that lock so that an in-progress ramp on device A cannot block a get_info()
call on the independent device B.

These tests use FakeDevice so no real VISA session is required.  Event objects
(not sleep) are used for thread synchronisation — the test is deterministic
regardless of scheduler timing.
"""

from __future__ import annotations

import threading

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo, GlobalDeviceManager

# ---------------------------------------------------------------------------
# Fixture: clean registry before/after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry() -> object:
    """Ensure GlobalDeviceManager._devices is empty before and after each test.

    GlobalDeviceManager is a class-level singleton; without cleanup a test's
    registrations would leak into subsequent tests and cause spurious failures.
    """
    GlobalDeviceManager._devices.clear()
    yield
    GlobalDeviceManager._devices.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slow_ramp_device() -> FakeDevice:
    """FakeDevice with real sleep ramp and a tiny rampstep.

    The ramp spans many linspace steps, giving other threads a wide window to
    call manager APIs while the ramp holds the per-instance lock.
    """
    dev = FakeDevice(fast_mode=False)
    dev._rampstep = 1e-2  # many steps => long ramp window
    return dev


def _make_fast_device() -> FakeDevice:
    """FakeDevice that completes setup instantly (for the observer device B)."""
    dev = FakeDevice(fast_mode=True)
    dev._rampstep = 0.1
    return dev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_info_not_blocked_by_concurrent_setup_on_other_device() -> None:
    """get_info("B") must return before device A's ramp finishes.

    Old implementation: manager._lock was held for the entire setup() call,
    so get_info("B") would block until A's ramp completed.

    New implementation: manager._lock is released before per-device I/O, so
    get_info("B") resolves independently from A's ramp.

    Verification strategy
    ---------------------
    - ramp_started  : set inside A's _setup hook, guarantees thread 2 fires
                      while A definitely holds its per-instance lock.
    - writer_done   : set after A's setup() returns.
    - get_info_returned : set by thread 2 as soon as get_info("B") returns.

    We assert get_info_returned is set while writer_done is *not* yet set,
    proving get_info("B") completed before A's ramp finished.
    """
    dev_a = _make_slow_ramp_device()
    dev_b = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)

    cfg_a = FakeDeviceInfo(
        address="none", output="on", value=1.0, rampstep=dev_a._rampstep
    )
    cfg_b = FakeDeviceInfo(
        address="none", output="on", value=0.5, rampstep=dev_b._rampstep
    )
    # Pre-configure B so get_info() has a meaningful state to return.
    GlobalDeviceManager.setup_devices({"B": cfg_b})

    ramp_started = threading.Event()
    writer_done = threading.Event()
    get_info_returned = threading.Event()
    writer_done_at_get_info_time: list[bool] = []

    original_setup = dev_a._setup

    def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
        ramp_started.set()  # signal: A's per-instance lock is now held
        original_setup(*args, **kwargs)

    dev_a._setup = hooked_setup  # type: ignore[method-assign]

    def writer() -> None:
        try:
            GlobalDeviceManager.setup_devices({"A": cfg_a})
        finally:
            dev_a._setup = original_setup  # type: ignore[method-assign]
            writer_done.set()

    def reader() -> None:
        ramp_started.wait()  # wait until A's ramp is in progress
        GlobalDeviceManager.get_info("B")
        # Record whether writer_done was already set at the moment get_info returned.
        writer_done_at_get_info_time.append(writer_done.is_set())
        get_info_returned.set()

    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)
    t_writer.start()
    t_reader.start()

    # get_info("B") must return well before the ramp finishes.
    # Timeout of 5 s is generous; a blocked call would hold until the full ramp
    # (many steps × 0.01 s sleep each) completes — orders of magnitude longer.
    returned_in_time = get_info_returned.wait(timeout=5.0)

    t_writer.join()
    t_reader.join()

    assert returned_in_time, (
        "get_info('B') did not return within 5 s while A's ramp was in progress; "
        "the manager lock is likely still held across the setup() I/O."
    )
    assert not writer_done_at_get_info_time[0], (
        "get_info('B') returned only after device A's ramp finished; "
        "expected it to return before writer_done was set (lock-free path)."
    )


def test_get_all_info_returns_correct_snapshot() -> None:
    """get_all_info() returns the current state of all registered devices.

    This is a functional regression check: the new out-of-lock implementation
    must still return accurate info for every device in the registry.
    """
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)

    cfg_a = FakeDeviceInfo(address="none", output="on", value=0.3, rampstep=0.1)
    cfg_b = FakeDeviceInfo(address="none", output="off", value=0.7, rampstep=0.1)
    GlobalDeviceManager.setup_devices({"A": cfg_a, "B": cfg_b})

    all_info = GlobalDeviceManager.get_all_info()

    assert set(all_info.keys()) == {"A", "B"}

    # Narrow to FakeDeviceInfo via isinstance so pyright sees the right type.
    info_a = all_info["A"]
    info_b = all_info["B"]
    assert isinstance(info_a, FakeDeviceInfo)
    assert isinstance(info_b, FakeDeviceInfo)
    assert info_a.value == pytest.approx(0.3)
    assert info_b.value == pytest.approx(0.7)
    assert info_a.output == "on"
    assert info_b.output == "off"


def test_setup_devices_fast_fails_on_unknown_name() -> None:
    """Unknown device name in setup_devices raises ValueError before any setup."""
    dev = _make_fast_device()
    GlobalDeviceManager.register_device("known", dev)

    cfg = FakeDeviceInfo(address="none", output="on", value=0.0, rampstep=0.1)
    with pytest.raises(ValueError, match="unknown"):
        GlobalDeviceManager.setup_devices({"known": cfg, "unknown": cfg})


def test_setup_devices_validates_all_names_before_any_setup() -> None:
    """If any name is missing the whole batch is rejected — no partial setup."""
    dev_a = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    # "B" is not registered.

    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)

    with pytest.raises(ValueError, match="B"):
        GlobalDeviceManager.setup_devices({"A": cfg, "B": cfg})

    # A must not have been set up (value stays at the default 0.0).
    assert dev_a.get_value() == 0.0, (
        "Device A was set up despite the batch being rejected; "
        "fast-fail before any I/O is broken."
    )
