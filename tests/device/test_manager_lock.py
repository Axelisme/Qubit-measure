"""Registry, setup, snapshot, and cancellation contracts for DeviceManager.

The registry lock must not cover device I/O. These tests use FakeDevice and
driver gates without a real VISA session.
"""

from __future__ import annotations

import re
import threading
from typing import Any, cast

import pytest
from zcu_tools.device import DeviceManager, FakeDevice, FakeDeviceInfo

from ._snapshot_fakes import SnapshotDevice
from ._threads import DeviceThreads

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_info_not_blocked_by_concurrent_setup_on_other_device(
    manager: DeviceManager,
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev_a = FakeDevice(fast_mode=False)
    dev_b = FakeDevice(fast_mode=True)
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)
    cfg_a = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.25)
    cfg_b = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    manager.setup_devices({"B": cfg_b}, progress=False)
    entered, release = paused_ramp
    worker = workers.thread(
        release=release,
        target=lambda: manager.setup_devices({"A": cfg_a}, progress=False),
    )

    worker.start()
    try:
        workers.arrived(entered)
        info_b = manager.get_info("B")
        assert isinstance(info_b, FakeDeviceInfo)
        assert info_b.value == cfg_b.value
        assert worker.is_alive(), "get_info('B') waited for A's ramp to finish"
    finally:
        release.set()
        workers.join(worker)

    info_a = manager.get_info("A")
    assert isinstance(info_a, FakeDeviceInfo)
    assert info_a.value == cfg_a.value


def test_register_device_rejects_non_base_device_without_mutating_registry(
    manager: DeviceManager,
) -> None:
    # tolerated(cast): Test registry rejection of a non-BaseDevice through its typed entry.
    bad_device = cast(Any, object())

    with pytest.raises(TypeError, match="register_device expected BaseDevice"):
        manager.register_device("bad", bad_device)

    assert "bad" not in manager.get_all_devices()


@pytest.mark.parametrize("second_name_exists", [False, True])
def test_register_device_rejects_second_name_without_mutating_registry(
    manager: DeviceManager, second_name_exists: bool
) -> None:
    device = FakeDevice(fast_mode=True)
    manager.register_device("flux", device)
    if second_name_exists:
        manager.register_device("other", FakeDevice(fast_mode=True))
    before = manager.get_all_devices()

    with pytest.raises(ValueError, match="already registered"):
        manager.register_device("other", device)

    assert manager.get_all_devices() == before
    assert manager.get_device("flux") is device


@pytest.mark.parametrize("replace_driver", [False, True])
def test_register_device_same_name_preserves_overwrite_contract(
    manager: DeviceManager, replace_driver: bool
) -> None:
    original = FakeDevice(fast_mode=True)
    manager.register_device("flux", original)
    replacement = FakeDevice(fast_mode=True) if replace_driver else original

    with pytest.warns(UserWarning, match="already registered, overwriting"):
        manager.register_device("flux", replacement)

    assert manager.get_all_devices() == {"flux": replacement}


def test_managers_isolate_names_setup_and_disconnect(manager: DeviceManager) -> None:
    other = DeviceManager()
    first = FakeDevice(fast_mode=True)
    second = FakeDevice(fast_mode=True)
    manager.register_device("flux", first)
    other.register_device("flux", second)

    manager.setup_devices(
        {"flux": FakeDeviceInfo(address="none", output="on", value=0.5)},
        progress=False,
    )
    assert manager.get_device("flux") is first
    assert other.get_device("flux") is second
    assert second.get_value() == 0.0

    manager.close_all_devices()
    assert manager.get_all_devices() == {}
    assert other.get_device("flux") is second
    assert second.get_value() == 0.0


def test_get_all_devices_is_an_independent_mapping(manager: DeviceManager) -> None:
    device = FakeDevice(fast_mode=True)
    manager.register_device("flux", device)
    snapshot = manager.get_all_devices()
    snapshot.clear()
    assert manager.get_device("flux") is device


@pytest.mark.parametrize("name", ["ok", "", " Coil/A-1 ", "線圈", "coil．left"])
def test_register_device_preserves_other_names(
    manager: DeviceManager, name: str
) -> None:
    dev = FakeDevice(fast_mode=True)

    manager.register_device(name, dev)

    assert manager.get_all_devices() == {name: dev}
    assert manager.get_device(name) is dev


@pytest.mark.parametrize("name", ["coil.left", ".coil", "coil.", "."])
def test_register_device_rejects_dots_without_registry_change_or_io(
    manager: DeviceManager, name: str
) -> None:
    original = SnapshotDevice()
    manager.register_device("coil", original)
    candidate = SnapshotDevice()
    before = manager.get_all_devices()

    with pytest.raises(ValueError, match=re.escape(name)):
        manager.register_device(name, candidate)

    assert manager.get_all_devices() == before
    assert original.events == []
    assert candidate.events == []


def test_get_all_info_returns_correct_snapshot(manager: DeviceManager) -> None:
    """get_all_info() returns the current state of all registered devices.

    This is a functional regression check: the new out-of-lock implementation
    must still return accurate info for every device in the registry.
    """
    dev_a = FakeDevice(fast_mode=True)
    dev_b = FakeDevice(fast_mode=True)
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)

    cfg_a = FakeDeviceInfo(address="none", output="on", value=0.3, rampstep=0.1)
    cfg_b = FakeDeviceInfo(address="none", output="off", value=0.7, rampstep=0.1)
    manager.setup_devices({"A": cfg_a, "B": cfg_b})

    all_info = manager.get_all_info()

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


def test_setup_devices_fast_fails_on_unknown_name(manager: DeviceManager) -> None:
    """Unknown device name in setup_devices raises ValueError before any setup."""
    dev = FakeDevice(fast_mode=True)
    manager.register_device("known", dev)

    cfg = FakeDeviceInfo(address="none", output="on", value=0.0, rampstep=0.1)
    with pytest.raises(ValueError, match="unknown"):
        manager.setup_devices({"known": cfg, "unknown": cfg})


def test_setup_devices_validates_all_names_before_any_setup(
    manager: DeviceManager,
) -> None:
    """If any name is missing the whole batch is rejected — no partial setup."""
    dev_a = FakeDevice(fast_mode=True)
    manager.register_device("A", dev_a)
    # "B" is not registered.

    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)

    with pytest.raises(ValueError, match="B"):
        manager.setup_devices({"A": cfg, "B": cfg})

    # A must not have been set up (value stays at the default 0.0).
    assert dev_a.get_value() == 0.0, (
        "Device A was set up despite the batch being rejected; "
        "fast-fail before any I/O is broken."
    )


def test_setup_devices_validates_names_before_cancel_short_circuit(
    manager: DeviceManager,
) -> None:
    dev = FakeDevice(fast_mode=True)
    manager.register_device("known", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    with pytest.raises(ValueError, match="unknown"):
        manager.setup_devices(
            {"known": cfg, "unknown": cfg},
            cancel_signal=cancel_signal,
        )

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_explicit_cancel_signal_skips_before_first_device(
    manager: DeviceManager,
) -> None:
    dev = FakeDevice(fast_mode=True)
    manager.register_device("A", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    manager.setup_devices({"A": cfg}, cancel_signal=cancel_signal)

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_mid_ramp_cancel_stops_before_next_device(
    manager: DeviceManager,
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev_a = FakeDevice(fast_mode=False)
    dev_b = FakeDevice(fast_mode=True)
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)
    cfg_a = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.01)
    cfg_b = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    entered, release = paused_ramp
    worker = workers.thread(
        target=lambda: manager.setup_devices(
            {"A": cfg_a, "B": cfg_b},
            cancel_signal=cancel_signal,
        ),
        release=release,
    )
    worker.start()
    try:
        workers.arrived(entered)
        cancel_signal.set()
    finally:
        release.set()
        workers.join(worker)

    assert cancel_signal.is_set()
    assert dev_a.get_output() == "on"
    assert 0.0 < dev_a.get_value() < 1.0
    assert dev_b.get_output() == "off"
    assert dev_b.get_value() == 0.0
