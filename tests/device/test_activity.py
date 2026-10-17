"""Observe driver activity only through the manager's public snapshot seam."""

import time
from threading import Event

import pytest
from zcu_tools.device import DeviceBusyError, DeviceManager, FakeDevice, FakeDeviceInfo

from ._snapshot_fakes import SnapshotDevice
from ._threads import DeviceThreads


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Control the external monotonic clock without changing driver logic."""
    now = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def registered() -> tuple[DeviceManager, SnapshotDevice]:
    manager = DeviceManager()
    device = SnapshotDevice()
    manager.register_device("flux", device)
    return manager, device


def test_activity_initial_state_and_unobserved_operations_stay_one_per_device(
    registered: tuple[DeviceManager, SnapshotDevice], clock: list[float]
) -> None:
    manager, device = registered
    other = SnapshotDevice()
    manager.register_device("other", other)

    initial = manager.activity_snapshot()
    assert set(initial) == {"flux", "other"}
    assert all(not state.in_operation for state in initial.values())
    assert all(state.last_finished_monotonic is None for state in initial.values())

    # No consumer reads activity during this burst; each device still has one record.
    for value in range(32):
        clock[0] = 20.0 + value
        device.set_value(float(value))
    other.output_on()

    snapshot = manager.activity_snapshot()
    assert len(snapshot) == 2
    assert all(not state.in_operation for state in snapshot.values())
    assert all(state.last_finished_monotonic == 51.0 for state in snapshot.values())
    assert device.events == ["write"] * 32
    assert other.events == []
    # Earlier snapshots remain values, not live driver views.
    assert initial["flux"].last_finished_monotonic is None


def test_activity_during_nested_setup_and_after_outer_completion(
    registered: tuple[DeviceManager, SnapshotDevice],
    paused_ramp: tuple[Event, Event],
    workers: DeviceThreads,
    clock: list[float],
) -> None:
    manager, device = registered
    entered, release = paused_ramp
    # SnapshotDevice normally disables sleeps; this real FakeDevice collaborator
    # pauses setup after its nested set_output has returned.
    ramp = FakeDevice()
    manager.register_device("ramp", ramp)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.25)
    worker = workers.thread(
        release=release, target=lambda: ramp.setup(cfg, progress=False)
    )
    worker.start()
    try:
        workers.arrived(entered)
        clock[0] = 20.0
        # This read returns while the operation is still blocked on release.
        during = manager.activity_snapshot()
        assert during["ramp"].in_operation
        assert during["ramp"].last_finished_monotonic is None
        assert not during["flux"].in_operation
        assert device.events == []
    finally:
        release.set()
        workers.join(worker)

    ended = manager.activity_snapshot()["ramp"]
    assert not ended.in_operation
    assert ended.last_finished_monotonic == 20.0
    assert during["ramp"].in_operation
    assert during["ramp"].last_finished_monotonic is None


def test_activity_failure_finishes_and_next_operation_can_succeed(
    registered: tuple[DeviceManager, SnapshotDevice], clock: list[float]
) -> None:
    manager, device = registered
    cfg = FakeDeviceInfo(address="wrong-address")
    with pytest.raises(RuntimeError, match="address"):
        device.setup(cfg, progress=False)

    failed = manager.activity_snapshot()["flux"]
    assert not failed.in_operation
    assert failed.last_finished_monotonic == 10.0

    clock[0] = 30.0
    assert device.set_value(0.5) == 0.5
    recovered = manager.activity_snapshot()["flux"]
    assert not recovered.in_operation
    assert recovered.last_finished_monotonic == 30.0
    assert failed.last_finished_monotonic == 10.0
    assert device.events == ["write"]


def test_activity_busy_rejection_does_not_finish_the_running_operation(
    paused_ramp: tuple[Event, Event], workers: DeviceThreads, clock: list[float]
) -> None:
    manager = DeviceManager()
    device = FakeDevice()
    manager.register_device("flux", device)
    device.output_on()
    clock[0] = 20.0
    entered, release = paused_ramp
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.25)
    worker = workers.thread(
        release=release, target=lambda: device.setup(cfg, progress=False)
    )
    worker.start()
    try:
        workers.arrived(entered)
        clock[0] = 30.0
        with pytest.raises(DeviceBusyError):
            device.set_value(-0.5)
        rejected = manager.activity_snapshot()["flux"]
        assert rejected.in_operation
        assert rejected.last_finished_monotonic == 10.0
        clock[0] = 40.0
    finally:
        release.set()
        workers.join(worker)

    ended = manager.activity_snapshot()["flux"]
    assert not ended.in_operation
    assert ended.last_finished_monotonic == 40.0
    assert device.get_value() == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [("in_operation", True), ("last_finished_monotonic", 999.0)],
)
def test_activity_snapshot_cannot_modify_driver_state(
    registered: tuple[DeviceManager, SnapshotDevice], field: str, value: bool | float
) -> None:
    manager, _device = registered
    snapshot = manager.activity_snapshot()
    state = snapshot["flux"]
    with pytest.raises((AttributeError, TypeError)):
        setattr(state, field, value)
    snapshot.clear()

    unchanged = manager.activity_snapshot()
    assert set(unchanged) == {"flux"}
    assert not unchanged["flux"].in_operation
    assert unchanged["flux"].last_finished_monotonic is None


@pytest.mark.parametrize("close", [False, True], ids=["drop", "close"])
def test_activity_snapshot_excludes_removed_devices(
    registered: tuple[DeviceManager, SnapshotDevice], close: bool
) -> None:
    manager, device = registered
    before = manager.activity_snapshot()
    if close:
        manager.close_device("flux")
    else:
        manager.drop_device("flux")
    device.set_value(1.0)

    assert manager.activity_snapshot() == {}
    assert set(before) == {"flux"}
    assert before["flux"].last_finished_monotonic is None
