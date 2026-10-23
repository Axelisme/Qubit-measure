"""DeviceService live-read, cache version and background poll contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.device_models import (
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.events import DeviceChangedPayload
from zcu_tools.gui.session.ports import BackgroundAdmissionClosedError, OperationKind
from zcu_tools.gui.session.state import DeviceStatus

from tests.gui._completion_helpers import on_device_disconnected
from tests.gui._qt_waits import qt_until
from tests.gui.session.services._device_service_support import (
    connect_device,
    device_connect_request,
    make_device_service,
)
from tests.gui.session.services._device_waits import DeviceWorkers


def test_get_device_info_unchanged_does_not_bump_or_emit(
    qapp, device_workers: DeviceWorkers
):
    state = State(MagicMock())
    bus = EventBus()
    svc, device = make_device_service(device_workers, state=state, bus=bus)
    connect_device(
        svc, device_connect_request()
    )  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    # Driver returns the same value already cached → pure cache sync.
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    info = svc.get_device_info("dev1")

    assert info is not None
    # Read of an unchanged value must not advance the version (would spuriously
    # invalidate another client's expected_versions) nor emit a change event.
    assert state.version.get("device:dev1") == before
    assert events == []


def test_get_device_info_changed_bumps_and_emits(qapp, device_workers: DeviceWorkers):
    state = State(MagicMock())
    bus = EventBus()
    svc, device = make_device_service(device_workers, state=state, bus=bus)
    connect_device(
        svc, device_connect_request()
    )  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    # Driver value moved underneath us (e.g. external hardware change).
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=2.0)
    info = svc.get_device_info("dev1")

    assert info is not None and getattr(info, "value", None) == 2.0
    # A genuine state change discovered on read: bump + DEVICE_CHANGED so readers
    # re-query and dependent guards can catch the external change.
    assert state.version.get("device:dev1") == before + 1
    assert events == ["dev1"]
    cached = state.get_device("dev1")
    assert cached is not None and getattr(cached.info, "value", None) == 2.0
    # but the cached info on State is refreshed
    cached = state.get_device("dev1")
    assert cached is not None and getattr(cached.info, "value", None) == 2.0


def _drain_poll(workers: DeviceWorkers, qapp) -> None:
    """Wait for the off-main poll read + flush its queued main-thread delivery.

    ``poll_device_info`` runs ``read_work`` in the shared pool and marshals
    ``on_read`` back to the main thread; quiesce the runner(s) then pump the
    event loop so the queued ``on_read`` actually fires.
    """
    for bg in workers.runners:
        # The scheduled poll producer and queued owner delivery must both drain.
        assert bg.quiesce(timeout_ms=30_000)
    qapp.processEvents()
    qapp.processEvents()


class _DeferredPollBackground:
    """BackgroundExecutor fake that captures poll delivery for late-result tests."""

    def __init__(self, workers: DeviceWorkers) -> None:
        self.submitted = False
        self.closed = False
        self.background = workers.background()
        self._delivery: Callable[[], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if not run_in_pool:
            self.background.submit(
                work, run_in_pool=False, on_done=on_done, on_error=on_error
            )
            return
        if self.closed:
            raise BackgroundAdmissionClosedError("executor closed")
        self.submitted = True
        try:
            result = work()
        # tolerated: Match BackgroundExecutor's open work-failure contract and defer the original error.
        except Exception as exc:  # noqa: BLE001
            self._delivery = lambda exc=exc: on_error(exc)
        else:
            self._delivery = lambda result=result: on_done(result)

    def deliver(self) -> None:
        if self._delivery is None:
            raise AssertionError("no deferred poll delivery")
        delivery = self._delivery
        self._delivery = None
        delivery()


@pytest.mark.parametrize(
    "skip",
    ["missing", "memory_only", "connecting", "disconnecting", "gate", "closed"],
)
def test_poll_receipt_distinguishes_skipped_from_submitted_reads(
    qapp, device_workers: DeviceWorkers, skip: str
):
    state = State(MagicMock())
    bus = EventBus()
    gate = OperationGate(bus)
    bg = _DeferredPollBackground(device_workers)
    svc, device = make_device_service(
        device_workers, state=state, bus=bus, gate=gate, bg=bg
    )
    facet = DeviceControlFacet(bus=bus, device=svc, progress=MagicMock())
    if skip != "missing":
        connect_device(svc, device_connect_request())
    statuses = {
        "memory_only": DeviceStatus.MEMORY_ONLY,
        "connecting": DeviceStatus.CONNECTING,
        "disconnecting": DeviceStatus.DISCONNECTING,
    }
    if skip in statuses:
        state.set_device_status("dev1", statuses[skip])
    if skip == "gate":
        gate.register(
            99,
            OperationKind.DEVICE_DISCONNECT,
            owner_id="held",
            origin_kind="user",
            note="held mutation",
            resource_id="dev1",
        )
    bg.closed = skip == "closed"
    device.get_info.reset_mock()
    try:
        assert facet.try_poll_device_info("dev1") is False
        assert bg.submitted is False
        device.get_info.assert_not_called()
    finally:
        if skip == "gate":
            gate.release(99)

    if skip == "missing":
        connect_device(svc, device_connect_request())
    elif skip in statuses:
        state.set_device_status("dev1", DeviceStatus.CONNECTED)
    bg.closed = False
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=3.0)
    before = state.version.get("device:dev1")
    events: list[str] = []
    facet.on_device_changed(lambda event: events.append(event.name))

    assert facet.try_poll_device_info("dev1") is True
    assert bg.submitted is True
    assert state.version.get("device:dev1") == before
    assert events == []
    bg.deliver()
    assert state.version.get("device:dev1") == before + 1
    assert events == ["dev1"]


def test_poll_device_info_changed_value_bumps_and_emits_on_main(
    qapp, device_workers: DeviceWorkers
):
    """The off-main poll reads the driver on a worker; the main-thread on_done
    does the cache compare + bump + DEVICE_CHANGED. A value that drifted under
    us (mock: backgrounded set_value) shows up after a poll."""
    state = State(MagicMock())
    bus = EventBus()
    svc, device = make_device_service(device_workers, state=state, bus=bus)
    connect_device(
        svc, device_connect_request()
    )  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    # Simulate an external drift: the driver now reports a different value than
    # the cache (mock stand-in for a real hardware change between reads).
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=3.0)
    svc.poll_device_info("dev1")
    _drain_poll(device_workers, qapp)

    # State write (bump) + emit happened on the main thread (on_done), not the
    # worker — the version moved and exactly one DEVICE_CHANGED fired.
    assert state.version.get("device:dev1") == before + 1
    assert events == ["dev1"]
    cached = state.get_device("dev1")
    assert cached is not None and getattr(cached.info, "value", None) == 3.0


def test_poll_device_info_unchanged_value_does_not_bump_or_emit(
    qapp, device_workers: DeviceWorkers
):
    """An unchanged poll is a pure cache sync — no spurious version bump (would
    invalidate other clients' expected_versions) and no DEVICE_CHANGED."""
    state = State(MagicMock())
    bus = EventBus()
    svc, device = make_device_service(device_workers, state=state, bus=bus)
    connect_device(
        svc, device_connect_request()
    )  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    svc.poll_device_info("dev1")
    _drain_poll(device_workers, qapp)

    assert state.version.get("device:dev1") == before
    assert events == []


def test_poll_device_info_skips_memory_only_device(qapp, device_workers: DeviceWorkers):
    """A memory-only (not connected) device is not a live-read target — the poll
    skips it without submitting any worker / raising."""
    svc, device = make_device_service(device_workers)
    connect_device(svc, device_connect_request())
    # Disconnect (remember=True) → MEMORY_ONLY.
    drop_loop = device_workers.completion()
    on_device_disconnected(svc, lambda _r: drop_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    drop_loop.exec()
    device.get_info.reset_mock()

    svc.poll_device_info("dev1")
    _drain_poll(device_workers, qapp)

    # No driver read attempted for a memory-only device.
    device.get_info.assert_not_called()


def test_poll_device_info_polls_setting_up_device(qapp, device_workers: DeviceWorkers):
    """Setup/ramp is the safe mutation: poll may read current driver values and
    refresh cache/UI while setup remains in flight."""
    state = State(MagicMock())
    bus = EventBus()
    svc, device = make_device_service(device_workers, state=state, bus=bus)
    connect_device(svc, device_connect_request())

    block = device_workers.setup()
    device.setup.side_effect = block
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=4.0)

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=4.0))
    )
    try:
        qt_until(block.entered.is_set, label="setup worker entered")
        snapshot = svc.get_device_snapshot("dev1")
        assert snapshot is not None
        assert snapshot.status is DeviceStatus.SETTING_UP

        device.get_info.reset_mock()
        before = state.version.get("device:dev1")
        events: list[object] = []
        bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

        svc.poll_device_info("dev1")
        qt_until(
            lambda: state.version.get("device:dev1") == before + 1,
            label="setup poll delivery",
        )

        device.get_info.assert_called_once_with()
        assert state.version.get("device:dev1") == before + 1
        assert events == ["dev1"]
        cached = state.get_device("dev1")
        assert cached is not None
        assert cached.status is DeviceStatus.SETTING_UP
        assert getattr(cached.info, "value", None) == 4.0
    finally:
        block.release.set()
        qt_until(
            lambda: (
                (snapshot := svc.get_device_snapshot("dev1")) is not None
                and snapshot.status is DeviceStatus.CONNECTED
            ),
            label="setup finished",
        )


@pytest.mark.parametrize(
    "kind",
    [OperationKind.DEVICE_CONNECT, OperationKind.DEVICE_DISCONNECT],
)
def test_poll_device_info_skips_non_setup_device_mutation(
    qapp, device_workers: DeviceWorkers, kind: OperationKind
):
    """Connect/disconnect leases are not safe live-read windows, so poll skips
    them without submitting a worker."""
    gate = OperationGate(EventBus())
    svc, device = make_device_service(device_workers, gate=gate)
    connect_device(svc, device_connect_request())
    device.get_info.reset_mock()
    gate.register(
        99,
        kind,
        owner_id="held",
        origin_kind="user",
        note="held mutation",
        resource_id="dev1",
    )

    try:
        svc.poll_device_info("dev1")  # skipped without raising
        _drain_poll(device_workers, qapp)

        device.get_info.assert_not_called()
    finally:
        gate.release(99)


def test_poll_device_info_late_result_skips_after_non_setup_mutation(
    qapp, device_workers: DeviceWorkers
):
    """A delayed poll delivery must not bump State after the device enters a
    non-setup mutation."""
    state = State(MagicMock())
    bus = EventBus()
    bg = _DeferredPollBackground(device_workers)
    svc, device = make_device_service(device_workers, state=state, bus=bus, bg=bg)
    connect_device(svc, device_connect_request())
    before = state.version.get("device:dev1")
    events: list[object] = []
    bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    device.get_info.return_value = FakeDeviceInfo(address="addr", value=8.0)

    svc.poll_device_info("dev1")
    assert bg.submitted is True
    assert state.version.get("device:dev1") == before

    state.set_device_status("dev1", DeviceStatus.DISCONNECTING)
    after_status_change = state.version.get("device:dev1")
    bg.deliver()

    assert state.version.get("device:dev1") == after_status_change
    assert events == []
    cached = state.get_device("dev1")
    assert cached is not None and getattr(cached.info, "value", None) == 0.0


def test_poll_device_info_swallows_read_failure(qapp, device_workers: DeviceWorkers):
    """A single failed read (timeout / driver boom) is logged and dropped so the
    poller keeps ticking — best-effort, not Fast-Fail."""
    state = State(MagicMock())
    bus = EventBus()
    svc, device = make_device_service(device_workers, state=state, bus=bus)
    connect_device(svc, device_connect_request())
    before = state.version.get("device:dev1")
    events: list[object] = []
    bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    device.get_info.side_effect = RuntimeError("read timeout")
    svc.poll_device_info("dev1")  # must not raise
    _drain_poll(device_workers, qapp)

    # Failure swallowed: no bump, no emit, state untouched.
    assert state.version.get("device:dev1") == before
    assert events == []
