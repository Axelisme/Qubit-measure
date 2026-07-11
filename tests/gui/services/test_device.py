"""Tests for async DeviceService commands and snapshot boundaries."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.main.services.operation_gate import OperationGate
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui.services._device_fakes import FakeDeviceRegistry

# Every BackgroundRunner created in a test is registered here so the autouse
# teardown can quiesce it: a DeviceService runs its commands on a dedicated worker
# thread whose done/failed outcome is delivered via a queued main-thread signal.
# If the service (and its runner) is GC'd while that delivery is still queued, the
# next processEvents() dispatches it onto a freed C++ object and segfaults.
_LIVE_BG: list[BackgroundRunner] = []


def _bg() -> BackgroundRunner:
    bg = BackgroundRunner()
    _LIVE_BG.append(bg)
    return bg


@pytest.fixture(autouse=True)
def _quiesce_services():
    """Drain every test-created BackgroundRunner before its objects are GC'd."""
    yield
    for bg in _LIVE_BG:
        bg.quiesce()
    _LIVE_BG.clear()


@pytest.fixture(autouse=True)
def _clean_devices():
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


def _drain_until(
    condition: Callable[[], bool], *, timeout: float = 3.0, label: str = "condition"
) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {label}")


def _make_svc(
    driver: MagicMock | None = None, gate: OperationGate | None = None
) -> tuple[DeviceService, MagicMock]:
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    bg = _bg()
    resolved_gate = gate or OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(resolved_gate, handles, progress, bg, bus)
    return (
        DeviceService(
            bus,
            State(MagicMock()),
            resolved_gate,
            bg,
            runner,
            handles,
            driver_factory=lambda _type, _address: device,  # type: ignore[arg-type]
            device_registry=FakeDeviceRegistry(),
        ),
        device,
    )


def _req(name: str = "dev1") -> ConnectDeviceRequest:
    return ConnectDeviceRequest(type_name="FakeDevice", name=name, address="addr")


def _connect(svc: DeviceService, req: ConnectDeviceRequest) -> None:
    connected: list[object] = []
    errors: list[str] = []
    svc.device_connected.connect(connected.append)
    svc.operation_failed.connect(lambda _name, error: errors.append(error))
    svc.start_connect_device(req)
    _drain_until(lambda: bool(connected or errors), label=f"connect {req.name}")
    assert not errors


def test_device_connect_returns_operation_handle(qapp):
    svc, _device = _make_svc()
    connected: list[object] = []
    errors: list[str] = []
    svc.device_connected.connect(connected.append)
    svc.operation_failed.connect(lambda _name, error: errors.append(error))
    token = svc.start_connect_device(_req())
    _drain_until(lambda: bool(connected or errors), label="device connect")
    assert not errors
    # connect now returns an operation token (parity with setup) for awaiting.
    assert isinstance(token, int)
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]


def test_device_setup_updates_value_and_disconnect_updates_snapshot(qapp):
    svc, device = _make_svc()
    _connect(svc, _req())
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]

    # Setting an output value goes through setup (ramped/cancellable), not a
    # separate set_value: the post-setup get_info reflects the new value.
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=1.0)
    setup_loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: setup_loop.quit())
    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=1.0))
    )
    setup_loop.exec()
    device.setup.assert_called_once()
    assert svc.get_device_snapshot("dev1").info.value == 1.0  # type: ignore[union-attr]

    drop_loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: drop_loop.quit())
    disc_token = svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    assert isinstance(disc_token, int)  # disconnect also returns a handle
    drop_loop.exec()
    device.close.assert_called_once_with()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.MEMORY_ONLY  # type: ignore[union-attr]


def test_device_setup_started_and_finished_keep_operation_origin(qapp) -> None:
    svc, device = _make_svc()
    _connect(svc, _req())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=1.0)
    observed: list[tuple[object, EventMeta]] = []
    svc._bus.subscribe_with_meta(
        DeviceSetupStartedPayload,
        lambda payload, meta: observed.append((payload, meta)),
    )
    svc._bus.subscribe_with_meta(
        DeviceSetupFinishedPayload,
        lambda payload, meta: observed.append((payload, meta)),
    )
    loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: loop.quit())

    with svc._bus.origin(EventOrigin(kind="agent", client_id="client-a")):
        token = svc.start_setup_device(
            SetupDeviceRequest(
                name="dev1", info=FakeDeviceInfo(address="addr", value=1.0)
            )
        )
    loop.exec()

    assert [type(payload) for payload, _meta in observed] == [
        DeviceSetupStartedPayload,
        DeviceSetupFinishedPayload,
    ]
    assert [meta.origin for _payload, meta in observed] == [
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
        EventOrigin(kind="agent", client_id="client-a", operation_id=str(token)),
    ]


def test_device_mutation_is_globally_exclusive_and_blocks_same_device_read(qapp):
    gate = OperationGate(EventBus())
    svc, _device = _make_svc(gate=gate)
    _connect(svc, _req())
    gate.register(
        1,
        OperationKind.DEVICE_SETUP,
        owner_id="held",
        origin_kind="user",
        note="held setup",
        resource_id="dev1",
    )

    with pytest.raises(OperationConflictError, match="Cannot start"):
        svc.start_setup_device(
            SetupDeviceRequest(
                name="dev1", info=FakeDeviceInfo(address="addr", value=1.0)
            )
        )
    with pytest.raises(OperationConflictError, match="Cannot read"):
        svc.get_device_info("dev1")
    gate.release(1)


def test_cancel_missing_device_operation_is_failed_precondition(qapp):  # noqa: ARG001
    svc, _device = _make_svc()

    with pytest.raises(FailedPreconditionError, match="No operation") as exc_info:
        svc.cancel_device_operation("dev1")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


def test_device_connect_failure_is_reported_without_live_registration(qapp):
    svc, device = _make_svc()
    device.get_info.side_effect = RuntimeError("cannot query")
    errors: list[str] = []
    loop = QEventLoop()
    svc.operation_failed.connect(
        lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_connect_device(_req())
    loop.exec()

    assert errors
    assert "cannot query" in errors[0]
    assert "dev1" not in svc._registry.get_all_devices()
    device.close.assert_called_once_with()
    assert svc.get_device_snapshot("dev1") is None


def test_setup_uses_gate_and_returns_to_connected_snapshot(qapp):
    svc, device = _make_svc()
    _connect(svc, _req())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=2.0)
    loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )
    active = svc.get_active_device_setups()
    assert [s.device_name for s in active] == ["dev1"]
    loop.exec()

    device.setup.assert_called_once()
    assert svc.get_active_device_setups() == ()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]


def test_active_device_setups_enumerates_concurrent_setups(qapp):
    """Phase C: two devices setting up concurrently both appear, sorted by name,
    and each in-flight op is reported with its kind (regression for the old
    single-valued getter that returned only one)."""
    import time

    # Per-name drivers so two devices coexist in one service.
    drivers: dict[str, MagicMock] = {}

    def factory(_type: str, _address: str) -> MagicMock:
        # The service does not pass the name to the factory, so build a generic
        # driver and let the caller register it under the connecting name.
        dev = MagicMock()
        dev.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
        return dev

    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    bg_svc = _bg()
    resolved_gate = OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(resolved_gate, handles, progress, bg_svc, bus)
    registry = FakeDeviceRegistry()
    svc = DeviceService(
        bus,
        State(MagicMock()),
        resolved_gate,
        bg_svc,
        runner,
        handles,
        driver_factory=factory,  # type: ignore[arg-type]
        device_registry=registry,
    )

    for name in ("alpha", "beta"):
        loop = QEventLoop()
        svc.device_connected.connect(lambda _r: loop.quit())
        svc.start_connect_device(_req(name))
        loop.exec()
        drivers[name] = cast(MagicMock, registry.get_device(name))

    # Both setups block on the same gate event so they stay in-flight together.
    release = threading.Event()

    def blocking_setup(_info, stop_event=None):
        release.wait(timeout=3.0)

    for name in ("alpha", "beta"):
        drivers[name].setup.side_effect = blocking_setup
        drivers[name].get_info.return_value = FakeDeviceInfo(address="addr", value=1.0)
        svc.start_setup_device(
            SetupDeviceRequest(
                name=name, info=FakeDeviceInfo(address="addr", value=1.0)
            )
        )

    # Wait until both workers have entered setup (status flipped to SETTING_UP).
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if len(svc.get_active_device_setups()) == 2:
            break
        time.sleep(0.01)

    setups = svc.get_active_device_setups()
    assert [s.device_name for s in setups] == ["alpha", "beta"]  # sorted by name

    ops = svc.get_active_device_operations()
    assert [o.device_name for o in ops] == ["alpha", "beta"]
    assert all(o.kind is OperationKind.DEVICE_SETUP for o in ops)
    assert all(o.snapshot.status is DeviceStatus.SETTING_UP for o in ops)

    # Let both setups finish so the autouse teardown can quiesce cleanly.
    finished: set[str] = set()
    done = QEventLoop()
    svc.setup_finished.connect(
        lambda name: (finished.add(name), len(finished) == 2 and done.quit())
    )
    release.set()
    spin_deadline = time.monotonic() + 3.0
    while len(finished) < 2 and time.monotonic() < spin_deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert finished == {"alpha", "beta"}


def test_wait_setup_done_wakes_from_worker_thread_without_deadlock(qapp):
    """wait_setup_done runs off the main thread; with the main event loop
    spinning, the setup terminal wakes it promptly (regression: previously it
    deadlocked because the handler occupied the main thread it needed)."""
    import threading
    import time

    svc, device = _make_svc()
    _connect(svc, _req())

    # Make setup take ~0.5s so the waiter genuinely has to block.
    def slow_setup(_info, stop_event=None):
        time.sleep(0.5)

    device.setup.side_effect = slow_setup
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=2.0)

    token = svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )

    result: dict[str, object] = {}

    def waiter() -> None:
        t0 = time.monotonic()
        outcome = svc._handles.await_outcome(token, timeout=3.0)
        result["outcome"] = outcome
        result["dt"] = time.monotonic() - t0

    wt = threading.Thread(target=waiter)
    wt.start()

    # Spin the main event loop (as the real GUI does) so the worker's
    # setup_finished signal is dispatched and the gate lease is released.
    deadline = time.monotonic() + 3.0
    while wt.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    wt.join(timeout=1.0)

    await_result: Any = result.get("outcome")
    assert await_result is not None
    assert await_result.reason == "completed"
    assert await_result.outcome is not None
    assert await_result.outcome.status == "finished"
    dt = result.get("dt")
    assert isinstance(dt, float)
    assert dt < 2.0  # woke shortly after the 0.5s setup, not on timeout


def test_wait_setup_done_raises_on_setup_failure(qapp):
    import threading
    import time

    svc, device = _make_svc()
    _connect(svc, _req())

    def failing_setup(_info, stop_event=None):
        time.sleep(0.2)
        raise RuntimeError("hardware boom")

    device.setup.side_effect = failing_setup

    token = svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )

    result: dict[str, object] = {}

    def waiter() -> None:
        result["outcome"] = svc._handles.await_outcome(token, timeout=3.0)

    wt = threading.Thread(target=waiter)
    wt.start()
    deadline = time.monotonic() + 3.0
    while wt.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    wt.join(timeout=1.0)

    await_result: Any = result.get("outcome")
    assert await_result is not None
    assert await_result.reason == "completed"
    assert await_result.outcome is not None
    assert await_result.outcome.status == "failed"
    assert "hardware boom" in str(await_result.outcome.error)


def test_disconnect_close_failure_retains_connected_device_and_releases_gate(qapp):
    svc, device = _make_svc()
    _connect(svc, _req())
    device.close.side_effect = OSError("close failed")
    loop = QEventLoop()
    errors: list[str] = []
    svc.operation_failed.connect(
        lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    loop.exec()

    assert "close failed" in errors[0]
    assert "dev1" in svc._registry.get_all_devices()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]
    device.close.side_effect = None
    retry_loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: retry_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    retry_loop.exec()


def test_failing_device_changed_subscriber_does_not_abort_connect(qapp):
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    bus = EventBus()
    bg_svc = _bg()
    resolved_gate = OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(resolved_gate, handles, progress, bg_svc, bus)
    svc = DeviceService(
        bus,
        State(MagicMock()),
        resolved_gate,
        bg_svc,
        runner,
        handles,
        driver_factory=lambda _type, _address: MagicMock(),  # type: ignore[arg-type]
        device_registry=FakeDeviceRegistry(),
    )
    # A DEVICE_CHANGED subscriber raising (e.g. a View redraw bug) is swallowed +
    # logged by the EventBus; it must NOT abort the connect or roll back the
    # optimistic device state (one bad subscriber must not break the publisher).
    bus.subscribe(
        DeviceChangedPayload,
        MagicMock(side_effect=RuntimeError("render failed")),
    )

    svc.start_connect_device(_req())  # no raise — the subscriber failure is swallowed

    # The optimistic device state survives (the operation proceeds; the lease is
    # released at its real terminal, not on the subscriber's failure) — previously
    # the re-raising bus rolled this back to None.
    assert svc.get_device_snapshot("dev1") is not None


def test_connect_writes_device_state_to_state_with_remember(qapp):
    svc, _device = _make_svc()
    state = svc._state  # white-box: State is the device-state SSOT
    _connect(svc, ConnectDeviceRequest("FakeDevice", "dev1", "addr", remember=False))
    dev = state.get_device("dev1")
    assert dev is not None
    assert dev.status is DeviceStatus.CONNECTED
    # `remember` is now persistent device state, not a transient request attribute.
    assert dev.remember is False


def test_get_device_info_unchanged_does_not_bump_or_emit(qapp):
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    # Driver returns the same value already cached → pure cache sync.
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    info = svc.get_device_info("dev1")

    assert info is not None
    # Read of an unchanged value must not advance the version (would spuriously
    # invalidate another client's expected_versions) nor emit a change event.
    assert state.version.get("device:dev1") == before
    assert events == []


def test_get_device_info_changed_bumps_and_emits(qapp):
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

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


def _drain_poll(qapp) -> None:
    """Wait for the off-main poll read + flush its queued main-thread delivery.

    ``poll_device_info`` runs ``read_work`` in the shared pool and marshals
    ``on_read`` back to the main thread; quiesce the runner(s) then pump the
    event loop so the queued ``on_read`` actually fires.
    """
    for bg in _LIVE_BG:
        bg.quiesce()
    qapp.processEvents()
    qapp.processEvents()


class _DeferredPollBackground:
    """BackgroundExecutor fake that captures poll delivery for late-result tests."""

    def __init__(self) -> None:
        self.submitted = False
        self._delivery: Callable[[], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        assert run_in_pool is True
        self.submitted = True
        try:
            result = work()
        except Exception as exc:
            self._delivery = lambda exc=exc: on_error(exc)
        else:
            self._delivery = lambda result=result: on_done(result)

    def deliver(self) -> None:
        if self._delivery is None:
            raise AssertionError("no deferred poll delivery")
        delivery = self._delivery
        self._delivery = None
        delivery()


def test_poll_device_info_changed_value_bumps_and_emits_on_main(qapp):
    """The off-main poll reads the driver on a worker; the main-thread on_done
    does the cache compare + bump + DEVICE_CHANGED. A value that drifted under
    us (mock: backgrounded set_value) shows up after a poll."""
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    # Simulate an external drift: the driver now reports a different value than
    # the cache (mock stand-in for a real hardware change between reads).
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=3.0)
    svc.poll_device_info("dev1")
    _drain_poll(qapp)

    # State write (bump) + emit happened on the main thread (on_done), not the
    # worker — the version moved and exactly one DEVICE_CHANGED fired.
    assert state.version.get("device:dev1") == before + 1
    assert events == ["dev1"]
    cached = state.get_device("dev1")
    assert cached is not None and getattr(cached.info, "value", None) == 3.0


def test_poll_device_info_unchanged_value_does_not_bump_or_emit(qapp):
    """An unchanged poll is a pure cache sync — no spurious version bump (would
    invalidate other clients' expected_versions) and no DEVICE_CHANGED."""
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())  # connected with FakeDeviceInfo(value=0.0)
    before = state.version.get("device:dev1")
    events: list[object] = []
    svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    svc.poll_device_info("dev1")
    _drain_poll(qapp)

    assert state.version.get("device:dev1") == before
    assert events == []


def test_poll_device_info_skips_memory_only_device(qapp):
    """A memory-only (not connected) device is not a live-read target — the poll
    skips it without submitting any worker / raising."""
    svc, device = _make_svc()
    _connect(svc, _req())
    # Disconnect (remember=True) → MEMORY_ONLY.
    drop_loop = QEventLoop()
    svc.device_disconnected.connect(lambda _r: drop_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    drop_loop.exec()
    device.get_info.reset_mock()

    svc.poll_device_info("dev1")
    _drain_poll(qapp)

    # No driver read attempted for a memory-only device.
    device.get_info.assert_not_called()


def test_poll_device_info_polls_setting_up_device(qapp):
    """Setup/ramp is the safe mutation: poll may read current driver values and
    refresh cache/UI while setup remains in flight."""
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())

    entered = threading.Event()
    release = threading.Event()

    def blocking_setup(_info, stop_event=None):
        entered.set()
        release.wait()

    device.setup.side_effect = blocking_setup
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=4.0)

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=4.0))
    )
    try:
        _drain_until(entered.is_set, label="setup worker entered")
        assert svc.get_device_snapshot("dev1").status is DeviceStatus.SETTING_UP  # type: ignore[union-attr]

        device.get_info.reset_mock()
        before = state.version.get("device:dev1")
        events: list[object] = []
        svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

        svc.poll_device_info("dev1")
        _drain_until(
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
        release.set()
        _drain_until(
            lambda: svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED,  # type: ignore[union-attr]
            label="setup finished",
        )


@pytest.mark.parametrize(
    "kind",
    [OperationKind.DEVICE_CONNECT, OperationKind.DEVICE_DISCONNECT],
)
def test_poll_device_info_skips_non_setup_device_mutation(qapp, kind: OperationKind):
    """Connect/disconnect leases are not safe live-read windows, so poll skips
    them without submitting a worker."""
    gate = OperationGate(EventBus())
    svc, device = _make_svc(gate=gate)
    _connect(svc, _req())
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
        _drain_poll(qapp)

        device.get_info.assert_not_called()
    finally:
        gate.release(99)


def test_poll_device_info_late_result_skips_after_non_setup_mutation(qapp):
    """A delayed poll delivery must not bump State after the device enters a
    non-setup mutation."""
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())
    before = state.version.get("device:dev1")
    events: list[object] = []
    svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    device.get_info.return_value = FakeDeviceInfo(address="addr", value=8.0)
    deferred_bg = _DeferredPollBackground()
    svc._bg = deferred_bg  # white-box: only poll_device_info reads this field.

    svc.poll_device_info("dev1")
    assert deferred_bg.submitted is True
    assert state.version.get("device:dev1") == before

    state.set_device_status("dev1", DeviceStatus.DISCONNECTING)
    after_status_change = state.version.get("device:dev1")
    deferred_bg.deliver()

    assert state.version.get("device:dev1") == after_status_change
    assert events == []
    cached = state.get_device("dev1")
    assert cached is not None and getattr(cached.info, "value", None) == 0.0


def test_poll_device_info_swallows_read_failure(qapp):
    """A single failed read (timeout / driver boom) is logged and dropped so the
    poller keeps ticking — best-effort, not Fast-Fail."""
    svc, device = _make_svc()
    state = svc._state
    _connect(svc, _req())
    before = state.version.get("device:dev1")
    events: list[object] = []
    svc._bus.subscribe(DeviceChangedPayload, lambda p: events.append(p.name))

    device.get_info.side_effect = RuntimeError("read timeout")
    svc.poll_device_info("dev1")  # must not raise
    _drain_poll(qapp)

    # Failure swallowed: no bump, no emit, state untouched.
    assert state.version.get("device:dev1") == before
    assert events == []


# ---------------------------------------------------------------------------
# Phase C: concurrent per-device setup (gate is resource-aware; the in-flight
# state machine is keyed by device name)
# ---------------------------------------------------------------------------


def _make_multi_svc(
    gate: OperationGate | None = None,
) -> tuple[DeviceService, dict[str, MagicMock]]:
    """A DeviceService whose driver_factory mints a distinct mock driver per
    device name, so two devices can be set up concurrently with independent
    setup() / get_info() behaviour. Returns (svc, drivers-by-name)."""
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    drivers: dict[str, MagicMock] = {}

    def factory(_type: str, address: str) -> MagicMock:
        # The address carries the device name in these tests (one driver each).
        device = MagicMock()
        device.get_info.return_value = FakeDeviceInfo(address=address, value=0.0)
        drivers[address] = device
        return device

    bg = _bg()
    resolved_gate = gate or OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(resolved_gate, handles, progress, bg, bus)
    registry = FakeDeviceRegistry()
    svc = DeviceService(
        bus,
        State(MagicMock()),
        resolved_gate,
        bg,
        runner,
        handles,
        driver_factory=factory,  # type: ignore[arg-type]
        device_registry=registry,
    )
    return svc, drivers


def _connect_named(svc: DeviceService, name: str) -> None:
    connected: list[object] = []
    errors: list[str] = []
    svc.device_connected.connect(connected.append)
    svc.operation_failed.connect(lambda _n, error: errors.append(error))
    # address == name so _make_multi_svc keys the driver by name.
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name=name, address=name)
    )
    _drain_until(lambda: bool(connected or errors), label=f"connect {name}")
    assert not errors


def test_gate_allows_concurrent_setup_of_different_devices(qapp):
    """Resource-aware gate: a setup of one device does NOT block a setup of a
    different device (the core of Phase C)."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(gate=gate)
    _connect_named(svc, "devA")
    _connect_named(svc, "devB")

    # Hold devA mid-setup on a barrier so its lease is still active.
    release_a = threading.Event()
    drivers["devA"].setup.side_effect = lambda _info, stop_event=None: release_a.wait(
        2.0
    )

    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    # devB setup must start concurrently (no OperationConflictError).
    finished_b = QEventLoop()
    svc.setup_finished.connect(lambda n: finished_b.quit() if n == "devB" else None)
    token_b = svc.start_setup_device(
        SetupDeviceRequest(name="devB", info=FakeDeviceInfo(address="devB", value=2.0))
    )
    assert isinstance(token_b, int)

    # Both devices are mid-setup at the same instant.
    assert svc.get_device_snapshot("devA").status is DeviceStatus.SETTING_UP  # type: ignore[union-attr]
    assert svc.get_device_snapshot("devB").status is DeviceStatus.SETTING_UP  # type: ignore[union-attr]
    assert gate.is_device_mutating("devA")
    assert gate.is_device_mutating("devB")

    finished_b.exec()  # devB finishes while devA is still ramping
    assert svc.get_device_snapshot("devB").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]
    # devA is independent: still mid-setup.
    assert svc.get_device_snapshot("devA").status is DeviceStatus.SETTING_UP  # type: ignore[union-attr]

    # Let devA finish too.
    finished_a = QEventLoop()
    svc.setup_finished.connect(lambda n: finished_a.quit() if n == "devA" else None)
    release_a.set()
    finished_a.exec()
    assert svc.get_device_snapshot("devA").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]


def test_same_device_setup_still_blocked_while_in_flight(qapp):
    """A second setup of the SAME device while one is in flight is rejected
    (resource-aware conflict only relaxes across *distinct* devices). The device
    is SETTING_UP, so _require_connected_device fast-fails first; the gate's
    same-resource conflict is the deeper guard (covered separately in
    test_device_mutation_is_globally_exclusive_and_blocks_same_device_read)."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(gate=gate)
    _connect_named(svc, "devA")

    release = threading.Event()
    drivers["devA"].setup.side_effect = lambda _info, stop_event=None: release.wait(2.0)
    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    try:
        with pytest.raises(RuntimeError):
            svc.start_setup_device(
                SetupDeviceRequest(
                    name="devA", info=FakeDeviceInfo(address="devA", value=2.0)
                )
            )
        # The in-flight op is untouched: exactly one lease for devA.
        assert gate.is_device_mutating("devA")
    finally:
        release.set()
        loop = QEventLoop()
        svc.setup_finished.connect(lambda n: loop.quit() if n == "devA" else None)
        loop.exec()


def test_concurrent_setups_cancel_independently(qapp):
    """Cancelling one in-flight setup must not affect the other device's setup."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(gate=gate)
    _connect_named(svc, "devA")
    _connect_named(svc, "devB")

    # Both setups poll their stop_event and only return when set (so cancel is
    # observable) — but devB never gets cancelled, it finishes on a barrier.
    def cancellable(_info, stop_event=None):
        # Wait until cancelled (stop_event set) or a short timeout elapses.
        if stop_event is not None:
            stop_event.wait(2.0)

    drivers["devA"].setup.side_effect = cancellable
    release_b = threading.Event()
    drivers["devB"].setup.side_effect = lambda _info, stop_event=None: release_b.wait(
        2.0
    )

    cancelled: list[str] = []
    svc.setup_cancelled.connect(lambda n: cancelled.append(n))
    finished: list[str] = []
    svc.setup_finished.connect(lambda n: finished.append(n))

    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    svc.start_setup_device(
        SetupDeviceRequest(name="devB", info=FakeDeviceInfo(address="devB", value=2.0))
    )
    assert gate.is_device_mutating("devA")
    assert gate.is_device_mutating("devB")

    # Cancel only devA.
    cancel_loop = QEventLoop()
    svc.setup_cancelled.connect(lambda n: cancel_loop.quit() if n == "devA" else None)
    svc.cancel_device_operation("devA")
    cancel_loop.exec()

    assert cancelled == ["devA"]
    assert svc.get_device_snapshot("devA").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]
    # devB is untouched: still mid-setup, then finishes on its own.
    assert svc.get_device_snapshot("devB").status is DeviceStatus.SETTING_UP  # type: ignore[union-attr]
    assert "devB" not in cancelled

    finish_loop = QEventLoop()
    svc.setup_finished.connect(lambda n: finish_loop.quit() if n == "devB" else None)
    release_b.set()
    finish_loop.exec()
    assert "devB" in finished
    assert svc.get_device_snapshot("devB").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# DeviceRegistryPort injection: fake in-memory registry replaces the real
# GlobalDeviceManager singleton so these tests never touch it.
# ---------------------------------------------------------------------------


def _make_svc_with_fake_registry(
    driver: MagicMock | None = None,
    gate: OperationGate | None = None,
) -> tuple[DeviceService, MagicMock, FakeDeviceRegistry]:
    """Like ``_make_svc`` but injects a ``_FakeRegistry`` instead of the real singleton."""
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    bg = _bg()
    resolved_gate = gate or OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(resolved_gate, handles, progress, bg, bus)
    registry = FakeDeviceRegistry()
    svc = DeviceService(
        bus,
        State(MagicMock()),
        resolved_gate,
        bg,
        runner,
        handles,
        driver_factory=lambda _type, _address: device,  # type: ignore[arg-type]
        device_registry=registry,
    )
    return svc, device, registry


def test_registry_port_connect_registers_in_fake_not_global(qapp):
    """connect goes through the injected port; the real GlobalDeviceManager stays clean."""
    svc, device, registry = _make_svc_with_fake_registry()
    _connect(svc, _req())

    # Device registered in the fake registry.
    assert "dev1" in registry.get_all_devices()
    # Real singleton untouched.
    assert "dev1" not in GlobalDeviceManager.get_all_devices()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]


def test_registry_port_disconnect_drops_from_fake(qapp):
    """disconnect calls drop_device on the port; device.close() is still called."""
    svc, device, registry = _make_svc_with_fake_registry()
    _connect(svc, _req())
    assert "dev1" in registry.get_all_devices()

    drop_loop = QEventLoop()
    svc.device_disconnected.connect(lambda _r: drop_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    drop_loop.exec()

    device.close.assert_called_once_with()
    assert "dev1" not in registry.get_all_devices()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.MEMORY_ONLY  # type: ignore[union-attr]


def test_registry_port_get_info_reads_from_fake(qapp):
    """get_device_info reads through the port (fake.get_info → driver.get_info)."""
    svc, device, _registry = _make_svc_with_fake_registry()
    _connect(svc, _req())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=7.0)

    info = svc.get_device_info("dev1")

    assert info is not None
    assert getattr(info, "value", None) == 7.0
    device.get_info.assert_called()


def test_registry_port_connect_failure_rollback_still_correct(qapp):
    """_cleanup_failed_connection (now instance method) drops via port on failure."""
    svc, device, registry = _make_svc_with_fake_registry()
    device.get_info.side_effect = RuntimeError("boom")
    errors: list[str] = []
    loop = QEventLoop()
    svc.operation_failed.connect(
        lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_connect_device(_req())
    loop.exec()

    assert errors and "boom" in errors[0]
    # Fake registry must be clean after rollback (drop_device was called).
    assert "dev1" not in registry.get_all_devices()
    # Real singleton untouched throughout.
    assert "dev1" not in GlobalDeviceManager.get_all_devices()
    device.close.assert_called_once_with()


def test_registry_port_setup_reads_driver_from_fake(qapp):
    """start_setup_device fetches the driver from the port (get_device → fake), not the real singleton."""
    svc, device, registry = _make_svc_with_fake_registry()
    _connect(svc, _req())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=5.0)

    setup_loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: setup_loop.quit())
    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=5.0))
    )
    setup_loop.exec()

    device.setup.assert_called_once()
    assert svc.get_device_snapshot("dev1").info.value == 5.0  # type: ignore[union-attr]
    # The driver in the registry is still the mock (not replaced by setup).
    assert registry.get_device("dev1") is device


def test_concurrent_setup_failure_does_not_affect_other(qapp):
    """A failed setup of one device rolls back only that device; the other
    device's concurrent setup is unaffected."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(gate=gate)
    _connect_named(svc, "devA")
    _connect_named(svc, "devB")

    drivers["devA"].setup.side_effect = lambda _info, stop_event=None: (
        _ for _ in ()
    ).throw(RuntimeError("devA boom"))
    release_b = threading.Event()
    drivers["devB"].setup.side_effect = lambda _info, stop_event=None: release_b.wait(
        2.0
    )

    failed: list[tuple[str, str]] = []
    svc.setup_failed.connect(lambda n, e: failed.append((n, e)))

    svc.start_setup_device(
        SetupDeviceRequest(name="devB", info=FakeDeviceInfo(address="devB", value=2.0))
    )
    fail_loop = QEventLoop()
    svc.setup_failed.connect(lambda n, _e: fail_loop.quit() if n == "devA" else None)
    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    fail_loop.exec()

    assert failed and failed[0][0] == "devA" and "devA boom" in failed[0][1]
    # devA rolled back to CONNECTED (prior state), its lease released.
    assert svc.get_device_snapshot("devA").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]
    assert not gate.is_device_mutating("devA")
    # devB still mid-setup, lease intact.
    assert svc.get_device_snapshot("devB").status is DeviceStatus.SETTING_UP  # type: ignore[union-attr]
    assert gate.is_device_mutating("devB")

    finish_loop = QEventLoop()
    svc.setup_finished.connect(lambda n: finish_loop.quit() if n == "devB" else None)
    release_b.set()
    finish_loop.exec()
    assert svc.get_device_snapshot("devB").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]
