"""Tests for async DeviceService commands and snapshot boundaries."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import EventMeta, EventOrigin
from zcu_tools.gui.expected_error import ExpectedErrorCategory, FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationInfrastructure
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._completion_helpers import (
    on_device_connected,
    on_device_disconnected,
    on_device_operation_failed,
    on_setup_cancelled,
    on_setup_failed,
    on_setup_finished,
)
from tests.gui._qt_waits import qt_until
from tests.gui.session.services._device_fakes import FakeDeviceRegistry
from tests.gui.session.services._device_service_support import (
    connect_device,
    device_connect_request,
    make_device_service,
)
from tests.gui.session.services._device_waits import DeviceWorkers


def test_device_connect_returns_operation_handle(qapp, device_workers: DeviceWorkers):
    svc, _device = make_device_service(device_workers)
    connected: list[object] = []
    errors: list[str] = []
    on_device_connected(svc, connected.append)
    on_device_operation_failed(svc, lambda _name, error: errors.append(error))
    token = svc.start_connect_device(device_connect_request())
    qt_until(lambda: bool(connected or errors), label="device connect")
    assert not errors
    # connect now returns an operation token (parity with setup) for awaiting.
    assert isinstance(token, int)
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED


def test_device_setup_updates_value_and_disconnect_updates_snapshot(
    qapp, device_workers: DeviceWorkers
):
    svc, device = make_device_service(device_workers)
    connect_device(svc, device_connect_request())
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED

    # Setting an output value goes through setup (ramped/cancellable), not a
    # separate set_value: the post-setup get_info reflects the new value.
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=1.0)
    setup_loop = device_workers.completion()
    on_setup_finished(svc, lambda _name: setup_loop.quit())
    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=1.0))
    )
    setup_loop.exec()
    device.setup.assert_called_once()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert isinstance(snapshot.info, FakeDeviceInfo)
    assert snapshot.info.value == 1.0

    drop_loop = device_workers.completion()
    on_device_disconnected(svc, lambda _request: drop_loop.quit())
    disc_token = svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    assert isinstance(disc_token, int)  # disconnect also returns a handle
    drop_loop.exec()
    device.close.assert_called_once_with()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.MEMORY_ONLY


def test_device_setup_started_and_finished_keep_operation_origin(
    qapp, device_workers: DeviceWorkers
) -> None:
    bus = EventBus()
    svc, device = make_device_service(device_workers, bus=bus)
    connect_device(svc, device_connect_request())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=1.0)
    observed: list[tuple[object, EventMeta]] = []
    bus.subscribe_with_meta(
        DeviceSetupStartedPayload,
        lambda payload, meta: observed.append((payload, meta)),
    )
    bus.subscribe_with_meta(
        DeviceSetupFinishedPayload,
        lambda payload, meta: observed.append((payload, meta)),
    )
    loop = device_workers.completion()
    on_setup_finished(svc, lambda _name: loop.quit())

    with bus.origin(EventOrigin(kind="agent", client_id="client-a")):
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


def test_device_mutation_is_globally_exclusive_and_blocks_same_device_read(
    qapp, device_workers: DeviceWorkers
):
    gate = OperationGate(EventBus())
    svc, _device = make_device_service(device_workers, gate=gate)
    connect_device(svc, device_connect_request())
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


def test_cancel_missing_device_operation_is_failed_precondition(
    qapp, device_workers: DeviceWorkers
):
    svc, _device = make_device_service(device_workers)

    with pytest.raises(FailedPreconditionError, match="No operation") as exc_info:
        svc.cancel_device_operation("dev1")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""


def test_device_connect_failure_is_reported_without_live_registration(
    qapp, device_workers: DeviceWorkers
):
    svc, device = make_device_service(device_workers)
    device.get_info.side_effect = RuntimeError("cannot query")
    errors: list[str] = []
    loop = device_workers.completion(allow_failure=True)
    on_device_operation_failed(
        svc, lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_connect_device(device_connect_request())
    loop.exec()

    assert errors
    assert "cannot query" in errors[0]
    assert "dev1" not in svc.get_connected_devices()
    device.close.assert_called_once_with()
    assert svc.get_device_snapshot("dev1") is None


def test_setup_uses_gate_and_returns_to_connected_snapshot(
    qapp, device_workers: DeviceWorkers
):
    svc, device = make_device_service(device_workers)
    connect_device(svc, device_connect_request())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=2.0)
    loop = device_workers.completion()
    on_setup_finished(svc, lambda _name: loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )
    active = svc.get_active_device_setups()
    assert [s.device_name for s in active] == ["dev1"]
    loop.exec()

    device.setup.assert_called_once()
    assert svc.get_active_device_setups() == ()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED


def test_active_device_setups_enumerates_concurrent_setups(
    qapp, device_workers: DeviceWorkers
):
    """Phase C: two devices setting up concurrently both appear, sorted by name,
    and each in-flight op is reported with its kind (regression for the old
    single-valued getter that returned only one)."""
    svc, drivers = _make_multi_svc(device_workers)
    for name in ("alpha", "beta"):
        _connect_named(svc, name)
    blocks = {name: device_workers.setup() for name in drivers}
    finished: set[str] = set()
    on_setup_finished(svc, finished.add)
    for name, block in blocks.items():
        drivers[name].setup.side_effect = block
        svc.start_setup_device(
            SetupDeviceRequest(name=name, info=FakeDeviceInfo(address=name, value=1.0))
        )
    for block in blocks.values():
        qt_until(block.entered.is_set, label="setup worker entered")

    setups = svc.get_active_device_setups()
    assert [s.device_name for s in setups] == ["alpha", "beta"]
    ops = svc.get_active_device_operations()
    assert [o.device_name for o in ops] == ["alpha", "beta"]
    assert all(o.kind is OperationKind.DEVICE_SETUP for o in ops)
    assert all(o.snapshot.status is DeviceStatus.SETTING_UP for o in ops)
    for block in blocks.values():
        block.release.set()
    qt_until(lambda: len(finished) == 2, label="both setups finished")
    assert finished == {"alpha", "beta"}


@pytest.mark.parametrize("failure", [False, True], ids=["finished", "failed"])
def test_wait_setup_done_wakes_from_worker_thread_without_deadlock(
    qapp, device_workers: DeviceWorkers, failure
):
    """A pending off-main waiter wakes from a real Qt owner terminal delivery."""
    from concurrent.futures import Future

    from zcu_tools.gui.session.operation_handles import AwaitResult

    handles = OperationHandles()
    svc, device = make_device_service(device_workers, handles=handles)
    connect_device(svc, device_connect_request())
    block = device_workers.setup(RuntimeError("hardware boom") if failure else None)
    device.setup.side_effect = block
    token = svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )
    qt_until(block.entered.is_set, label="setup worker entered")
    pending: Future[AwaitResult | None] = Future()
    result: Future[AwaitResult | None] = Future()
    done = threading.Event()

    def waiter() -> None:
        try:
            pending.set_result(handles.await_outcome(token, timeout=0))
            # Releasing the driver and pumping Qt will settle this known token.
            result.set_result(handles.await_outcome(token, timeout=30))
        except BaseException as exc:
            logging.exception("Device waiter failed")
            result.set_exception(exc)
            if not pending.done():
                pending.set_exception(exc)
        finally:
            done.set()

    wt = threading.Thread(target=waiter)
    wt.start()
    try:
        # The started waiter publishes its zero-timeout observation before blocking.
        observed = pending.result(timeout=30)
        assert observed is not None and observed.reason == "timeout"
        assert handles.live_count() == 1  # timeout must not cancel the operation
        # No public blocking-arrival hook: this window only adds regression sensitivity,
        # never proves arrival; load cannot fail a correct still-pending waiter.
        assert not done.wait(0.02)
        block.release.set()
        qt_until(done.is_set, label="off-main waiter terminal")
        outcome = result.result()  # rethrow worker errors on the caller
        assert outcome is not None and outcome.reason == "completed"
        assert outcome.outcome is not None
        assert outcome.outcome.status == ("failed" if failure else "finished")
        if failure:
            assert "hardware boom" in str(outcome.outcome.error)
    finally:
        block.release.set()
        try:
            qt_until(done.is_set, label="waiter cleanup after driver release")
        finally:
            # The released driver and owner delivery terminate the waiter.
            wt.join(timeout=30)
        assert not wt.is_alive()
        result.result()


def test_disconnect_close_failure_retains_connected_device_and_releases_gate(
    qapp, device_workers: DeviceWorkers
):
    svc, device = make_device_service(device_workers)
    connect_device(svc, device_connect_request())
    device.close.side_effect = OSError("close failed")
    loop = device_workers.completion(allow_failure=True)
    errors: list[str] = []
    on_device_operation_failed(
        svc, lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    loop.exec()

    assert "close failed" in errors[0]
    assert "dev1" in svc.get_connected_devices()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED
    device.close.side_effect = None
    retry_loop = device_workers.completion()
    on_device_disconnected(svc, lambda _request: retry_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    retry_loop.exec()


def test_failing_device_changed_subscriber_does_not_abort_connect(
    qapp, device_workers: DeviceWorkers
):
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    bus = EventBus()
    bg_svc = device_workers.background()
    resolved_gate = OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(resolved_gate, handles, progress, bg_svc, bus)
    svc = DeviceService(
        bus,
        State(MagicMock()),
        OperationInfrastructure(
            gate=resolved_gate,
            background=bg_svc,
            runner=runner,
            handles=handles,
        ),
        driver_factory=lambda type_name, address: MagicMock(),
        device_registry=FakeDeviceRegistry(),
    )
    # A DEVICE_CHANGED subscriber raising (e.g. a View redraw bug) is swallowed +
    # logged by the EventBus; it must NOT abort the connect or roll back the
    # optimistic device state (one bad subscriber must not break the publisher).
    bus.subscribe(
        DeviceChangedPayload,
        MagicMock(side_effect=RuntimeError("render failed")),
    )

    svc.start_connect_device(
        device_connect_request()
    )  # no raise — the subscriber failure is swallowed

    # The optimistic device state survives (the operation proceeds; the lease is
    # released at its real terminal, not on the subscriber's failure) — previously
    # the re-raising bus rolled this back to None.
    assert svc.get_device_snapshot("dev1") is not None


def test_connect_writes_device_state_to_state_with_remember(
    qapp, device_workers: DeviceWorkers
):
    state = State(MagicMock())
    svc, _device = make_device_service(device_workers, state=state)
    connect_device(
        svc, ConnectDeviceRequest("FakeDevice", "dev1", "addr", remember=False)
    )
    dev = state.get_device("dev1")
    assert dev is not None
    assert dev.status is DeviceStatus.CONNECTED
    # `remember` is now persistent device state, not a transient request attribute.
    assert dev.remember is False


# ---------------------------------------------------------------------------
# Phase C: concurrent per-device setup (gate is resource-aware; the in-flight
# state machine is keyed by device name)
# ---------------------------------------------------------------------------


def _make_multi_svc(
    workers: DeviceWorkers,
    gate: OperationGate | None = None,
) -> tuple[DeviceService, dict[str, MagicMock]]:
    """A DeviceService whose driver_factory mints a distinct mock driver per
    device name, so two devices can be set up concurrently with independent
    setup() / get_info() behaviour. Returns (svc, drivers-by-name)."""
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    drivers: dict[str, MagicMock] = {}

    def factory(type_name: str, address: str) -> MagicMock:
        # The address carries the device name in these tests (one driver each).
        device = MagicMock()
        device.get_info.return_value = FakeDeviceInfo(address=address, value=0.0)
        drivers[address] = device
        return device

    bg = workers.background()
    resolved_gate = gate or OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    workers.watch(bus)
    runner = OperationRunner(resolved_gate, handles, progress, bg, bus)
    registry = FakeDeviceRegistry()
    svc = DeviceService(
        bus,
        State(MagicMock()),
        OperationInfrastructure(
            gate=resolved_gate,
            background=bg,
            runner=runner,
            handles=handles,
        ),
        driver_factory=factory,
        device_registry=registry,
    )
    return svc, drivers


def _connect_named(svc: DeviceService, name: str) -> None:
    connected: list[object] = []
    errors: list[str] = []
    on_device_connected(svc, connected.append)
    on_device_operation_failed(svc, lambda _n, error: errors.append(error))
    # address == name so _make_multi_svc keys the driver by name.
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name=name, address=name)
    )
    qt_until(lambda: bool(connected or errors), label=f"connect {name}")
    assert not errors


def test_gate_allows_concurrent_setup_of_different_devices(
    qapp, device_workers: DeviceWorkers
):
    """Resource-aware gate: a setup of one device does NOT block a setup of a
    different device (the core of Phase C)."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(device_workers, gate=gate)
    _connect_named(svc, "devA")
    _connect_named(svc, "devB")

    # Hold devA mid-setup on a barrier so its lease is still active.
    block_a = device_workers.setup()
    drivers["devA"].setup.side_effect = block_a

    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    qt_until(block_a.entered.is_set, label="devA driver entered")
    # devB setup must start concurrently (no OperationConflictError).
    finished_b = device_workers.completion()
    on_setup_finished(svc, lambda n: finished_b.quit() if n == "devB" else None)
    token_b = svc.start_setup_device(
        SetupDeviceRequest(name="devB", info=FakeDeviceInfo(address="devB", value=2.0))
    )
    assert isinstance(token_b, int)

    # Both devices are mid-setup at the same instant.
    snapshot = svc.get_device_snapshot("devA")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.SETTING_UP
    snapshot = svc.get_device_snapshot("devB")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.SETTING_UP
    assert gate.is_device_mutating("devA")
    assert gate.is_device_mutating("devB")

    finished_b.exec()  # devB finishes while devA is still ramping
    snapshot = svc.get_device_snapshot("devB")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED
    # devA is independent: still mid-setup.
    snapshot = svc.get_device_snapshot("devA")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.SETTING_UP

    # Let devA finish too.
    finished_a = device_workers.completion()
    on_setup_finished(svc, lambda n: finished_a.quit() if n == "devA" else None)
    block_a.release.set()
    finished_a.exec()
    snapshot = svc.get_device_snapshot("devA")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED


def test_same_device_setup_still_blocked_while_in_flight(
    qapp, device_workers: DeviceWorkers
):
    """A second setup of the SAME device while one is in flight is rejected
    (resource-aware conflict only relaxes across *distinct* devices). The device
    is SETTING_UP, so _require_connected_device fast-fails first; the gate's
    same-resource conflict is the deeper guard (covered separately in
    test_device_mutation_is_globally_exclusive_and_blocks_same_device_read)."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(device_workers, gate=gate)
    _connect_named(svc, "devA")

    block = device_workers.setup()
    drivers["devA"].setup.side_effect = block
    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    qt_until(block.entered.is_set, label="devA driver entered")
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
        block.release.set()
        loop = device_workers.completion()
        on_setup_finished(svc, lambda n: loop.quit() if n == "devA" else None)
        loop.exec()


def test_concurrent_setups_cancel_independently(qapp, device_workers: DeviceWorkers):
    """Cancelling one in-flight setup must not affect the other device's setup."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(device_workers, gate=gate)
    _connect_named(svc, "devA")
    _connect_named(svc, "devB")

    block_a, block_b = device_workers.setup(), device_workers.setup()
    drivers["devA"].setup.side_effect = block_a
    drivers["devB"].setup.side_effect = block_b

    cancelled: list[str] = []
    on_setup_cancelled(svc, lambda n: cancelled.append(n))
    finished: list[str] = []
    on_setup_finished(svc, lambda n: finished.append(n))

    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    svc.start_setup_device(
        SetupDeviceRequest(name="devB", info=FakeDeviceInfo(address="devB", value=2.0))
    )
    assert gate.is_device_mutating("devA")
    assert gate.is_device_mutating("devB")

    qt_until(block_a.entered.is_set, label="devA driver entered")
    qt_until(block_b.entered.is_set, label="devB driver entered")
    # Cancel only devA.
    cancel_loop = device_workers.completion()
    on_setup_cancelled(svc, lambda n: cancel_loop.quit() if n == "devA" else None)
    svc.cancel_device_operation("devA")
    assert block_a.stop_event is not None and block_a.stop_event.is_set()
    assert block_b.stop_event is not None and not block_b.stop_event.is_set()
    block_a.release.set()
    cancel_loop.exec()

    assert cancelled == ["devA"]
    snapshot = svc.get_device_snapshot("devA")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED
    # devB is untouched: still mid-setup, then finishes on its own.
    snapshot = svc.get_device_snapshot("devB")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.SETTING_UP
    assert "devB" not in cancelled

    finish_loop = device_workers.completion()
    on_setup_finished(svc, lambda n: finish_loop.quit() if n == "devB" else None)
    block_b.release.set()
    finish_loop.exec()
    assert "devB" in finished
    snapshot = svc.get_device_snapshot("devB")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED


# ---------------------------------------------------------------------------
# DeviceRegistryPort injection: fake in-memory registry replaces the real
# DeviceManager instance with no driver ownership outside the injected port.
# ---------------------------------------------------------------------------


def _make_svc_with_fake_registry(
    workers: DeviceWorkers,
    driver: MagicMock | None = None,
    gate: OperationGate | None = None,
) -> tuple[DeviceService, MagicMock, FakeDeviceRegistry]:
    """Like ``make_device_service`` but injects a ``FakeDeviceRegistry``."""
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner

    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    bg = workers.background()
    resolved_gate = gate or OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    workers.watch(bus)
    runner = OperationRunner(resolved_gate, handles, progress, bg, bus)
    registry = FakeDeviceRegistry()
    svc = DeviceService(
        bus,
        State(MagicMock()),
        OperationInfrastructure(
            gate=resolved_gate,
            background=bg,
            runner=runner,
            handles=handles,
        ),
        driver_factory=lambda type_name, address: device,
        device_registry=registry,
    )
    return svc, device, registry


def test_registry_port_connect_registers_in_fake_not_global(
    qapp, device_workers: DeviceWorkers
):
    """Connect registers the driver through the injected registry port."""
    svc, _device, registry = _make_svc_with_fake_registry(device_workers)
    connect_device(svc, device_connect_request())

    # Device registered in the fake registry.
    assert "dev1" in registry.get_all_devices()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED


def test_registry_port_disconnect_drops_from_fake(qapp, device_workers: DeviceWorkers):
    """disconnect calls drop_device on the port; device.close() is still called."""
    svc, device, registry = _make_svc_with_fake_registry(device_workers)
    connect_device(svc, device_connect_request())
    assert "dev1" in registry.get_all_devices()

    drop_loop = device_workers.completion()
    on_device_disconnected(svc, lambda _r: drop_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    drop_loop.exec()

    device.close.assert_called_once_with()
    assert "dev1" not in registry.get_all_devices()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.MEMORY_ONLY


def test_registry_port_get_info_reads_from_fake(qapp, device_workers: DeviceWorkers):
    """get_device_info reads through the port (fake.get_info → driver.get_info)."""
    svc, device, _registry = _make_svc_with_fake_registry(device_workers)
    connect_device(svc, device_connect_request())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=7.0)

    info = svc.get_device_info("dev1")

    assert info is not None
    assert getattr(info, "value", None) == 7.0
    device.get_info.assert_called()


def test_registry_port_connect_failure_rollback_still_correct(
    qapp, device_workers: DeviceWorkers
):
    """_cleanup_failed_connection (now instance method) drops via port on failure."""
    svc, device, registry = _make_svc_with_fake_registry(device_workers)
    device.get_info.side_effect = RuntimeError("boom")
    errors: list[str] = []
    loop = device_workers.completion(allow_failure=True)
    on_device_operation_failed(
        svc, lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_connect_device(device_connect_request())
    loop.exec()

    assert errors and "boom" in errors[0]
    # Fake registry must be clean after rollback (drop_device was called).
    assert "dev1" not in registry.get_all_devices()
    device.close.assert_called_once_with()


def test_registry_port_setup_reads_driver_from_fake(
    qapp, device_workers: DeviceWorkers
):
    """Setup fetches the driver through the injected registry port."""
    svc, device, registry = _make_svc_with_fake_registry(device_workers)
    connect_device(svc, device_connect_request())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=5.0)

    setup_loop = device_workers.completion()
    on_setup_finished(svc, lambda _name: setup_loop.quit())
    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=5.0))
    )
    setup_loop.exec()

    device.setup.assert_called_once()
    snapshot = svc.get_device_snapshot("dev1")
    assert snapshot is not None
    assert isinstance(snapshot.info, FakeDeviceInfo)
    assert snapshot.info.value == 5.0
    # The driver in the registry is still the mock (not replaced by setup).
    assert registry.get_device("dev1") is device


def test_concurrent_setup_failure_does_not_affect_other(
    qapp, device_workers: DeviceWorkers
):
    """A failed setup of one device rolls back only that device; the other
    device's concurrent setup is unaffected."""
    gate = OperationGate(EventBus())
    svc, drivers = _make_multi_svc(device_workers, gate=gate)
    _connect_named(svc, "devA")
    _connect_named(svc, "devB")

    drivers["devA"].setup.side_effect = lambda _info, stop_event=None: (
        _ for _ in ()
    ).throw(RuntimeError("devA boom"))
    block_b = device_workers.setup()
    drivers["devB"].setup.side_effect = block_b

    failed: list[tuple[str, str]] = []
    on_setup_failed(svc, lambda n, e: failed.append((n, e)))

    svc.start_setup_device(
        SetupDeviceRequest(name="devB", info=FakeDeviceInfo(address="devB", value=2.0))
    )
    qt_until(block_b.entered.is_set, label="devB driver entered")
    fail_loop = device_workers.completion(allow_failure=True)
    on_setup_failed(svc, lambda n, _e: fail_loop.quit() if n == "devA" else None)
    svc.start_setup_device(
        SetupDeviceRequest(name="devA", info=FakeDeviceInfo(address="devA", value=1.0))
    )
    fail_loop.exec()

    assert failed and failed[0][0] == "devA" and "devA boom" in failed[0][1]
    # devA rolled back to CONNECTED (prior state), its lease released.
    snapshot = svc.get_device_snapshot("devA")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED
    assert not gate.is_device_mutating("devA")
    # devB still mid-setup, lease intact.
    snapshot = svc.get_device_snapshot("devB")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.SETTING_UP
    assert gate.is_device_mutating("devB")

    finish_loop = device_workers.completion()
    on_setup_finished(svc, lambda n: finish_loop.quit() if n == "devB" else None)
    block_b.release.set()
    finish_loop.exec()
    snapshot = svc.get_device_snapshot("devB")
    assert snapshot is not None
    assert snapshot.status is DeviceStatus.CONNECTED
