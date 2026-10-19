"""Tests for SoCConnectionService (stage 3b).

Uses a synchronous _FakeBg that drives on_done/on_error inline so tests can
step through the runner lifecycle without a real QEventLoop event-pump for most
paths, and use qt_until + real BackgroundRunner only for end-to-end
async signal tests (offline remote success / failure) where the bg submit matters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.services.operation_gate import (
    OperationGate,
)
from zcu_tools.gui.app.measure.services.operation_gate import (
    OperationKind as MeasureOpKind,
)
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._completion_helpers import on_connection_failed
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until


@pytest.fixture
def real_bg(qapp):
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    background = BackgroundRunner()
    try:
        yield background
    finally:
        # The finite offline proxy returns or raises; drain workers and queued delivery.
        assert background.quiesce(timeout_ms=30_000)


@pytest.fixture(autouse=True)
def offline_remote(monkeypatch):
    import zcu_tools.qick_remote as remote
    from zcu_tools.experiment.simulation import white_noise

    def proxy(ip, port):
        env = white_noise()
        return env.soc, env.soccfg

    monkeypatch.setattr(remote, "make_soc_proxy", proxy)


# ---------------------------------------------------------------------------
# Fakes (from test_operation_runner.py pattern)
# ---------------------------------------------------------------------------


class _FakeBg:
    """Synchronous background executor stub.

    submit() captures callbacks; call deliver_result() or deliver_error() to
    drive the on_done / on_error path.  This simulates the runner bg path without
    a real thread, so on_terminal runs synchronously on the test thread.
    """

    def __init__(self) -> None:
        self._work: Callable[[], Any] | None = None
        self._on_done: Callable[[Any], None] | None = None
        self._on_error: Callable[[Exception], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._work = work
        self._on_done = on_done
        self._on_error = on_error

    def deliver_result(self) -> None:
        """Execute the captured work thunk and pass its return value to on_done."""
        assert self._work is not None and self._on_done is not None
        result = self._work()
        self._on_done(result)

    def deliver_error(self, exc: Exception) -> None:
        assert self._on_error is not None
        self._on_error(exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state() -> State:
    return State(SessionEnv(parameters=None, soc=None, soccfg=None, result_dir=""))


def _make_svc(
    gate: OperationGate | None = None,
    bg: _FakeBg | None = None,
    state: State | None = None,
    bus: EventBus | None = None,
) -> tuple[SoCConnectionService, _FakeBg, OperationHandles]:
    """Build SoCConnectionService with a synchronous fake bg executor."""
    state = state or _make_state()
    bus = bus or EventBus()
    real_gate = gate or OperationGate(bus)
    handles = OperationHandles()
    fake_bg = bg or _FakeBg()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(real_gate, handles, progress, fake_bg, bus)
    svc = SoCConnectionService(state, bus, real_gate, handles, runner)
    return svc, fake_bg, handles


# ---------------------------------------------------------------------------
# Mock connect — end-to-end via QEventLoop (real bg path for signal delivery)
# ---------------------------------------------------------------------------


def test_start_connect_mock_emits_finished_and_updates_context(qapp, real_bg):
    """Offline remote connect delivers completion and sets soc in state."""
    from zcu_tools.gui.session.operation_runner import OperationRunner

    state = _make_state()
    bus = EventBus()
    gate = OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, real_bg, bus)
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    token = svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    # The finite offline proxy returns, then the owner publishes this handle's outcome.
    qt_until(lambda: handles.poll(token) is not None, label="SoC connection outcome")

    assert svc.has_soc()
    assert state.session_env.soccfg is not None


def test_connect_bumps_soc_not_context_version(qapp, real_bg):
    """Connect must bump soc version only; context version must stay unchanged."""
    state = _make_state()
    bus = EventBus()
    gate = OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, real_bg, bus)
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    ctx_before = state.version.get("context")
    soc_before = state.version.get("soc")

    token = svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    # The finite offline proxy returns, then the owner publishes this handle's outcome.
    qt_until(lambda: handles.poll(token) is not None, label="SoC connection outcome")

    assert state.version.get("soc") == soc_before + 1
    assert state.version.get("context") == ctx_before


# ---------------------------------------------------------------------------
# Synchronous connect (the soc.connect wire path) — connect_sync
# ---------------------------------------------------------------------------


def test_connect_sync_mock_sets_soc_and_emits_payload(qapp):
    """connect_sync runs inline (no bg), sets the soc, and emits SocChangedPayload
    synchronously (the hook the MockFluxProvisioner rides). It returns the handles
    directly and releases the SOC_CONNECT lease."""
    gate = OperationGate(EventBus())
    state = _make_state()
    bus = EventBus()
    svc, _bg, _handles = _make_svc(gate=gate, state=state, bus=bus)

    payloads: list[SocChangedPayload] = []
    bus.subscribe(SocChangedPayload, lambda p: payloads.append(p))

    soc_before = state.version.get("soc")
    soc, soccfg = svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.1", port=8888))

    assert soc is not None and soccfg is not None
    assert svc.has_soc()
    assert state.session_env.soc is soc
    # The shared _apply_connection side effects fired synchronously.
    assert state.version.get("soc") == soc_before + 1
    assert len(payloads) == 1
    assert payloads[0].is_mock is False
    # Lease released, no lingering active token.
    assert not gate.has_active(OperationKind.SOC_CONNECT)
    assert not svc.is_connect_active()


def test_successful_endpoint_changes_and_failed_reconnect_preserves_it(
    qapp, monkeypatch
):
    import zcu_tools.qick_remote as remote
    from zcu_tools.experiment.simulation import white_noise

    svc, _bg, _handles = _make_svc()
    assert not svc.has_soc()

    def proxy(ip: str, port: int):
        assert (ip, port) == ("192.0.2.1", 8888)
        env = white_noise()
        return env.soc, env.soccfg

    monkeypatch.setattr(remote, "make_soc_proxy", proxy)
    svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    assert not svc.is_mock_soc()
    assert svc.connected_endpoint() == {"address": "192.0.2.1", "port": 8888}

    def fail(ip: str, port: int):
        raise ConnectionRefusedError("offline")

    monkeypatch.setattr(remote, "make_soc_proxy", fail)
    with pytest.raises(ConnectionRefusedError, match="offline"):
        svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.2", port=8888))
    assert svc.has_soc()
    assert not svc.is_mock_soc()
    assert svc.connected_endpoint() == {"address": "192.0.2.1", "port": 8888}


def test_gui_async_remote_connect_updates_the_successful_endpoint(qapp, monkeypatch):
    import zcu_tools.qick_remote as remote
    from zcu_tools.experiment.simulation import white_noise

    def proxy(ip: str, port: int):
        assert (ip, port) == ("192.0.2.3", 8000)
        env = white_noise()
        return env.soc, env.soccfg

    monkeypatch.setattr(remote, "make_soc_proxy", proxy)
    svc, background, _handles = _make_svc()
    svc.start_connect(ConnectRemoteRequest(ip="192.0.2.3", port=8000))
    assert svc.connected_endpoint() == {"address": None, "port": None}
    background.deliver_result()
    assert svc.has_soc()
    assert svc.connected_endpoint() == {"address": "192.0.2.3", "port": 8000}


def test_runtime_soc_replacement_and_uninstall_forward_board_identity(monkeypatch):
    """Board identity belongs to the installed SoC, not its acquisition path."""
    import zcu_tools.qick_remote as remote

    class Board:
        def __init__(self, is_mock: bool) -> None:
            self.is_mock = is_mock

        def get_cfg(self) -> dict[str, object]:
            return {}

    state = _make_state()
    bus = EventBus()
    svc, _background, _handles = _make_svc(state=state, bus=bus)
    facts: list[SocChangedPayload] = []
    bus.subscribe(SocChangedPayload, facts.append)
    remote_board = Board(is_mock=True)
    config = MagicMock()
    monkeypatch.setattr(remote, "make_soc_proxy", lambda *_: (remote_board, config))

    svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    assert svc.is_mock_soc() is True
    assert facts[-1].is_mock is True

    context_version = state.version.get("context")
    soc_version = state.version.get("soc")
    local_board = Board(is_mock=False)
    svc.install_soc(local_board, config)
    assert state.session_env.soc is local_board
    assert svc.get_soccfg() is config
    assert svc.is_mock_soc() is False
    assert svc.connected_endpoint() == {"address": None, "port": None}
    assert state.version.get("soc") == soc_version + 1
    assert state.version.get("context") == context_version
    assert facts[-1].soc is local_board
    assert facts[-1].is_mock is False

    svc.uninstall_soc()
    assert svc.has_soc() is False
    assert svc.get_soccfg() is None
    assert svc.is_mock_soc() is False
    assert state.version.get("soc") == soc_version + 2
    assert state.version.get("context") == context_version
    assert facts[-1].soc is None
    assert facts[-1].soccfg is None
    assert facts[-1].is_mock is False

    # Uninstalling does not permanently lock the connection owner.
    svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.2", port=8888))
    assert svc.has_soc() is True
    assert svc.is_mock_soc() is True


@pytest.mark.parametrize("busy", ["run", "connect"])
def test_runtime_soc_changes_reject_busy_gate_without_losing_installed_pair(busy):
    state = _make_state()
    bus = EventBus()
    gate = OperationGate(bus)
    svc, bg, _handles = _make_svc(gate=gate, state=state, bus=bus)
    svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    before = state.session_env
    endpoint = svc.connected_endpoint()
    version = state.version.get("soc")
    facts: list[SocChangedPayload] = []
    bus.subscribe(SocChangedPayload, facts.append)
    if busy == "connect":
        svc.start_connect(ConnectRemoteRequest(ip="192.0.2.2", port=8888))
    else:
        gate.register(999, MeasureOpKind.RUN, owner_id="tab", origin_kind="user", note="running")
    try:
        with pytest.raises(OperationConflictError):
            svc.install_soc(MagicMock(), MagicMock())
        with pytest.raises(OperationConflictError):
            svc.uninstall_soc()
        assert state.session_env is before
        assert svc.connected_endpoint() == endpoint
        assert state.version.get("soc") == version
        assert facts == []
    finally:
        if busy == "connect":
            bg.deliver_result()
        else:
            gate.release(999)


def test_connect_sync_rejects_concurrent_calls(qapp):
    """connect_sync holds the same SOC_CONNECT lease as the async path, so a
    concurrent connect (or the GUI button) fast-fails."""
    gate = OperationGate(EventBus())
    svc, _bg, _handles = _make_svc(gate=gate)
    gate.register(
        1,
        OperationKind.SOC_CONNECT,
        owner_id="existing",
        origin_kind="user",
        note="existing connect",
    )
    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        svc.connect_sync(ConnectRemoteRequest(ip="192.0.2.1", port=8888))


def test_connect_sync_remote_failure_releases_lease_and_raises(qapp, monkeypatch):
    """A failed remote connect re-raises and still releases the lease (finally)."""
    gate = OperationGate(EventBus())
    svc, _bg, _handles = _make_svc(gate=gate)

    import zcu_tools.qick_remote as remote

    def fail(ip: str, port: int) -> None:
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(remote, "make_soc_proxy", fail, raising=False)

    with pytest.raises(ConnectionRefusedError, match="nope"):
        svc.connect_sync(ConnectRemoteRequest(ip="127.0.0.1", port=7000))

    assert not gate.has_active(OperationKind.SOC_CONNECT)
    assert not svc.is_connect_active()
    assert not svc.has_soc()


def test_connect_sync_rejects_unsupported_request(qapp):
    svc, _bg, _handles = _make_svc()

    class Other:
        pass

    with pytest.raises(TypeError, match="Unsupported connect request"):
        svc.connect_sync(Other())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reject concurrent / conflicting ops — synchronous gate check
# ---------------------------------------------------------------------------


def test_start_connect_rejects_concurrent_calls(qapp):
    gate = OperationGate(EventBus())
    svc, _bg, _handles = _make_svc(gate=gate)
    gate.register(
        1,
        OperationKind.SOC_CONNECT,
        owner_id="existing",
        origin_kind="user",
        note="existing connect",
    )
    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))


def test_start_connect_remote_unsupported_request_raises(qapp):
    svc, _bg, _handles = _make_svc()

    class Other:
        pass

    with pytest.raises(TypeError, match="Unsupported connect request"):
        svc.start_connect(Other())  # type: ignore[arg-type]


def test_start_connect_rejected_while_run_active(qapp):
    gate = OperationGate(EventBus())
    svc, _bg, _handles = _make_svc(gate=gate)
    gate.register(
        1,
        MeasureOpKind.RUN,
        owner_id="tab",
        origin_kind="user",
        note="existing run",
    )

    with pytest.raises(OperationConflictError, match="run is active"):
        svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))


# ---------------------------------------------------------------------------
# Remote failure — end-to-end via QEventLoop
# ---------------------------------------------------------------------------


def test_start_connect_remote_failure_emits_failed(qapp, monkeypatch, real_bg):
    """Remote connect failure: connection_failed emitted with 'nope' in message."""
    state = _make_state()
    bus = EventBus()
    gate = OperationGate(EventBus())
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, real_bg, bus)
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    import zcu_tools.qick_remote as remote

    def fail(ip: str, port: int) -> None:
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(remote, "make_soc_proxy", fail, raising=False)

    errors: list[str] = []
    on_connection_failed(svc, errors.append)

    token = svc.start_connect(ConnectRemoteRequest(ip="127.0.0.1", port=7000))
    # The failing proxy raises immediately; the owner settles the accepted operation.
    qt_until(lambda: handles.poll(token) is not None, label="SoC connection failure")

    assert errors
    assert "nope" in errors[0]
    assert not svc.is_connect_active()


# ---------------------------------------------------------------------------
# Subscriber failure still releases lease — synchronous fake bg
# ---------------------------------------------------------------------------


def test_soc_changed_subscriber_failure_releases_connection_lease(qapp):
    """A SOC_CHANGED subscriber raising must NOT propagate; lease is released.

    With the runner path, on_terminal runs on the main thread (bg marshal); the
    EventBus swallows subscriber exceptions. This test drives the full runner path
    via _FakeBg.deliver_result() so on_terminal executes synchronously, then
    asserts gate.has_active is False.
    """
    gate = OperationGate(EventBus())
    state = _make_state()
    bus = EventBus()
    bus.subscribe(
        SocChangedPayload, MagicMock(side_effect=RuntimeError("render failed"))
    )
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    fake_bg = _FakeBg()
    runner = OperationRunner(gate, handles, progress, fake_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    # start_connect registers the exclusion lease and submits to fake_bg
    svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    assert gate.has_active(OperationKind.SOC_CONNECT)

    # Drive the synchronous bg path: work() runs make_mock_soc, on_terminal is called
    # on the same thread without a real Qt marshal. Subscriber failure is swallowed.
    fake_bg.deliver_result()

    assert not gate.has_active(OperationKind.SOC_CONNECT)


# ---------------------------------------------------------------------------
# cancel_hook=None equivalence (§B.3) — connect is not interrupted by cancel
# ---------------------------------------------------------------------------


def test_cancel_token_does_not_interrupt_connect(qapp):
    """handles.cancel(token) on a connect token is a no-op: op runs to completion.

    cancel_hook=None means the Stop event is queued but the work thunk is not
    interrupted (§B.3). After deliver_result the lease is released normally.
    """
    gate = OperationGate(EventBus())
    svc, fake_bg, handles = _make_svc(gate=gate)

    token = svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    assert gate.has_active(OperationKind.SOC_CONNECT)

    # cancel does NOT raise, does NOT interrupt the in-flight work
    handles.cancel(token)

    # Op runs to completion (deliver_result drives on_terminal)
    fake_bg.deliver_result()

    # Lease is released — connect finished normally despite the cancel call
    assert not gate.has_active(OperationKind.SOC_CONNECT)
    assert not svc.is_connect_active()


def test_cancel_all_does_not_interrupt_connect(qapp):
    """handles.cancel_all() on a connect-only handles is also a no-op (§B.3)."""
    gate = OperationGate(EventBus())
    svc, fake_bg, handles = _make_svc(gate=gate)

    svc.start_connect(ConnectRemoteRequest(ip="192.0.2.1", port=8888))
    assert gate.has_active(OperationKind.SOC_CONNECT)

    handles.cancel_all()

    # Op still runs to completion
    fake_bg.deliver_result()

    assert not gate.has_active(OperationKind.SOC_CONNECT)
    assert not svc.is_connect_active()
