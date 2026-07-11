"""Tests for SoCConnectionService (stage 3b).

Uses a synchronous _FakeBg that drives on_done/on_error inline so tests can
step through the runner lifecycle without a real QEventLoop event-pump for most
paths, and fall back to QEventLoop + real BackgroundRunner only for end-to-end
async signal tests (mock / remote failure) where the bg submit matters.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationGate,
)
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationKind as MeasureOpKind,
)
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind
from zcu_tools.gui.session.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

from ._progress_fakes import DirectProgressTransport

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


class _FakeProgress:
    def make_factory(self, operation_id: int, owner_id: str) -> None:
        return None

    def discard_operation(self, operation_id: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state() -> State:
    return State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )


def _make_svc(
    gate: OperationGate | None = None,
    bg: _FakeBg | None = None,
) -> tuple[SoCConnectionService, _FakeBg, OperationHandles]:
    """Build SoCConnectionService with a synchronous fake bg executor."""
    state = _make_state()
    bus = EventBus()
    real_gate = gate or OperationGate()
    handles = OperationHandles()
    fake_bg = bg or _FakeBg()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(real_gate, handles, progress, fake_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, real_gate, handles, runner)
    return svc, fake_bg, handles


# ---------------------------------------------------------------------------
# Mock connect — end-to-end via QEventLoop (real bg path for signal delivery)
# ---------------------------------------------------------------------------


def test_start_connect_mock_emits_finished_and_updates_context(qapp):
    """Mock connect triggers connection_finished and sets soc in state."""
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
    from zcu_tools.gui.session.operation_runner import OperationRunner

    state = _make_state()
    bus = EventBus()
    gate = OperationGate()
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    real_bg = BackgroundRunner()
    runner = OperationRunner(gate, handles, progress, real_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    assert svc.has_soc()
    assert state.exp_context.soccfg is not None


def test_start_connect_mock_soc_carries_default_simparam(qapp):
    """Mock-connect injects DEFAULT_SIMPARAM so the soc yields physical sim data."""
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    state = _make_state()
    bus = EventBus()
    gate = OperationGate()
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    real_bg = BackgroundRunner()
    runner = OperationRunner(gate, handles, progress, real_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    soc = state.exp_context.soc
    assert soc is not None, "soc must be set after mock connect"
    assert hasattr(soc, "_sim_params"), "mock soc must expose _sim_params"
    sim_params = getattr(soc, "_sim_params")
    assert sim_params is not DEFAULT_SIMPARAM  # copy-on-input
    assert sim_params == DEFAULT_SIMPARAM


def test_start_connect_mock_sim_params_override_is_honoured(qapp):
    """ConnectMockRequest(sim_params=...) propagates the override into the soc."""
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    custom = DEFAULT_SIMPARAM.model_copy(update={"snr": 9999.0})
    state = _make_state()
    bus = EventBus()
    gate = OperationGate()
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    real_bg = BackgroundRunner()
    runner = OperationRunner(gate, handles, progress, real_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest(sim_params=custom))
    loop.exec()

    soc = state.exp_context.soc
    assert soc is not None
    sim_params = getattr(soc, "_sim_params", None)
    assert sim_params is not None
    assert sim_params == custom


def test_connect_bumps_soc_not_context_version(qapp):
    """Connect must bump soc version only; context version must stay unchanged."""
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    state = _make_state()
    bus = EventBus()
    gate = OperationGate()
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    real_bg = BackgroundRunner()
    runner = OperationRunner(gate, handles, progress, real_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    ctx_before = state.version.get("context")
    soc_before = state.version.get("soc")

    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    assert state.version.get("soc") == soc_before + 1
    assert state.version.get("context") == ctx_before


# ---------------------------------------------------------------------------
# Synchronous connect (the soc.connect wire path) — connect_sync
# ---------------------------------------------------------------------------


def test_connect_sync_mock_sets_soc_and_emits_payload(qapp):
    """connect_sync runs inline (no bg), sets the soc, and emits SocChangedPayload
    synchronously (the hook the MockFluxProvisioner rides). It returns the handles
    directly and releases the SOC_CONNECT lease."""
    gate = OperationGate()
    svc, _bg, _handles = _make_svc(gate=gate)
    state = svc._state  # type: ignore[attr-defined]
    bus = svc._bus  # type: ignore[attr-defined]

    payloads: list[SocChangedPayload] = []
    bus.subscribe(SocChangedPayload, lambda p: payloads.append(p))

    soc_before = state.version.get("soc")
    soc, soccfg = svc.connect_sync(ConnectMockRequest())

    assert soc is not None and soccfg is not None
    assert svc.has_soc()
    assert state.exp_context.soc is soc
    # The shared _apply_connection side effects fired synchronously.
    assert state.version.get("soc") == soc_before + 1
    assert len(payloads) == 1
    assert payloads[0].is_mock is True
    # Lease released, no lingering active token.
    assert not gate.has_active(OperationKind.SOC_CONNECT)
    assert not svc.is_connect_active()


def test_connect_sync_rejects_concurrent_calls(qapp):
    """connect_sync holds the same SOC_CONNECT lease as the async path, so a
    concurrent connect (or the GUI button) fast-fails."""
    gate = OperationGate()
    svc, _bg, _handles = _make_svc(gate=gate)
    gate.register(1, OperationKind.SOC_CONNECT, owner_id="existing")
    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        svc.connect_sync(ConnectMockRequest())


def test_connect_sync_remote_failure_releases_lease_and_raises(qapp, monkeypatch):
    """A failed remote connect re-raises and still releases the lease (finally)."""
    gate = OperationGate()
    svc, _bg, _handles = _make_svc(gate=gate)

    import zcu_tools.remote as remote

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
    gate = OperationGate()
    svc, _bg, _handles = _make_svc(gate=gate)
    gate.register(1, OperationKind.SOC_CONNECT, owner_id="existing")
    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        svc.start_connect(ConnectMockRequest())


def test_start_connect_remote_unsupported_request_raises(qapp):
    svc, _bg, _handles = _make_svc()

    class Other:
        pass

    with pytest.raises(TypeError, match="Unsupported connect request"):
        svc.start_connect(Other())  # type: ignore[arg-type]


def test_start_connect_rejected_while_run_active(qapp):
    gate = OperationGate()
    svc, _bg, _handles = _make_svc(gate=gate)
    gate.register(1, MeasureOpKind.RUN, owner_id="tab")

    with pytest.raises(OperationConflictError, match="run is active"):
        svc.start_connect(ConnectMockRequest())


# ---------------------------------------------------------------------------
# Remote failure — end-to-end via QEventLoop
# ---------------------------------------------------------------------------


def test_start_connect_remote_failure_emits_failed(qapp, monkeypatch):
    """Remote connect failure: connection_failed emitted with 'nope' in message."""
    from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner

    state = _make_state()
    bus = EventBus()
    gate = OperationGate()
    handles = OperationHandles()
    progress = ProgressService(DirectProgressTransport())
    real_bg = BackgroundRunner()
    runner = OperationRunner(gate, handles, progress, real_bg, bus)  # type: ignore[arg-type]
    svc = SoCConnectionService(state, bus, gate, handles, runner)

    import zcu_tools.remote as remote

    def fail(ip: str, port: int) -> None:
        raise ConnectionRefusedError("nope")

    monkeypatch.setattr(remote, "make_soc_proxy", fail, raising=False)

    loop = QEventLoop()
    errors: list[str] = []
    svc.connection_failed.connect(lambda msg: errors.append(msg) or loop.quit())
    svc.connection_finished.connect(loop.quit)

    svc.start_connect(ConnectRemoteRequest(ip="127.0.0.1", port=7000))
    loop.exec()

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
    gate = OperationGate()
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
    svc.start_connect(ConnectMockRequest())
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
    gate = OperationGate()
    svc, fake_bg, handles = _make_svc(gate=gate)

    token = svc.start_connect(ConnectMockRequest())
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
    gate = OperationGate()
    svc, fake_bg, handles = _make_svc(gate=gate)

    svc.start_connect(ConnectMockRequest())
    assert gate.has_active(OperationKind.SOC_CONNECT)

    handles.cancel_all()

    # Op still runs to completion
    fake_bg.deliver_result()

    assert not gate.has_active(OperationKind.SOC_CONNECT)
    assert not svc.is_connect_active()
