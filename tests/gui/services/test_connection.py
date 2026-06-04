"""Tests for ConnectionService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.gui.app.main.event_bus import EventBus, GuiEvent
from zcu_tools.gui.app.main.services.connection import (
    ConnectionService,
    ConnectMockRequest,
    ConnectRemoteRequest,
    LoadPredictorRequest,
    PredictFreqRequest,
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
)
from zcu_tools.gui.app.main.state import ExpContext, State


def _make_svc(gate: OperationGate | None = None) -> ConnectionService:
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    return ConnectionService(state, EventBus(), gate or OperationGate())


def test_start_connect_mock_emits_finished_and_updates_context(qapp):
    svc = _make_svc()
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    assert svc.has_soc()
    assert svc._state.exp_context.soccfg is not None


def test_connect_bumps_soc_not_context_version(qapp):
    svc = _make_svc()
    ctx_before = svc._state.version.get("context")
    soc_before = svc._state.version.get("soc")
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    # soc is its own resource; a connect must not spuriously bump context
    # (md/ml content did not change).
    assert svc._state.version.get("soc") == soc_before + 1
    assert svc._state.version.get("context") == ctx_before


def test_predictor_load_clear_does_not_bump_context_version(qapp):
    import dataclasses

    svc = _make_svc()
    ctx_before = svc._state.version.get("context")
    fake = MagicMock()
    svc._state.set_context(dataclasses.replace(svc._state.exp_context, predictor=fake))
    svc.clear_predictor()
    # predictor is not a guarded resource; swapping it must not bump context.
    assert svc._state.version.get("context") == ctx_before


def test_start_connect_rejects_concurrent_calls(qapp):
    gate = OperationGate()
    svc = _make_svc(gate)
    gate.acquire(OperationKind.SOC_CONNECT, owner_id="existing")
    with pytest.raises(OperationConflictError, match="soc_connect is active"):
        svc.start_connect(ConnectMockRequest())


def test_load_predictor_wraps_io_errors(qapp, tmp_path):
    svc = _make_svc()
    with pytest.raises(PredictorLoadError):
        svc.load_predictor(
            LoadPredictorRequest(path=str(tmp_path / "missing.json"), flux_bias=0.0)
        )


def test_predict_freq_without_predictor_raises(qapp):
    svc = _make_svc()
    with pytest.raises(PredictorNotLoaded):
        svc.predict_freq(PredictFreqRequest(value=0.0, transition=(0, 1)))


def test_clear_predictor_resets_state(qapp):
    svc = _make_svc()
    # Inject a fake predictor without going through load_predictor.
    import dataclasses

    fake = MagicMock()
    fake.flux_bias = 0.3
    svc._state.set_context(dataclasses.replace(svc._state.exp_context, predictor=fake))
    svc._predictor_path = "/fake/path.json"

    svc.clear_predictor()
    assert svc.get_predictor() is None
    assert svc.get_predictor_info() is None


def test_start_connect_remote_unsupported_request_raises(qapp):
    svc = _make_svc()

    class Other:
        pass

    with pytest.raises(TypeError, match="Unsupported connect request"):
        svc.start_connect(Other())  # type: ignore[arg-type]


def test_start_connect_remote_failure_emits_failed(qapp, monkeypatch):
    svc = _make_svc()

    # Force make_soc_proxy to raise a connection error inside the worker.
    import zcu_tools.remote as remote

    def fail(ip, port):
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


def test_start_connect_rejected_while_run_active(qapp):
    gate = OperationGate()
    svc = _make_svc(gate)
    gate.acquire(OperationKind.RUN, owner_id="tab")

    with pytest.raises(OperationConflictError, match="run is active"):
        svc.start_connect(ConnectMockRequest())


def test_soc_changed_subscriber_failure_releases_connection_lease(qapp):
    gate = OperationGate()
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    bus = EventBus()
    bus.subscribe(
        GuiEvent.SOC_CHANGED, MagicMock(side_effect=RuntimeError("render failed"))
    )
    svc = ConnectionService(state, bus, gate)
    svc._active_lease = gate.acquire(OperationKind.SOC_CONNECT, owner_id="soc")
    with pytest.raises(RuntimeError, match="render failed"):
        svc._finish_success(MagicMock(), MagicMock())
    assert not gate.has_active(OperationKind.SOC_CONNECT)
