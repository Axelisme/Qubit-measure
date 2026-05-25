"""Tests for ConnectionService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.connection import (
    ConnectionService,
    ConnectMockRequest,
    ConnectRemoteRequest,
    LoadPredictorRequest,
    PredictFreqRequest,
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.gui.state import ExpContext, State


def _make_svc() -> ConnectionService:
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    return ConnectionService(state, EventBus())


def test_start_connect_mock_emits_finished_and_updates_context(qapp):
    svc = _make_svc()
    loop = QEventLoop()
    svc.connection_finished.connect(loop.quit)
    svc.connection_failed.connect(lambda msg: loop.quit())

    svc.start_connect(ConnectMockRequest())
    loop.exec()

    assert svc.has_soc()
    assert svc._state.exp_context.soccfg is not None


def test_start_connect_rejects_concurrent_calls(qapp):
    svc = _make_svc()
    # First mock dispatches via QTimer.singleShot, so the worker slot is empty
    # by the time we call again — instead, exercise the remote path which holds
    # the worker until its QThread finishes.
    svc._active_worker = MagicMock()
    with pytest.raises(RuntimeError, match="already in progress"):
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
