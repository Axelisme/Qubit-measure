"""operation.await dispatch handler.

The handler is off_main_thread (blocks the IO worker). It calls
operation_control.await_operation(operation_id, timeout) and shapes the AwaitResult into
a wire result (ADR-0066):
  - completed/cancelled → structured {reason:'completed', status:'cancelled',
    feedback?} (NOT a raise; feedback present only when a Stop reason was latched).
  - completed/failed → structured failed/error, not a failed tool call.
  - timeout → structured timeout/running signal.
  - completed/finished → {reason:'completed', status:'finished'}.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import Adapter
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.measure.services.operation_control import OperationControlFacet
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    OperationHandles,
    OperationOutcome,
)

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._synthetic import SampleAdapter, SampleCfg
from tests.gui.app.measure.services._operation_owner_fakes import (
    DeviceOperationOwner,
    TabOperationOwner,
    UnusedProgress,
)

from ._helpers import Fixture, call, open_client, recv_response, send


def _HANDLER(ctrl, params):
    # Generic operation handlers must not require the giant ctrl surface.
    adapter = SimpleNamespace(operation_control=ctrl)
    return METHOD_REGISTRY["operation.await"].handler(adapter, params)


def _ctrl(result: AwaitResult | None) -> MagicMock:
    ctrl = MagicMock()
    ctrl.await_operation.return_value = result
    return ctrl


def test_transport_await_leaves_gui_owner_responsive(tmp_path, qapp, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    awaiting = threading.Event()

    def run(inp: SampleCfg, *, context: RunContext):
        entered.set()
        # The owner query (or finally cleanup) releases this admitted adapter call.
        assert release.wait(30), "Run collaborator was not released"
        return SampleAdapter.run(inp, context=context)

    fixture = Fixture(project_root=str(tmp_path / "project"))
    fixture.registry.register(
        "held",
        Adapter(
            spec=SampleAdapter.spec,
            make_form=SampleAdapter.make_form,
            run=run,
            guide=None,
        ),
    )
    fixture.start()
    try:
        with open_client(fixture.service.port) as client:
            tab_id = call(client, "tab.new", {"adapter_name": "held"})["result"][
                "tab_id"
            ]
            started = call(client, "tab.run_start", {"tab_id": tab_id})
            assert started["ok"] is True
            token = started["result"]["operation_id"]
            qt_until(entered.is_set, label="admitted adapter Run")
            original = fixture.ctrl.operation_control.await_operation

            def observe_await(operation_id: int, timeout: float) -> AwaitResult:
                # This announces the delegate call, not arrival inside blocking await.
                awaiting.set()
                return original(operation_id, timeout)

            monkeypatch.setattr(
                fixture.ctrl.operation_control, "await_operation", observe_await
            )
            send(
                client,
                {
                    "id": "await",
                    "method": "operation.await",
                    "params": {"operation_id": token, "timeout": 30},
                },
            )
            qt_until(awaiting.is_set, label="transport await delegation")
            owner_results: list[bool] = []

            def query_owner() -> None:
                interaction = fixture.ctrl.tab_control.get_tab_snapshot(
                    tab_id
                ).interaction
                assert interaction is not None
                owner_results.append(interaction.is_running)

            fixture.ctrl.owner_scheduler.post(query_owner)
            qt_until(lambda: bool(owner_results), label="submitted owner query")
            assert owner_results == [True]
            pending = original(token, 0)
            assert pending is not None
            assert pending.reason == "timeout"
            release.set()
            reply = recv_response(client, "await")
            assert reply["ok"] is True
            assert reply["result"] == {"reason": "completed", "status": "finished"}
    finally:
        release.set()
        # These public lifecycles join the endpoint and Run worker/owner deliveries.
        fixture.stop()


# ---------------------------------------------------------------------------
# completed path
# ---------------------------------------------------------------------------


def test_unknown_and_evicted_handle_do_not_become_finished():
    handles = OperationHandles()
    control = OperationControlFacet(
        handles=handles,
        progress=UnusedProgress(),
        run_analyze=TabOperationOwner(),
        device=DeviceOperationOwner(),
    )
    for op in (777,):
        with pytest.raises(RemoteError) as exc_info:
            _HANDLER(control, {"operation_id": op, "timeout": 0})
        assert exc_info.value.code == ErrorCode.INVALID_PARAMS
        assert exc_info.value.reason == "unknown_op"

    first = handles.create(origin=EventOrigin(kind="user"))
    handles.settle(first, OperationOutcome("failed", "original failure"))
    for _ in range(40):
        op = handles.create(origin=EventOrigin(kind="user"))
        handles.settle(op, OperationOutcome("finished"))
    with pytest.raises(RemoteError) as exc_info:
        _HANDLER(control, {"operation_id": first, "timeout": 0})
    assert exc_info.value.reason == "unknown_op"


def test_invalid_timeout_fails_before_await():
    with pytest.raises(RemoteError) as exc_info:
        _HANDLER(
            _ctrl(AwaitResult(reason="timeout")), {"operation_id": 1, "timeout": 301}
        )
    assert exc_info.value.reason == "invalid_timeout"


def test_finished_returns_reason_and_status():
    ctrl = _ctrl(AwaitResult(reason="completed", outcome=OperationOutcome("finished")))
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out == {"reason": "completed", "status": "finished"}
    ctrl.await_operation.assert_called_once_with(7, 5.0)


def test_failed_returns_status_and_original_error():
    ctrl = _ctrl(
        AwaitResult(
            reason="completed",
            outcome=OperationOutcome("failed", "hardware boom"),
        )
    )
    assert _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0}) == {
        "reason": "completed",
        "status": "failed",
        "error": {"reason": "failed", "message": "hardware boom"},
    }


# ---------------------------------------------------------------------------
# cancelled path (ADR-0066) — structured result, NOT a raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [None, "", "stop reason from user"])
@pytest.mark.parametrize("settle_in_hook", [False, True])
def test_cancelled_stop_reason_reaches_wire(
    reason: str | None, settle_in_hook: bool
) -> None:
    handles = OperationHandles()

    def cancel_hook() -> None:
        if settle_in_hook:
            handles.settle(token, OperationOutcome("cancelled"))

    token = handles.create(cancel_hook, origin=EventOrigin(kind="user"))
    control = OperationControlFacet(
        handles=handles,
        progress=UnusedProgress(),
        run_analyze=TabOperationOwner(),
        device=DeviceOperationOwner(),
    )
    handles.stop(token, reason)
    params = {"operation_id": token, "timeout": 0}
    if not settle_in_hook:
        assert _HANDLER(control, params) == {"reason": "timeout"}
        handles.settle(token, OperationOutcome("cancelled"))

    expected = {"reason": "completed", "status": "cancelled"}
    if reason:
        expected["feedback"] = reason
    assert _HANDLER(control, params) == expected
    assert _HANDLER(control, params) == expected


def test_cancelled_without_feedback_no_raise():
    # Plain cancel (no Stop reason): status='cancelled', no feedback key.
    ctrl = _ctrl(
        AwaitResult(
            reason="completed",
            outcome=OperationOutcome("cancelled"),
            feedback=None,
        )
    )
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})
    assert out["reason"] == "completed"
    assert out["status"] == "cancelled"
    assert "feedback" not in out


def test_cancelled_does_not_raise():
    # Regression guard: a cancelled outcome must never raise (pre-fix behavior).
    ctrl = _ctrl(
        AwaitResult(
            reason="completed",
            outcome=OperationOutcome("cancelled"),
            feedback=None,
        )
    )
    out = _HANDLER(ctrl, {"operation_id": 7, "timeout": 5.0})  # must not raise
    assert out["status"] == "cancelled"


# ---------------------------------------------------------------------------
# timeout path
# ---------------------------------------------------------------------------


def test_timeout_returns_running_signal():
    ctrl = _ctrl(AwaitResult(reason="timeout"))
    assert _HANDLER(ctrl, {"operation_id": 7, "timeout": 0.1}) == {"reason": "timeout"}


# ---------------------------------------------------------------------------
# terminal result rereads
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# non-regression: finished / failed still work as before
# ---------------------------------------------------------------------------


def test_finished_not_affected_by_cancelled_change():
    ctrl = _ctrl(AwaitResult(reason="completed", outcome=OperationOutcome("finished")))
    out = _HANDLER(ctrl, {"operation_id": 42, "timeout": 1.0})
    assert out["status"] == "finished"
    assert "feedback" not in out


def test_failed_after_another_completed_read_stays_structured():
    ctrl = _ctrl(
        AwaitResult(reason="completed", outcome=OperationOutcome("failed", "boom"))
    )
    assert _HANDLER(ctrl, {"operation_id": 42, "timeout": 1.0})["status"] == "failed"
