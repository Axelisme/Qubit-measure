"""Tests for RunService — operation-boundary behavior given a RunPermit.

Static preconditions (context readiness, committed-cfg validity, soc capability)
are GuardService's responsibility (see test_guard.py). RunService only handles
the dynamic boundary: tab-busy, lease acquisition, bg submit, and — since
ADR-0019 — the cancel *interpretation* of bg's done/failed (it owns the
stop_event, so it relabels finished vs cancelled here).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    AdapterCapabilities,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    RunRequest,
)
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.app.main.services.background import NO_RESULT
from zcu_tools.gui.app.main.services.guard import RunPermit
from zcu_tools.gui.app.main.services.operation_gate import OperationGate, OperationKind
from zcu_tools.gui.app.main.services.run import RunService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.session.operation_handles import OperationHandles


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _make_state() -> tuple[State, str, MagicMock]:
    md = MagicMock()
    ml = MagicMock()
    state = State(ExpContext(md=md, ml=ml, soc=MagicMock(), soccfg=MagicMock()))
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(requires_soc=True)
    state.add_tab(
        tab_id,
        Session(adapter_name="any", adapter=adapter, cfg_schema=_empty_schema()),
    )
    return state, tab_id, adapter


def _make_permit(state: State, tab_id: str, adapter: MagicMock) -> RunPermit:
    ctx = state.exp_context
    return RunPermit(
        tab_id=tab_id,
        request=RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg),
        schema=state.get_tab(tab_id).cfg_schema,
        adapter=adapter,
    )


def _make_run_service(state: State) -> tuple[RunService, OperationGate, MagicMock]:
    from zcu_tools.gui.app.main.services.progress import ProgressService

    from ._progress_fakes import DirectProgressTransport

    bg = MagicMock()  # BackgroundService stand-in; submit() is inspected per-test
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    gate = OperationGate()
    handles = OperationHandles()
    writeback = MagicMock()  # teardown_tab_items is a no-op in these tests
    progress = ProgressService(DirectProgressTransport())
    svc = RunService(state, bg, bus, gate, handles, writeback, progress)
    return svc, gate, bg


def test_start_run_acquires_lease_and_submits_to_bg():
    state, tab_id, adapter = _make_state()
    svc, gate, bg = _make_run_service(state)

    svc.start_run(_make_permit(state, tab_id, adapter))

    bg.submit.assert_called_once()
    # OffMain-thread strategy: not pooled.
    assert bg.submit.call_args.kwargs["run_in_pool"] is False
    assert gate.has_active(OperationKind.RUN)
    assert state.is_tab_running(tab_id)


def test_start_run_rejects_when_tab_busy():
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, True)
    svc, gate, bg = _make_run_service(state)

    with pytest.raises(RuntimeError, match="busy"):
        svc.start_run(_make_permit(state, tab_id, adapter))

    assert not gate.has_active(OperationKind.RUN)
    bg.submit.assert_not_called()


def test_start_run_releases_lease_when_submit_raises():
    state, tab_id, adapter = _make_state()
    svc, gate, bg = _make_run_service(state)
    bg.submit.side_effect = RuntimeError("worker boom")

    with pytest.raises(RuntimeError, match="worker boom"):
        svc.start_run(_make_permit(state, tab_id, adapter))

    assert not gate.has_active(OperationKind.RUN)
    assert not state.is_tab_running(tab_id)


def _last_run_finished_payload(bus_emit: MagicMock):
    from zcu_tools.gui.app.main.event_bus import RunFinishedPayload

    for call in reversed(bus_emit.call_args_list):
        (payload,) = call.args
        if isinstance(payload, RunFinishedPayload):
            return payload
    raise AssertionError("no RUN_FINISHED emitted")


def test_run_finished_emits_outcome_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, _bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc._on_run_finished(tab_id, object())

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.tab_id == tab_id
    assert payload.outcome == "finished"


def test_run_failed_emits_outcome_failed_with_message():
    state, tab_id, adapter = _make_state()
    svc, _gate, _bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc._on_run_failed(tab_id, RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_cancel_run_sets_operation_stop_event():
    # cancel_run is async-notification: it sets the operation's stop_event via
    # the gate. The worker then self-judges and drives _on_run_cancelled.
    state, tab_id, adapter = _make_state()
    svc, gate, _bg = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter))

    assert svc._handles.poll(token) is None  # still pending before cancel
    svc.cancel_run()
    # The stop_event is set, but the operation only settles when the worker
    # self-judges and the terminal handler releases the lease.
    assert gate.has_active(OperationKind.RUN)


def test_run_cancelled_with_partial_result_reports_cancelled_and_keeps_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, _bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    partial = object()
    svc._on_run_cancelled(tab_id, partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is partial
    assert not state.is_tab_running(tab_id)


def test_run_cancelled_without_result_reports_cancelled_and_keeps_no_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, _bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc._on_run_cancelled(tab_id, NO_RESULT)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is None
    assert not state.is_tab_running(tab_id)


# --- bg outcome -> cancel interpretation (ADR-0019): RunService owns the
#     stop_event, so it relabels bg's done/failed into finished vs cancelled.
#     Drive the real closures bg.submit was handed.


def _submitted_callbacks(bg: MagicMock):
    kwargs = bg.submit.call_args.kwargs
    return kwargs["on_done"], kwargs["on_error"]


def test_bg_done_without_cancel_reports_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))
    on_done, _ = _submitted_callbacks(bg)

    result = object()
    on_done(result)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "finished"
    assert state.get_tab(tab_id).run_result is result


def test_bg_done_after_cancel_reports_cancelled_with_partial():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))
    on_done, _ = _submitted_callbacks(bg)

    svc.cancel_run()  # sets the captured stop_event
    partial = object()
    on_done(partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is partial


def test_bg_error_without_cancel_reports_failed():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))
    _, on_error = _submitted_callbacks(bg)

    on_error(RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_bg_error_after_cancel_reports_cancelled_without_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, bg = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))
    _, on_error = _submitted_callbacks(bg)

    svc.cancel_run()
    on_error(RuntimeError("interrupted"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is None
