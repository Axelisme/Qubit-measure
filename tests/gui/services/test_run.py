"""Tests for RunService — operation-boundary behavior given a RunPermit.

Static preconditions (context readiness, committed-cfg validity, soc capability)
are GuardService's responsibility (see test_guard.py). RunService only handles
the dynamic boundary: tab-busy, lease acquisition, worker start.
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
from zcu_tools.gui.app.main.services.guard import RunPermit
from zcu_tools.gui.app.main.services.operation_gate import OperationGate, OperationKind
from zcu_tools.gui.app.main.services.run import RunService
from zcu_tools.gui.app.main.state import ExpContext, Session, State


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

    runner = MagicMock()
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    gate = OperationGate()
    writeback = MagicMock()  # teardown_tab_items is a no-op in these tests
    progress = ProgressService(DirectProgressTransport())
    svc = RunService(state, runner, bus, gate, writeback, progress)
    return svc, gate, runner


def test_start_run_acquires_lease_and_starts_worker():
    state, tab_id, adapter = _make_state()
    svc, gate, runner = _make_run_service(state)

    svc.start_run(_make_permit(state, tab_id, adapter))

    runner.start_run.assert_called_once()
    assert gate.has_active(OperationKind.RUN)
    assert state.is_tab_running(tab_id)


def test_start_run_rejects_when_tab_busy():
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, True)
    svc, gate, runner = _make_run_service(state)

    with pytest.raises(RuntimeError, match="busy"):
        svc.start_run(_make_permit(state, tab_id, adapter))

    assert not gate.has_active(OperationKind.RUN)
    runner.start_run.assert_not_called()


def test_start_run_releases_lease_when_worker_start_raises():
    state, tab_id, adapter = _make_state()
    svc, gate, runner = _make_run_service(state)
    runner.start_run.side_effect = RuntimeError("worker boom")

    with pytest.raises(RuntimeError, match="worker boom"):
        svc.start_run(_make_permit(state, tab_id, adapter))

    assert not gate.has_active(OperationKind.RUN)
    assert not state.is_tab_running(tab_id)


def _last_run_finished_payload(bus_emit: MagicMock):
    from zcu_tools.gui.app.main.event_bus import GuiEvent, RunFinishedPayload

    for call in reversed(bus_emit.call_args_list):
        event, payload = call.args
        if event is GuiEvent.RUN_FINISHED and isinstance(payload, RunFinishedPayload):
            return payload
    raise AssertionError("no RUN_FINISHED emitted")


def test_run_finished_emits_outcome_finished():
    state, tab_id, adapter = _make_state()
    svc, _gate, _runner = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc._on_run_finished(tab_id, object())

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.tab_id == tab_id
    assert payload.outcome == "finished"


def test_run_failed_emits_outcome_failed_with_message():
    state, tab_id, adapter = _make_state()
    svc, _gate, _runner = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc._on_run_failed(tab_id, RuntimeError("boom"))

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "failed"
    assert payload.error_message == "boom"


def test_cancel_run_sets_operation_stop_event():
    # cancel_run is async-notification: it sets the operation's stop_event via
    # the gate. The worker then self-judges and drives _on_run_cancelled.
    state, tab_id, adapter = _make_state()
    svc, gate, _runner = _make_run_service(state)
    token = svc.start_run(_make_permit(state, tab_id, adapter))

    assert gate.poll(token) is None  # still pending before cancel
    svc.cancel_run()
    # The stop_event is set, but the operation only settles when the worker
    # self-judges and the terminal handler releases the lease.
    assert gate.has_active(OperationKind.RUN)


def test_run_cancelled_with_partial_result_reports_cancelled_and_keeps_result():
    state, tab_id, adapter = _make_state()
    svc, _gate, _runner = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    partial = object()
    svc._on_run_cancelled(tab_id, partial)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is partial
    assert not state.is_tab_running(tab_id)


def test_run_cancelled_without_result_reports_cancelled_and_keeps_no_result():
    from zcu_tools.gui.app.main.runner import NO_RESULT

    state, tab_id, adapter = _make_state()
    svc, _gate, _runner = _make_run_service(state)
    svc.start_run(_make_permit(state, tab_id, adapter))

    svc._on_run_cancelled(tab_id, NO_RESULT)

    payload = _last_run_finished_payload(svc._bus.emit)  # type: ignore[attr-defined]
    assert payload.outcome == "cancelled"
    assert state.get_tab(tab_id).run_result is None
    assert not state.is_tab_running(tab_id)
