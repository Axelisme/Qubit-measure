"""Tests for RunService — operation-boundary behavior given a RunPermit.

Static preconditions (context readiness, committed-cfg validity, soc capability)
are GuardService's responsibility (see test_guard.py). RunService only handles
the dynamic boundary: tab-busy, lease acquisition, worker start.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    RunRequest,
)
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.guard import RunPermit
from zcu_tools.gui.services.operation_gate import OperationGate, OperationKind
from zcu_tools.gui.services.run import RunService
from zcu_tools.gui.state import ExpContext, State, TabState


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
        TabState(adapter_name="any", adapter=adapter, cfg_schema=_empty_schema()),
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
    runner = MagicMock()
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    gate = OperationGate()
    svc = RunService(state, runner, bus, gate)
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
