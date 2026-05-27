"""Tests for RunService capability-aware SoC validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    AdapterCapabilities,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
)
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.operation_gate import OperationGate, OperationKind
from zcu_tools.gui.services.run import RunService
from zcu_tools.gui.state import ExpContext, State, TabState


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _make_state(*, soc_attached: bool, requires_soc: bool = True) -> tuple[State, str]:
    md = MagicMock()
    ml = MagicMock()
    soc = MagicMock() if soc_attached else None
    soccfg = MagicMock() if soc_attached else None
    state = State(ExpContext(md=md, ml=ml, soc=soc, soccfg=soccfg, result_dir=""))
    tab_id = "tab-1"
    adapter = MagicMock()
    # Use a plain object whose `capabilities` is read by RunService; bypassing the
    # ClassVar guard while keeping the runtime contract intact for the test.
    adapter.capabilities = AdapterCapabilities(requires_soc=requires_soc)
    state.add_tab(
        tab_id,
        TabState(adapter_name="any", adapter=adapter, cfg_schema=_empty_schema()),
    )
    return state, tab_id


def _make_run_service(state: State) -> tuple[RunService, OperationGate, MagicMock]:
    runner = MagicMock()
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    gate = OperationGate()
    svc = RunService(state, runner, bus, gate)
    return svc, gate, runner


def test_start_run_fast_fails_before_acquire_when_capability_requires_soc():
    """requires_soc=True adapter with no soc must raise before acquiring a lease."""
    state, tab_id = _make_state(soc_attached=False)
    svc, gate, runner = _make_run_service(state)

    with pytest.raises(RuntimeError, match="soc"):
        svc.start_run(tab_id, _empty_schema())

    # Did not acquire a lease and did not start the worker
    assert not gate.has_active(OperationKind.RUN)
    runner.start_run.assert_not_called()
    # Tab not marked running
    assert not state.is_tab_running(tab_id)


def test_start_run_proceeds_when_capability_does_not_require_soc():
    """requires_soc=False adapter must start even without an active soc handle."""
    state, tab_id = _make_state(soc_attached=False, requires_soc=False)

    svc, gate, runner = _make_run_service(state)

    svc.start_run(tab_id, _empty_schema())

    runner.start_run.assert_called_once()
    assert gate.has_active(OperationKind.RUN)
    assert state.is_tab_running(tab_id)
