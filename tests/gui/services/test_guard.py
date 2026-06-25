"""Tests for GuardService Permit issuance across readiness and result states."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, ContextReadiness
from zcu_tools.gui.app.main.services.guard import (
    AnalyzePermit,
    GuardError,
    GuardService,
    LoadPermit,
    RunPermit,
    SavePermit,
    WritebackPermit,
)
from zcu_tools.gui.app.main.state import ExpContext, Session, State


def _make_state(
    *,
    readiness: ContextReadiness,
    soc_attached: bool = True,
    requires_soc: bool = True,
    lowering_raises: bool = False,
    run_result: object = object(),
    analyze_result: object = object(),
) -> tuple[State, str]:
    md = MagicMock()
    ml = MagicMock()
    soc = MagicMock() if soc_attached else None
    soccfg = MagicMock() if soc_attached else None
    state = State(ExpContext(md=md, ml=ml, soc=soc, soccfg=soccfg, readiness=readiness))
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(requires_soc=requires_soc)

    schema = MagicMock()
    if lowering_raises:
        schema.to_raw_dict.side_effect = RuntimeError("field 'gain' is unset")
    else:
        schema.to_raw_dict.return_value = {"ok": True}

    tab = Session(adapter_name="any", adapter=adapter, cfg_schema=schema)
    tab.run_result = run_result
    tab.analyze_result = analyze_result
    state.add_tab(tab_id, tab)
    return state, tab_id


# ---------------------------------------------------------------------------
# Run permit
# ---------------------------------------------------------------------------


def test_run_permit_issued_for_active_valid_cfg():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = GuardService(state)

    permit = guard.acquire_run_permit(tab_id)

    assert isinstance(permit, RunPermit)
    assert permit.tab_id == tab_id
    assert permit.schema is state.get_tab(tab_id).cfg_schema
    assert permit.request.soc is state.exp_context.soc
    assert permit.adapter is state.get_tab(tab_id).adapter


@pytest.mark.parametrize("readiness", [ContextReadiness.EMPTY, ContextReadiness.DRAFT])
def test_run_permit_rejected_when_not_active(readiness: ContextReadiness):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="active file-backed context") as exc:
        guard.acquire_run_permit(tab_id)
    assert exc.value.reason_code == "no_active_context"


def test_save_permit_reason_code_no_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError) as exc:
        guard.acquire_save_permit(tab_id)
    assert exc.value.reason_code == "no_run_result"


def test_analyze_permit_reason_code_no_context():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError) as exc:
        guard.acquire_analyze_permit(tab_id)
    assert exc.value.reason_code == "no_context"


def test_run_permit_rejected_on_invalid_cfg():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, lowering_raises=True)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="Config invalid"):
        guard.acquire_run_permit(tab_id)


def test_run_permit_rejected_when_soc_required_but_missing():
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, soc_attached=False, requires_soc=True
    )
    guard = GuardService(state)

    with pytest.raises(GuardError, match="soc"):
        guard.acquire_run_permit(tab_id)


def test_run_permit_issued_without_soc_when_capability_does_not_require():
    state, tab_id = _make_state(
        readiness=ContextReadiness.ACTIVE, soc_attached=False, requires_soc=False
    )
    guard = GuardService(state)

    permit = guard.acquire_run_permit(tab_id)
    assert isinstance(permit, RunPermit)


def test_run_permit_rejected_for_unknown_tab():
    state, _ = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="Unknown tab"):
        guard.acquire_run_permit("does-not-exist")


# ---------------------------------------------------------------------------
# Save permit
# ---------------------------------------------------------------------------


def test_save_permit_issued_for_active_with_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE)
    guard = GuardService(state)

    assert isinstance(guard.acquire_save_permit(tab_id), SavePermit)


@pytest.mark.parametrize("readiness", [ContextReadiness.EMPTY, ContextReadiness.DRAFT])
def test_save_permit_rejected_when_not_active(readiness: ContextReadiness):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="active file-backed context"):
        guard.acquire_save_permit(tab_id)


def test_save_permit_rejected_without_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="No run result"):
        guard.acquire_save_permit(tab_id)


# ---------------------------------------------------------------------------
# Load permit (allows DRAFT, no SoC/result requirement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [ContextReadiness.DRAFT, ContextReadiness.ACTIVE])
def test_load_permit_issued_with_context_without_soc_or_result(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(
        readiness=readiness,
        soc_attached=False,
        run_result=None,
        analyze_result=None,
    )
    guard = GuardService(state)

    assert isinstance(guard.acquire_load_permit(tab_id), LoadPermit)


def test_load_permit_rejected_when_empty():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="no experiment context") as exc:
        guard.acquire_load_permit(tab_id)
    assert exc.value.reason_code == "no_context"


# ---------------------------------------------------------------------------
# Analyze permit (allows DRAFT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [ContextReadiness.DRAFT, ContextReadiness.ACTIVE])
def test_analyze_permit_issued_with_context_and_result(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    assert isinstance(guard.acquire_analyze_permit(tab_id), AnalyzePermit)


def test_analyze_permit_rejected_when_empty():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="no experiment context"):
        guard.acquire_analyze_permit(tab_id)


def test_analyze_permit_rejected_without_run_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, run_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="No run result"):
        guard.acquire_analyze_permit(tab_id)


# ---------------------------------------------------------------------------
# Writeback permit (allows DRAFT)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readiness", [ContextReadiness.DRAFT, ContextReadiness.ACTIVE])
def test_writeback_permit_issued_with_context_and_analyze(
    readiness: ContextReadiness,
):
    state, tab_id = _make_state(readiness=readiness)
    guard = GuardService(state)

    assert isinstance(guard.acquire_writeback_permit(tab_id), WritebackPermit)


def test_writeback_permit_rejected_when_empty():
    state, tab_id = _make_state(readiness=ContextReadiness.EMPTY)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="no experiment context"):
        guard.acquire_writeback_permit(tab_id)


def test_writeback_permit_rejected_without_analyze_result():
    state, tab_id = _make_state(readiness=ContextReadiness.ACTIVE, analyze_result=None)
    guard = GuardService(state)

    with pytest.raises(GuardError, match="No analyze result"):
        guard.acquire_writeback_permit(tab_id)
