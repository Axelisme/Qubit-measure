"""Unit tests for WritebackService.apply_tab_writeback_items.

Drives the service against a real State + ExpContext (real MetaDict) so we can
assert the context resource version is bumped when writeback writes md — the
writeback path writes md/ml directly (bypassing ContextService), so it must bump
``context`` itself for concurrency guards to detect the change.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from zcu_tools.gui.adapter import (
    ContextReadiness,
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.event_bus import EventBus, GuiEvent
from zcu_tools.gui.services.guard import WritebackPermit
from zcu_tools.gui.services.writeback import WritebackService
from zcu_tools.gui.state import ExpContext, State, TabState
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_state_with_tab(tab_id: str = "t1") -> State:
    state = State(
        ExpContext(
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.ACTIVE,
        )
    )
    state.add_tab(
        tab_id,
        TabState(adapter_name="fake", adapter=MagicMock(), cfg_schema=MagicMock()),
    )
    return state


def test_apply_md_writeback_bumps_context_version():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())
    before = state.version.get("context")

    item = MetaDictWriteback(
        key="md:r_f",
        description="update r_f",
        current_value=6000.0,
        md_key="r_f",
        proposed_value=6100.0,
    )
    applied = svc.apply_tab_writeback_items(WritebackPermit(tab_id="t1"), [item])

    assert applied == [item.key]
    assert state.exp_context.md.r_f == 6100.0
    # Writeback wrote md → context version must advance so a later
    # context-dependent op detects the change.
    assert state.version.get("context") == before + 1


def test_apply_nothing_selected_does_not_bump_context():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())
    before = state.version.get("context")

    item = MetaDictWriteback(
        key="md:r_f",
        description="update r_f",
        current_value=6000.0,
        md_key="r_f",
        proposed_value=6100.0,
    )
    item.selected = False
    applied = svc.apply_tab_writeback_items(WritebackPermit(tab_id="t1"), [item])

    assert applied == []
    # No md/ml touched → no spurious context bump.
    assert state.version.get("context") == before


# ---------------------------------------------------------------------------
# get_tab_writeback_items
# ---------------------------------------------------------------------------


def test_get_writeback_items_returns_empty_when_no_run_result():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())

    items = svc.get_tab_writeback_items("t1")

    assert items == []


def test_get_writeback_items_returns_empty_when_no_analyze_result():
    state = _make_state_with_tab()
    state.update_tab_result("t1", object())
    svc = WritebackService(state, EventBus())

    items = svc.get_tab_writeback_items("t1")

    assert items == []


def test_get_writeback_items_calls_adapter_and_sets_selected():
    state = _make_state_with_tab()
    state.update_tab_result("t1", object())

    fake_analyze_result = MagicMock()
    fake_analyze_result.figure = None
    state.update_tab_analyze("t1", fake_analyze_result, None)

    adapter: MagicMock = state.get_tab("t1").adapter  # type: ignore[assignment]
    item = MetaDictWriteback(
        key="md:r_f",
        description="update r_f",
        current_value=6000.0,
        md_key="r_f",
        proposed_value=6100.0,
    )
    adapter.get_writeback_items.return_value = [item]

    svc = WritebackService(state, EventBus())
    items = svc.get_tab_writeback_items("t1")

    assert len(items) == 1
    assert items[0] is item
    assert items[0].selected is True  # not yet applied


# ---------------------------------------------------------------------------
# apply_tab_writeback_items — ModuleWriteback (proposed_module path)
# ---------------------------------------------------------------------------


def test_apply_module_writeback_proposed_module_registers_and_bumps():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())
    before = state.version.get("context")

    fake_module = MagicMock()
    item = ModuleWriteback(
        key="ml:qub",
        description="update qub",
        current_value=None,
        module_name="qub",
        proposed_module=fake_module,
    )
    applied = svc.apply_tab_writeback_items(WritebackPermit(tab_id="t1"), [item])

    assert applied == [item.key]
    assert state.version.get("context") == before + 1


def test_apply_module_writeback_emits_ml_changed():
    state = _make_state_with_tab()
    bus = EventBus()
    received: list = []
    bus.subscribe(GuiEvent.ML_CHANGED, lambda p: received.append(p))
    svc = WritebackService(state, bus)

    fake_module = MagicMock()
    item = ModuleWriteback(
        key="ml:qub",
        description="update qub",
        current_value=None,
        module_name="qub",
        proposed_module=fake_module,
    )
    svc.apply_tab_writeback_items(WritebackPermit(tab_id="t1"), [item])

    assert len(received) == 1


# ---------------------------------------------------------------------------
# apply_tab_writeback_items — WaveformWriteback (proposed_waveform path)
# ---------------------------------------------------------------------------


def test_apply_waveform_writeback_proposed_waveform_registers_and_bumps():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())
    before = state.version.get("context")

    fake_waveform = MagicMock()
    item = WaveformWriteback(
        key="wf:gauss",
        description="update gauss",
        current_value=None,
        waveform_name="gauss",
        proposed_waveform=fake_waveform,
    )
    applied = svc.apply_tab_writeback_items(WritebackPermit(tab_id="t1"), [item])

    assert applied == [item.key]
    assert state.version.get("context") == before + 1


# ---------------------------------------------------------------------------
# touched_ml → dump() when has_persistence
# ---------------------------------------------------------------------------


def test_apply_ml_writeback_calls_dump_when_has_persistence():
    state = _make_state_with_tab()
    # Replace ml with a MagicMock that reports has_persistence=True
    mock_ml = MagicMock()
    mock_ml.has_persistence = True
    from dataclasses import replace as dc_replace

    new_ctx = dc_replace(state.exp_context, ml=mock_ml)
    state.set_context(new_ctx)
    svc = WritebackService(state, EventBus())

    fake_module = MagicMock()
    item = ModuleWriteback(
        key="ml:qub",
        description="update qub",
        current_value=None,
        module_name="qub",
        proposed_module=fake_module,
    )
    svc.apply_tab_writeback_items(WritebackPermit(tab_id="t1"), [item])

    mock_ml.dump.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_module_item — both None → RuntimeError
# ---------------------------------------------------------------------------


def test_resolve_module_item_neither_proposal_raises():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())

    item = ModuleWriteback(
        key="ml:qub",
        description="update qub",
        current_value=None,
        module_name="qub",
        proposed_module=None,
        edit_schema=None,
        edited_schema=None,
    )
    with pytest.raises(RuntimeError, match="no proposal"):
        svc._resolve_module_item(item)


# ---------------------------------------------------------------------------
# _resolve_waveform_item — both None → RuntimeError
# ---------------------------------------------------------------------------


def test_resolve_waveform_item_neither_proposal_raises():
    state = _make_state_with_tab()
    svc = WritebackService(state, EventBus())

    item = WaveformWriteback(
        key="wf:gauss",
        description="update gauss",
        current_value=None,
        waveform_name="gauss",
        proposed_waveform=None,
        edit_schema=None,
        edited_schema=None,
    )
    with pytest.raises(RuntimeError, match="no proposal"):
        svc._resolve_waveform_item(item)
