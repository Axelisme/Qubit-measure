"""Unit tests for WritebackService.apply_tab_writeback_items.

Drives the service against a real State + ExpContext (real MetaDict) so we can
assert the context resource version is bumped when writeback writes md — the
writeback path writes md/ml directly (bypassing ContextService), so it must bump
``context`` itself for concurrency guards to detect the change.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.adapter import ContextReadiness, MetaDictWriteback
from zcu_tools.gui.event_bus import EventBus
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
