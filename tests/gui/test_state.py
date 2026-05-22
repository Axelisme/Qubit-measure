"""Unit tests for zcu_tools.gui.state (Phase 3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.state import State, TabInteractionState, TabState


def _make_ctx():
    return MagicMock()


def _make_adapter():
    return MagicMock()


# ---------------------------------------------------------------------------
# TabInteractionState
# ---------------------------------------------------------------------------


def test_tab_interaction_state_creation():
    s = TabInteractionState(
        is_running=True,
        has_context=False,
        has_soc=True,
        has_run_result=True,
        has_analyze_result=False,
    )
    assert s.is_running is True
    assert s.has_context is False
    assert s.has_soc is True
    assert s.has_run_result is True
    assert s.has_analyze_result is False


# ---------------------------------------------------------------------------
# add_tab / get_tab
# ---------------------------------------------------------------------------


def test_add_tab_then_get_tab_returns_correct_tabstate():
    state = State(_make_ctx())
    adapter = _make_adapter()
    default_cfg = object()
    adapter.make_default_cfg.return_value = default_cfg
    state.add_tab("t1", adapter, _make_ctx())
    tab = state.get_tab("t1")
    assert isinstance(tab, TabState)
    assert tab.adapter is adapter
    assert tab.default_cfg is default_cfg


def test_add_tab_duplicate_raises():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    with pytest.raises(ValueError, match="already exists"):
        dup_adapter = _make_adapter()
        dup_adapter.make_default_cfg.return_value = object()
        state.add_tab("t1", dup_adapter, _make_ctx())


def test_remove_tab_clears_tab():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    state.remove_tab("t1")
    assert "t1" not in state.tabs


def test_remove_tab_clears_active_tab_id():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    state.set_active_tab("t1")
    state.remove_tab("t1")
    assert state.active_tab_id is None


# ---------------------------------------------------------------------------
# set_active_tab
# ---------------------------------------------------------------------------


def test_set_active_tab_updates_active_tab_id():
    state = State(_make_ctx())
    adapter1 = _make_adapter()
    adapter1.make_default_cfg.return_value = object()
    adapter2 = _make_adapter()
    adapter2.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter1, _make_ctx())
    state.add_tab("t2", adapter2, _make_ctx())
    state.set_active_tab("t2")
    assert state.active_tab_id == "t2"


def test_set_active_tab_unknown_raises():
    state = State(_make_ctx())
    with pytest.raises(KeyError):
        state.set_active_tab("nonexistent")


# ---------------------------------------------------------------------------
# set_running
# ---------------------------------------------------------------------------


def test_set_running_updates_is_running():
    state = State(_make_ctx())
    assert not state.is_running
    state.set_running(True)
    assert state.is_running
    state.set_running(False)
    assert not state.is_running


# ---------------------------------------------------------------------------
# update_tab_result / update_tab_analyze
# ---------------------------------------------------------------------------


def test_update_tab_result_stores_result_and_clears_stale_analyze_data():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    result = object()
    state.update_tab_analyze("t1", object(), Figure())
    state.update_tab_result("t1", result)
    tab = state.get_tab("t1")
    assert tab.run_result is result
    assert tab.analyze_result is None
    assert tab.figure is None


def test_update_tab_analyze_stores_analyze_result_and_figure():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    ar = object()
    fig = Figure()
    state.update_tab_analyze("t1", ar, fig)
    tab = state.get_tab("t1")
    assert tab.analyze_result is ar
    assert tab.figure is fig


# ---------------------------------------------------------------------------
# set_context
# ---------------------------------------------------------------------------


def test_set_context_replaces_exp_context():
    ctx1 = _make_ctx()
    ctx2 = _make_ctx()
    state = State(ctx1)
    assert state.exp_context is ctx1
    state.set_context(ctx2)
    assert state.exp_context is ctx2
