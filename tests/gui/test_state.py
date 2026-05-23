"""Unit tests for zcu_tools.gui.state."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.adapter import AnalyzeParam, SavePaths
from zcu_tools.gui.state import State, TabInteractionState, TabState


def _make_ctx():
    return MagicMock()


def _make_adapter():
    return MagicMock()


def test_tab_interaction_state_creation():
    state = TabInteractionState(
        global_run_active=True,
        is_running=False,
        is_analyzing=True,
        is_saving_data=False,
        has_context=False,
        has_soc=True,
        has_run_result=True,
        has_analyze_result=False,
    )
    assert state.global_run_active is True
    assert state.is_running is False
    assert state.is_analyzing is True
    assert state.is_saving_data is False


def test_add_tab_then_get_tab_returns_correct_tabstate():
    state = State(_make_ctx())
    adapter = _make_adapter()
    cfg_schema = object()
    adapter.make_default_cfg.return_value = cfg_schema
    state.add_tab("t1", adapter, _make_ctx())
    tab = state.get_tab("t1")
    assert isinstance(tab, TabState)
    assert tab.adapter is adapter
    assert tab.cfg_schema is cfg_schema


def test_add_tab_duplicate_raises():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    with pytest.raises(ValueError, match="already exists"):
        dup_adapter = _make_adapter()
        dup_adapter.make_default_cfg.return_value = object()
        state.add_tab("t1", dup_adapter, _make_ctx())


def test_remove_tab_clears_active_tab_id():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    state.set_active_tab("t1")
    state.remove_tab("t1")
    assert state.active_tab_id is None


def test_remove_busy_tab_raises():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    state.set_tab_analyzing("t1", True)
    with pytest.raises(RuntimeError, match="busy tab"):
        state.remove_tab("t1")


def test_set_active_tab_unknown_raises():
    state = State(_make_ctx())
    with pytest.raises(KeyError):
        state.set_active_tab("nonexistent")


def test_set_tab_running_updates_running_tab_id():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    assert state.is_run_active() is False
    state.set_tab_running("t1", True)
    assert state.is_run_active() is True
    assert state.running_tab_id == "t1"
    assert state.is_tab_running("t1") is True
    state.set_tab_running("t1", False)
    assert state.is_run_active() is False
    assert state.running_tab_id is None


def test_is_tab_busy_checks_per_tab_flags():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    state.add_tab("t2", adapter, _make_ctx())
    assert state.is_tab_busy("t1") is False
    state.set_tab_saving_data("t1", True)
    assert state.is_tab_busy("t1") is True
    assert state.is_tab_busy("t2") is False


def test_update_tab_result_stores_result_and_clears_stale_analyze_data():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    param = AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.0)
    state.update_tab_analyze_params("t1", [param], {"threshold": 0.5})
    state.update_tab_analyze("t1", object(), Figure())
    state.update_tab_suggested_save_paths("t1", SavePaths("/tmp/a", "/tmp/b"))
    state.update_tab_result("t1", object())
    tab = state.get_tab("t1")
    assert tab.analyze_result is None
    assert tab.figure is None
    assert tab.analyze_params == []
    assert tab.analyze_param_values == {}
    assert tab.suggested_save_paths is None


def test_update_tab_analyze_stores_analyze_result_and_figure():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    analyze_result = object()
    fig = Figure()
    state.update_tab_analyze("t1", analyze_result, fig)
    tab = state.get_tab("t1")
    assert tab.analyze_result is analyze_result
    assert tab.figure is fig


def test_update_tab_analyze_params_defaults_values_when_missing():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    params = [AnalyzeParam(key="threshold", label="Threshold", type=float, default=0.2)]
    state.update_tab_analyze_params("t1", params)
    assert state.get_tab("t1").analyze_param_values == {"threshold": 0.2}


def test_update_tab_save_path_overrides_merges_paths():
    state = State(_make_ctx())
    adapter = _make_adapter()
    adapter.make_default_cfg.return_value = object()
    state.add_tab("t1", adapter, _make_ctx())
    state.update_tab_suggested_save_paths("t1", SavePaths("/tmp/data", "/tmp/image"))
    state.update_tab_save_path_overrides("t1", data_path="/tmp/custom-data")
    state.update_tab_save_path_overrides("t1", image_path="/tmp/custom-image")
    assert state.get_effective_save_paths("t1") == SavePaths(
        "/tmp/custom-data", "/tmp/custom-image"
    )


def test_set_context_replaces_exp_context():
    ctx1 = _make_ctx()
    ctx2 = _make_ctx()
    state = State(ctx1)
    assert state.exp_context is ctx1
    state.set_context(ctx2)
    assert state.exp_context is ctx2
