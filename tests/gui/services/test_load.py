from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    NoAnalyzeParams,
)
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.app.main.services.guard import LoadPermit
from zcu_tools.gui.app.main.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _make_state() -> tuple[State, str, MagicMock]:
    state = State(ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None))
    tab_id = "tab-1"
    adapter = MagicMock()
    state.add_tab(
        tab_id,
        Session(adapter_name="any", adapter=adapter, cfg_schema=_empty_schema()),
    )
    return state, tab_id, adapter


def _service(state: State) -> tuple[LoadService, MagicMock, MagicMock]:
    bus = EventBus()
    emit = MagicMock()
    bus.emit = emit  # type: ignore[method-assign]
    writeback = MagicMock()
    return LoadService(state, bus, writeback), emit, writeback


def test_load_result_replaces_run_result_and_invalidates_dependents() -> None:
    state, tab_id, adapter = _make_state()
    stale_result = object()
    loaded = MagicMock()
    loaded.cfg_snapshot = object()
    adapter.load.return_value = loaded
    tab = state.get_tab(tab_id)
    tab.run_result = stale_result
    tab.result_source_path = "/tmp/old.hdf5"
    tab.analyze_result = object()
    tab.figure = Figure()
    tab.analyze_param_instance = NoAnalyzeParams()
    tab.post_analyze_result = object()
    tab.post_figure = Figure()
    tab.post_analyze_param_instance = object()
    tab.writeback_items = [MagicMock()]
    tab.applied_session_ids.add("md-1")
    cfg_version = state.version.get(f"tab:{tab_id}:cfg")
    save_path_version = state.version.get(f"tab:{tab_id}:save_path")
    svc, emit, writeback = _service(state)

    outcome = svc.load_result(LoadPermit(tab_id), "/tmp/new.hdf5")

    assert tab.run_result is loaded
    assert tab.result_source_path == "/tmp/new.hdf5"
    assert tab.analyze_result is None
    assert tab.figure is None
    assert tab.analyze_param_instance is None
    assert tab.post_analyze_result is None
    assert tab.post_figure is None
    assert tab.post_analyze_param_instance is None
    assert tab.writeback_items == []
    assert tab.applied_session_ids == set()
    writeback.teardown_tab_items.assert_called_once_with(tab_id)
    emit.assert_called_once_with(TabInteractionChangedPayload(tab_id=tab_id))
    assert outcome.result_type == type(loaded).__name__
    assert outcome.has_cfg_snapshot is True
    assert outcome.has_analyze_params is False
    assert state.version.get(f"tab:{tab_id}:result") == 1
    assert state.version.get(f"tab:{tab_id}:analyze") == 1
    assert state.version.get(f"tab:{tab_id}:post_analyze") == 1
    assert state.version.get(f"tab:{tab_id}:cfg") == cfg_version
    assert state.version.get(f"tab:{tab_id}:save_path") == save_path_version


def test_load_result_rejects_busy_tab_without_calling_adapter() -> None:
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, True)
    svc, _emit, writeback = _service(state)

    with pytest.raises(RuntimeError, match="busy"):
        svc.load_result(LoadPermit(tab_id), "/tmp/new.hdf5")

    adapter.load.assert_not_called()
    writeback.teardown_tab_items.assert_not_called()
    assert state.get_tab(tab_id).run_result is None


def test_load_result_error_leaves_state_unchanged() -> None:
    state, tab_id, adapter = _make_state()
    old = object()
    state.get_tab(tab_id).run_result = old
    adapter.load.side_effect = ValueError("bad file")
    svc, _emit, writeback = _service(state)

    with pytest.raises(LoadDataError, match="Cannot load this data file") as exc_info:
        svc.load_result(LoadPermit(tab_id), "/tmp/bad.hdf5")

    assert exc_info.value.reason_code == "invalid_data_file"
    assert "Details: bad file" in str(exc_info.value)
    writeback.teardown_tab_items.assert_not_called()
    assert state.get_tab(tab_id).run_result is old
    assert state.version.get(f"tab:{tab_id}:result") == 0


def test_load_result_wraps_unsupported_adapter() -> None:
    state, tab_id, adapter = _make_state()
    adapter.load.side_effect = NotImplementedError("unsupported")
    svc, _emit, writeback = _service(state)

    with pytest.raises(LoadDataError, match="does not support loading") as exc_info:
        svc.load_result(LoadPermit(tab_id), "/tmp/file.hdf5")

    assert exc_info.value.reason_code == "unsupported_load"
    writeback.teardown_tab_items.assert_not_called()
