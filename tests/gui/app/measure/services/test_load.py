from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, NoAnalyzeParams
from zcu_tools.gui.app.measure.services.guard import LoadPermit
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._cfg_fakes import make_cfg


def _empty_schema() -> CfgSchema:
    return CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())


def _make_state(*, load_data: bool = True) -> tuple[State, str, MagicMock]:
    state = State(SessionEnv(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None))
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=load_data)
    state.add_tab(
        tab_id,
        Session(adapter_name="any", adapter=adapter, cfg=make_cfg(_empty_schema())),
    )
    return state, tab_id, adapter


def _service(state: State) -> tuple[LoadService, MagicMock, MagicMock]:
    bus = EventBus()
    emit = MagicMock()
    bus.emit = emit  # type: ignore[method-assign]
    writeback = MagicMock()
    return (
        LoadService(state, writeback, provide_options=lambda source_id: ()),
        emit,
        writeback,
    )


@pytest.mark.parametrize("missing_cfg", [False, True])
def test_load_result_replaces_run_result_and_invalidates_dependents(
    missing_cfg: bool,
) -> None:
    state, tab_id, adapter = _make_state()
    stale_result = object()
    loaded = RunRecord(cfg=None if missing_cfg else ExpCfgModel(), result=object())
    adapter.load.return_value = loaded
    tab = state.get_tab(tab_id)
    tab.run.result = stale_result
    tab.run.source_path = "/tmp/old.hdf5"
    tab.analysis.result = object()
    analysis_plots = Plots(NonPresentingHost())
    analysis_plots.adopt("fit", Figure())
    analysis_plots.finish()
    tab.analysis.plots = analysis_plots
    tab.analysis.params = NoAnalyzeParams()
    tab.analysis.writeback_draft = MagicMock(is_active=True)
    tab.post_analysis.result = object()
    post_plots = Plots(NonPresentingHost())
    post_plots.adopt("fit", Figure())
    post_plots.finish()
    tab.post_analysis.plots = post_plots
    tab.post_analysis.params = object()
    tab.post_analysis.writeback_draft = MagicMock(is_active=True)
    cfg_version = state.version.get(f"tab:{tab_id}:cfg")
    save_path_version = state.version.get(f"tab:{tab_id}:save_path")
    svc, emit, writeback = _service(state)

    outcome = svc.load_result(LoadPermit(tab_id), "/tmp/new.hdf5")

    assert tab.run.result is loaded
    assert tab.run.source_path == "/tmp/new.hdf5"
    assert tab.analysis.result is None
    assert tab.analysis.plots is None
    assert tab.analysis.params is None
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None
    assert tab.post_analysis.params is None
    assert tab.analysis.writeback_draft is None
    assert tab.post_analysis.writeback_draft is None
    assert writeback.teardown_draft.call_count == 2
    emit.assert_not_called()
    assert outcome.result_type == "RunRecord"
    assert outcome.has_cfg_snapshot is (not missing_cfg)
    assert outcome.cfg_backfill == "not_applied"
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

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.load_result(LoadPermit(tab_id), "/tmp/new.hdf5")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""
    adapter.load.assert_not_called()
    writeback.teardown_draft.assert_not_called()
    assert state.get_tab(tab_id).run.result is None


def test_load_result_error_leaves_state_unchanged() -> None:
    state, tab_id, adapter = _make_state()
    old = object()
    state.get_tab(tab_id).run.result = old
    adapter.load.side_effect = ValueError("bad file")
    svc, _emit, writeback = _service(state)

    with pytest.raises(LoadDataError, match="Cannot load this data file") as exc_info:
        svc.load_result(LoadPermit(tab_id), "/tmp/bad.hdf5")

    assert exc_info.value.reason_code == "invalid_data_file"
    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert "Details: bad file" in str(exc_info.value)
    writeback.teardown_draft.assert_not_called()
    assert state.get_tab(tab_id).run.result is old
    assert state.version.get(f"tab:{tab_id}:result") == 0


def test_load_result_wraps_unsupported_adapter() -> None:
    state, tab_id, adapter = _make_state()
    adapter.load.side_effect = NotImplementedError("unsupported")
    svc, _emit, writeback = _service(state)

    with pytest.raises(LoadDataError, match="does not support loading") as exc_info:
        svc.load_result(LoadPermit(tab_id), "/tmp/file.hdf5")

    assert exc_info.value.reason_code == "unsupported_load"
    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    writeback.teardown_draft.assert_not_called()
