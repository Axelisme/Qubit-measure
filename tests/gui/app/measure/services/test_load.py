from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.datafile import RunMetadata
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    Form,
    LoadDataRequest,
    NoAnalyzeParams,
)
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
)
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._load_support import historical_metadata


def _make_state(*, load_data: bool = True) -> tuple[State, str, MagicMock]:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=load_data)
    state.add_tab(
        tab_id,
        Session(
            adapter_name="any",
            adapter=adapter,
            cfg=make_form(Form(ExpCfgModel, {}), state=state),
        ),
    )
    return state, tab_id, adapter


def _service(state: State) -> tuple[LoadService, MagicMock, MagicMock]:
    bus = EventBus()
    emit = MagicMock()
    bus.emit = emit  # type: ignore[method-assign]
    writeback = MagicMock()
    context = MagicMock()
    context.read_context.side_effect = lambda: state.session_env.parameters
    return (
        LoadService(
            state,
            writeback,
            context=context,
            captured_parameters=lambda _aliases: state.session_env.parameters,
        ),
        emit,
        writeback,
    )


@pytest.mark.parametrize("context_free", [False, True])
def test_load_retains_request_and_history_without_foreign_analysis_fallback(
    result_sources: tuple[ParameterContext, ParameterContext],
    context_free: bool,
) -> None:
    state, tab_id, adapter = _make_state()
    q1, q2 = result_sources
    captured = None if context_free else q1
    tab = state.get_tab(tab_id)
    state.update_tab_result(tab_id, object(), parameters=q2, source_operation_id=91)
    state.set_context(replace(state.session_env, parameters=captured))
    loaded: RunRecord[ExpCfgModel, object] = RunRecord(cfg=None, result=object())
    metadata = historical_metadata()

    def load_and_rebind(
        _request: LoadDataRequest,
    ) -> tuple[RunRecord[ExpCfgModel, object], RunMetadata]:
        state.set_context(replace(state.session_env, parameters=q2))
        return loaded, metadata

    adapter.load.side_effect = load_and_rebind
    service, _emit, _writeback = _service(state)
    service.load_result(tab_id, "result.h5")

    request = adapter.load.call_args.args[0]
    assert isinstance(request, LoadDataRequest)
    assert request.data_path == "result.h5"
    assert request.parameters is captured
    if request.parameters is not None:
        assert request.parameters.value("source_qubit.rate") == 1.0
    assert tab.run.result is loaded
    assert tab.run.parameters is None
    assert tab.run.storage is None
    assert tab.run.source_operation_id is None
    assert tab.run.metadata is metadata
    state.clear_tab_results(tab_id)
    assert tab.run.parameters is None


@pytest.mark.parametrize("missing_cfg", [False, True])
def test_load_result_replaces_run_result_and_invalidates_dependents(
    missing_cfg: bool,
) -> None:
    state, tab_id, adapter = _make_state()
    stale_result = object()
    loaded = RunRecord(cfg=None if missing_cfg else ExpCfgModel(), result=object())
    metadata = historical_metadata()
    adapter.load.return_value = (loaded, metadata)
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

    outcome = svc.load_result(tab_id, "/tmp/new.hdf5")

    assert tab.run.result is loaded
    assert tab.run.metadata == metadata
    assert tab.run.confirmed_run_id is None
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
    assert outcome.cfg_backfill == ("not_applied" if missing_cfg else "applied")
    assert outcome.has_analyze_params is False
    assert state.version.get(f"tab:{tab_id}:result") == 1
    assert state.version.get(f"tab:{tab_id}:analyze") == 1
    assert state.version.get(f"tab:{tab_id}:post_analyze") == 1
    assert state.version.get(f"tab:{tab_id}:cfg") == cfg_version
    assert state.version.get(f"tab:{tab_id}:save_path") == save_path_version


def test_load_result_rejects_busy_tab_without_calling_adapter() -> None:
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, analyzing=True)
    svc, _emit, writeback = _service(state)

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.load_result(tab_id, "/tmp/new.hdf5")

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
        svc.load_result(tab_id, "/tmp/bad.hdf5")

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
        svc.load_result(tab_id, "/tmp/file.hdf5")

    assert exc_info.value.reason_code == "unsupported_load"
    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    writeback.teardown_draft.assert_not_called()
