from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from pydantic import Field, JsonValue
from zcu_tools.datafile import StoredRun
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.native_persistence import RestoredRun
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, NoAnalyzeParams
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedErrorCategory, FailedPreconditionError
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._load_support import historical_metadata, write_native_file


@pytest.fixture
def native_path(tmp_path: Path) -> str:
    path = tmp_path / "data.h5"
    write_native_file(path)
    return str(path)


def _make_state(*, load_data: bool = True) -> tuple[State, str, MagicMock]:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=load_data)
    adapter.experiment_tag = "test/load"
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
    bus.emit = emit
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
def test_load_retains_history_without_foreign_analysis_fallback(
    result_sources: tuple[ParameterContext, ParameterContext],
    context_free: bool,
    native_path: str,
) -> None:
    state, tab_id, adapter = _make_state()
    q1, q2 = result_sources
    captured = None if context_free else q1
    tab = state.get_tab(tab_id)
    state.update_tab_result(tab_id, object(), parameters=q2, source_operation_id=91)
    state.set_context(replace(state.session_env, parameters=captured))
    loaded: RunRecord[ExpCfgModel, object] = RunRecord(cfg=None, result=object())
    metadata = historical_metadata()

    def restore_and_rebind(
        stored: StoredRun,
    ) -> RestoredRun[RunRecord[ExpCfgModel, object]]:
        assert stored.metadata == metadata
        state.set_context(replace(state.session_env, parameters=q2))
        return RestoredRun(loaded, "cfg unavailable")

    adapter.restore.side_effect = restore_and_rebind
    service, _emit, _writeback = _service(state)
    service.load_result(tab_id, native_path)

    assert tab.run.result is loaded
    assert tab.run.parameters is None
    assert tab.run.storage is None
    assert tab.run.source_operation_id is None
    assert tab.run.metadata == metadata
    state.clear_tab_results(tab_id)
    assert tab.run.parameters is None


@pytest.mark.parametrize("missing_cfg", [False, True])
def test_load_result_replaces_run_result_and_invalidates_dependents(
    missing_cfg: bool, native_path: str
) -> None:
    state, tab_id, adapter = _make_state()
    stale_result = object()
    loaded = RunRecord(cfg=None if missing_cfg else ExpCfgModel(), result=object())
    metadata = historical_metadata()
    issue = "cfg unavailable" if missing_cfg else None
    adapter.restore.return_value = RestoredRun(loaded, issue)
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

    outcome = svc.load_result(tab_id, native_path)

    assert tab.run.result is loaded
    assert tab.run.metadata == metadata
    assert tab.run.confirmed_run_id is None
    assert tab.run.source_path == native_path
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
    assert outcome.cfg_issue == issue
    assert outcome.has_analyze_params is False
    assert state.version.get(f"tab:{tab_id}:result") == 1
    assert state.version.get(f"tab:{tab_id}:analyze") == 1
    assert state.version.get(f"tab:{tab_id}:post_analyze") == 1
    assert state.version.get(f"tab:{tab_id}:cfg") == cfg_version
    assert state.version.get(f"tab:{tab_id}:save_path") == save_path_version


class RepeatsCfg(ExpCfgModel):
    """A result snapshot and the editable draft can have different repeats."""

    reps: int = Field(default=20, ge=1)


@pytest.mark.parametrize("draft", [7, 0, {"expr": "unfinished +"}])
def test_load_preserves_authored_input_including_invalid_drafts(
    draft: JsonValue, native_path: str
) -> None:
    state, tab_id, adapter = _make_state()
    tab = state.get_tab(tab_id)
    tab.cfg.revoke()
    tab.cfg = make_form(Form(RepeatsCfg, {}), state=state)
    tab.cfg.write({("reps",): draft})
    before = tab.cfg.input.data
    env = tab.cfg.env
    cfg_version = state.version.get(f"tab:{tab_id}:cfg")
    loaded = RunRecord(RepeatsCfg(reps=40), object())
    metadata = historical_metadata()
    adapter.restore.return_value = RestoredRun(loaded, None)
    service, _emit, _writeback = _service(state)

    try:
        outcome = service.load_result(tab_id, native_path)

        assert tab.cfg.input.data == before
        assert tab.cfg.env is env
        assert state.version.get(f"tab:{tab_id}:cfg") == cfg_version
        assert tab.run.result is loaded
        assert loaded.cfg is not None
        assert loaded.cfg.reps == 40
        assert tab.run.metadata == metadata
        assert tab.run.source_path == native_path
        assert outcome.cfg_issue is None
    finally:
        tab.cfg.revoke()


def test_reentrant_load_does_not_replace_result() -> None:
    state, tab_id, adapter = _make_state()
    tab = state.get_tab(tab_id)
    service, _emit, _writeback = _service(state)
    failures: list[CfgPreconditionReason] = []

    def callback() -> None:
        try:
            service.load_result(tab_id, "result.h5")
        except CfgPreconditionError as exc:
            failures.append(exc.reason)

    unsubscribe = tab.cfg.subscribe(callback)
    try:
        tab.cfg.reset()
    finally:
        unsubscribe()
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert tab.run.result is None
    adapter.restore.assert_not_called()


def test_load_result_rejects_busy_tab_without_calling_adapter() -> None:
    state, tab_id, adapter = _make_state()
    state.set_tab_analyzing(tab_id, analyzing=True)
    svc, _emit, writeback = _service(state)

    with pytest.raises(FailedPreconditionError, match="busy") as exc_info:
        svc.load_result(tab_id, "/tmp/new.hdf5")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == ""
    adapter.restore.assert_not_called()
    writeback.teardown_draft.assert_not_called()
    assert state.get_tab(tab_id).run.result is None


def test_load_result_error_leaves_state_unchanged(native_path: str) -> None:
    state, tab_id, adapter = _make_state()
    old = object()
    state.get_tab(tab_id).run.result = old
    adapter.restore.side_effect = ValueError("bad file")
    svc, _emit, writeback = _service(state)

    with pytest.raises(LoadDataError, match="Cannot load this data file") as exc_info:
        svc.load_result(tab_id, native_path)

    assert exc_info.value.reason_code == "invalid_data_file"
    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert "Details: bad file" in str(exc_info.value)
    writeback.teardown_draft.assert_not_called()
    assert state.get_tab(tab_id).run.result is old
    assert state.version.get(f"tab:{tab_id}:result") == 0


def test_load_result_wraps_unsupported_adapter(native_path: str) -> None:
    state, tab_id, adapter = _make_state()
    adapter.restore.side_effect = NotImplementedError("unsupported")
    svc, _emit, writeback = _service(state)

    with pytest.raises(LoadDataError, match="does not support loading") as exc_info:
        svc.load_result(tab_id, native_path)

    assert exc_info.value.reason_code == "unsupported_load"
    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    writeback.teardown_draft.assert_not_called()
