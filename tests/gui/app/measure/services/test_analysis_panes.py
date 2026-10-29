"""Analysis operations publish captured results, plots and writeback proposals."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    ContextReadiness,
    SessionEnv,
)
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.pane_commit import (
    commit_analysis,
    commit_post_analysis,
    commit_run,
)
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import (
    AnalysisPaneState,
    PostAnalysisPaneState,
    RunPaneState,
    Session,
    State,
)
from zcu_tools.gui.app.measure.writeback import (
    WritebackDraft,
    WritebackItem,
    WritebackService,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.entry import ArtifactKey, JsonObject

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import (
    parameter_capture,
)
from tests.gui.app.measure._components import QubitSchema
from tests.gui.app.measure._form_fakes import form_resources, make_form
from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    native_save_session,
)
from tests.gui.app.measure._pane_adapter import (
    PaneAdapter,
    PaneParams,
    PaneResult,
    PostResult,
)
from tests.gui.app.measure._record_support import require_record


@pytest.fixture
def native_session(tmp_path, qapp):
    with native_save_session(tmp_path, qapp) as session:
        yield session


class _Writeback(WritebackService):
    def __init__(self) -> None:
        super().__init__(MagicMock(), arb_names=lambda: ())
        self.created: list[WritebackDraft] = []
        self.fail_create = False

    def create_draft(
        self,
        items: Iterable[WritebackItem],
        *,
        parameters: ParameterContext | None,
        experiment_name: str,
        run_id: str | None = None,
        device_snapshot: (
            Mapping[str, JsonObject]
            | Callable[[], Mapping[str, JsonObject] | None]
            | None
        ) = None,
    ) -> WritebackDraft:
        if self.fail_create:
            raise RuntimeError("draft creation failed")
        draft = super().create_draft(
            items,
            parameters=parameters,
            experiment_name=experiment_name,
            run_id=run_id,
            device_snapshot=device_snapshot,
        )
        self.created.append(draft)
        return draft


class _Bg:
    def submit(
        self, work: Any, *, run_in_pool: bool, on_done: Any, on_error: Any
    ) -> None:
        self.work = work
        self.on_done = on_done
        self.on_error = on_error


def _state() -> tuple[State, str, MagicMock, SessionEnv]:
    ctx = SessionEnv(
        parameters=parameter_capture(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        database_path="/db",
        result_dir="/result",
        active_label="ctx",
        readiness=ContextReadiness.POINT,
    )
    state = State(ctx)
    adapter = MagicMock()
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg=make_form(Form(SavingCfg, {}), state=state),
        ),
    )
    commit_run(
        state,
        "tab",
        RunPaneState(
            result=RunRecord(None, "run"),
            storage=None if ctx.parameters is None else ctx.parameters.storage,
            parameters=ctx.parameters,
        ),
    )
    return state, "tab", adapter, ctx


def _analyze_service(
    state: State, writeback: _Writeback | WritebackService, bus: EventBus | None = None
) -> tuple[AnalyzeService, _Bg]:
    bus = bus or EventBus()
    handles = OperationHandles()
    bg = _Bg()
    runner = OperationRunner(
        MagicMock(), handles, ProgressService(DirectProgressTransport()), bg, bus
    )
    return AnalyzeService(state, runner, bus, writeback, handles), bg


def test_analyze_publishes_result_from_captured_inputs() -> None:
    state, tab_id, adapter, _ctx = _state()
    writeback = _Writeback()
    commit_analysis(state, tab_id, AnalysisPaneState(result="old-primary"))
    commit_post_analysis(state, tab_id, PostAnalysisPaneState(result="old-post"))
    adapter.get_writeback_items.return_value = []
    service, bg = _analyze_service(state, writeback)
    plots = Plots(NonPresentingHost())
    plots.subplots("fit")
    captured_run_parameters = state.get_tab(tab_id).run.parameters
    assert captured_run_parameters is not None
    service.start_analyze(tab_id, "params", plots=plots)

    new_ctx = SessionEnv(parameters=parameter_capture(), soc=None, soccfg=None)
    state.set_context(new_ctx)
    state.get_tab(tab_id).run.result = RunRecord(None, "changed-after-start")
    result = object()
    bg.on_done(result)

    request = adapter.get_writeback_items.call_args.args[0]
    assert request.run_result.result == "run"
    assert request.ctx.parameters is captured_run_parameters
    assert state.get_tab(tab_id).analysis.result is result
    assert state.get_tab(tab_id).analysis.plots is plots
    assert tuple(plots) == ("fit",)
    assert state.get_tab(tab_id).post_analysis.result is None


def _pane_source_with_excited_initial_state() -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(
        None, SavingResult(np.array([1.0, 2.0]), np.array([1 + 0j, 2 + 0j]))
    )


def _pane_capture(original: ParameterContext | None) -> ParameterContext:
    if original is None:
        raise RuntimeError("Native fixture must retain its Run parameter capture")
    return ParameterContext(
        point=original.whole_document().model_copy(
            update={"components": {"Q1": QubitSchema(kind="qubit/fluxonium")}}
        ),
        library=original.library,
        target=original.target,
        roles=None,
        storage=original.storage,
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})


@dataclass
class _AnalysisPanes:
    """Adopted synthetic panes and their replacement operation seam.

    state/tab_id locate the live tab. writeback previews and retires pane drafts.
    primary_service/primary_bg admit and complete replacement fits.
    fit_plots/post_plots are the original named collections, retained for exports.
    """

    state: State
    tab_id: str
    writeback: WritebackService
    primary_service: AnalyzeService
    primary_bg: _Bg
    fit_plots: Plots
    post_plots: Plots


@pytest.fixture
def analysis_panes(native_session: NativeSaveSession):
    state, tab_id, ctx = native_session.state, "tab", native_session.state.session_env
    adapter = PaneAdapter
    state.get_tab(tab_id).adapter = adapter
    commit_run(
        state,
        tab_id,
        RunPaneState(
            result=replace(
                _pane_source_with_excited_initial_state(),
                metadata=native_session.metadata.model_copy(
                    update={"experiment": SavingCore.AXES_SPEC.tag}
                ),
            ),
            storage=_pane_capture(state.get_tab(tab_id).run.parameters).storage,
            parameters=_pane_capture(state.get_tab(tab_id).run.parameters),
        ),
    )
    bus = EventBus()
    writeback = WritebackService(MagicMock(), arb_names=lambda: ())
    primary_service, primary_bg = _analyze_service(state, writeback, bus)
    params = PaneParams(initial_state="excited", length_ratio=0.01)
    fit_plots = Plots(NonPresentingHost())
    primary_service.start_analyze(tab_id, params, plots=fit_plots)
    primary_bg.on_done(primary_bg.work())

    tab = state.get_tab(tab_id)
    primary = tab.analysis.result
    assert isinstance(primary, PaneResult)
    # The post operation must use the adopted primary, not the later form edit.
    params.initial_state = "ground"
    post_params = adapter.get_post_analyze_params(require_record(tab.run), primary, ctx)
    post_bg = _Bg()
    post_handles = OperationHandles()
    post_runner = OperationRunner(
        MagicMock(),
        post_handles,
        ProgressService(DirectProgressTransport()),
        post_bg,
        bus,
    )
    post_service = PostAnalyzeService(state, post_runner, bus, post_handles, writeback)
    post_plots = Plots(NonPresentingHost())
    post_service.start_post_analyze(tab_id, post_params, plots=post_plots)
    post_bg.on_done(post_bg.work())

    try:
        yield _AnalysisPanes(
            state, tab_id, writeback, primary_service, primary_bg, fit_plots, post_plots
        )
    finally:
        fit_plots.release()
        post_plots.release()
        current_draft = tab.analysis.writeback_draft
        if isinstance(current_draft, WritebackDraft):
            current_draft.teardown()


def test_pane_primary_publishes_excited_fit_and_calibration_writeback(
    analysis_panes: _AnalysisPanes,
) -> None:
    panes = analysis_panes
    tab = panes.state.get_tab(panes.tab_id)
    primary = tab.analysis.result
    assert isinstance(primary, PaneResult)
    assert primary.initial_state == "excited"
    assert tab.analysis.plots is panes.fit_plots
    assert tuple(panes.fit_plots) == ("fit",)
    primary_draft = tab.analysis.writeback_draft
    assert isinstance(primary_draft, WritebackDraft)
    assert {item.target_name for item in primary_draft.preview()} == {
        "$qubit.ext.pane_sample.score",
        "$qubit.ext.pane_sample.width",
        "$qubit.ext.pane_sample.first",
        "$qubit.ext.pane_sample.second",
    }


def test_pane_post_uses_adopted_fit_and_projects_separate_artifacts_and_writeback(
    analysis_panes: _AnalysisPanes,
) -> None:
    panes = analysis_panes
    tab = panes.state.get_tab(panes.tab_id)
    post = tab.post_analysis.result
    assert isinstance(post, PostResult)
    np.testing.assert_allclose(post.confusion, np.eye(3), atol=0.06)
    assert tab.post_analysis.plots is panes.post_plots
    assert tuple(panes.post_plots) == ("post",)
    post_draft = tab.post_analysis.writeback_draft
    assert isinstance(post_draft, WritebackDraft)
    assert {item.target_name for item in post_draft.preview()} == {
        "$qubit.ext.pane_sample.extent",
        "$qubit.ext.pane_sample.matrix",
    }
    snapshot = TabService(
        panes.state,
        MagicMock(),
        form_resources(panes.state),
        MagicMock(),
    ).get_snapshot(panes.tab_id)
    assert snapshot.analysis is not None and snapshot.post_analysis is not None
    assert snapshot.analysis.figures is panes.fit_plots
    assert snapshot.post_analysis.figures is panes.post_plots
    assert {
        member.key
        for section in snapshot.artifacts
        for artifact in section.artifacts
        for member in artifact.members
    } >= {
        ArtifactKey("analysis", "results", "figure", "fit"),
        ArtifactKey("post_analysis", "results", "figure", "post"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
        ArtifactKey("post_analysis", "results", "analysis", "post_analysis"),
    }


def test_pane_primary_replacement_clears_post_but_retained_plots_still_export(
    analysis_panes: _AnalysisPanes,
) -> None:
    panes = analysis_panes
    tab = panes.state.get_tab(panes.tab_id)
    new_fit = Plots(NonPresentingHost())
    try:
        panes.primary_service.start_analyze(
            panes.tab_id, PaneParams(initial_state="excited"), plots=new_fit
        )
        panes.primary_bg.on_done(panes.primary_bg.work())
        assert tab.post_analysis.result is None
        assert tab.post_analysis.plots is None
        old_fit, old_post = BytesIO(), BytesIO()
        panes.fit_plots["fit"].savefig(old_fit, format="png")
        panes.post_plots["post"].savefig(old_post, format="png")
        assert old_fit.getvalue().startswith(b"\x89PNG")
        assert old_post.getvalue().startswith(b"\x89PNG")
        current_draft = tab.analysis.writeback_draft
        assert isinstance(current_draft, WritebackDraft)
    finally:
        new_fit.release()


def test_failed_draft_build_preserves_previous_primary_and_post() -> None:
    state, tab_id, adapter, _ctx = _state()
    adapter.get_writeback_items.return_value = []
    commit_analysis(state, tab_id, AnalysisPaneState(result="old-primary", plots=None))
    commit_post_analysis(
        state, tab_id, PostAnalysisPaneState(result="old-post", plots=None)
    )
    writeback = _Writeback()
    writeback.fail_create = True
    service, bg = _analyze_service(state, writeback)
    plots = Plots(NonPresentingHost())
    plots.subplots("fit")
    service.start_analyze(tab_id, "params", plots=plots)

    bg.on_done(object())

    tab = state.get_tab(tab_id)
    assert tab.analysis.result == "old-primary"
    assert tab.post_analysis.result == "old-post"
    assert writeback.created == []
