from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.run_saving import ArtifactRef
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ContextReadiness,
    Form,
    ParameterWriteback,
    SessionEnv,
)
from zcu_tools.gui.app.measure.artifact_tracker import SaveStatus
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.services.writeback import (
    WritebackDraft,
    WritebackService,
)
from zcu_tools.gui.app.measure.state import (
    Session,
    State,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedErrorCategory
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
    PaneSummary,
    PostSummary,
    native_save_session,
)
from tests.gui.app.measure._pane_adapter import (
    PaneAdapter,
    PaneParams,
    PaneResult,
    PostResult,
)


@pytest.fixture
def native_session(tmp_path, qapp):
    with native_save_session(tmp_path, qapp) as session:
        yield session


@dataclass
class _Draft:
    items: tuple[Any, ...]


class _Writeback:
    def __init__(self) -> None:
        self.created: list[_Draft] = []
        self.torn_down: list[_Draft] = []
        self.fail_teardown = False
        self.fail_create = False

    def create_draft(
        self,
        items: list[Any],
        *,
        parameters: ParameterContext | None,
        experiment_name: str,
        run_id: str | None = None,
        device_snapshot: Mapping[str, JsonObject] | None = None,
    ) -> _Draft:
        if self.fail_create:
            raise RuntimeError("draft creation failed")
        draft = _Draft(tuple(items))
        self.created.append(draft)
        return draft

    def preview_draft(self, draft: _Draft) -> list[Any]:
        return list(draft.items)

    def teardown_draft(self, draft: _Draft) -> None:
        self.torn_down.append(draft)
        if self.fail_teardown:
            raise RuntimeError("teardown failed")


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
        readiness=ContextReadiness.ACTIVE,
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
    state.update_tab_result("tab", "run", parameters=ctx.parameters)
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
    return AnalyzeService(state, runner, bus, cast(Any, writeback), handles), bg


def test_snapshot_exposes_independent_panes_and_paths(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    state, tab_id = session.state, "tab"
    run_result = state.get_tab(tab_id).run.result
    primary, post = PaneSummary(), PostSummary()
    primary_plots = Plots(NonPresentingHost())
    primary_plots.subplots("fit")
    primary_plots.finish()
    post_plots = Plots(NonPresentingHost())
    post_plots.subplots("fit")
    post_plots.finish()
    primary_draft, post_draft = MagicMock(), MagicMock()
    state.replace_analysis_pane(
        tab_id,
        result=primary,
        plots=primary_plots,
        params="primary-params",
        writeback_draft=primary_draft,
    )
    state.replace_post_analysis_pane(
        tab_id,
        result=post,
        plots=post_plots,
        params="post-params",
        writeback_draft=post_draft,
    )
    writeback = MagicMock()
    writeback.preview_draft.side_effect = lambda draft: (
        ["primary-item"] if draft is primary_draft else ["post-item"]
    )
    tabs = TabService(
        state, MagicMock(), writeback, form_resources(state), MagicMock(), MagicMock()
    )
    snapshot = tabs.get_snapshot(tab_id)
    assert snapshot.run is not None and snapshot.run.result is run_result
    assert snapshot.analysis is not None and snapshot.analysis.result is primary
    assert snapshot.post_analysis is not None and snapshot.post_analysis.result is post
    assert snapshot.analysis.writeback_items == ("primary-item",)
    assert snapshot.post_analysis.writeback_items == ("post-item",)
    assert [
        artifact.ref for section in snapshot.artifacts for artifact in section.artifacts
    ] == [
        ArtifactRef("run", "measurement_data"),
        ArtifactRef("analysis", "results"),
        ArtifactRef("post_analysis", "results"),
    ]
    assert all(
        artifact.status is SaveStatus.NOT_SAVED
        for section in snapshot.artifacts
        for artifact in section.artifacts
    )
    report = session.save_artifacts()
    saved = tabs.get_snapshot(tab_id)
    assert {
        member.key: tuple(
            (output.path, output.status, output.error) for output in member.outputs
        )
        for section in saved.artifacts
        for artifact in section.artifacts
        for member in artifact.members
    } == {
        key: tuple(
            (item.output.path, item.status, item.error)
            for item in report.outputs
            if item.key == key
        )
        for key in {item.key for item in report.outputs}
    }
    assert all(
        artifact.status is SaveStatus.SAVED
        for section in saved.artifacts
        for artifact in section.artifacts
    )
    primary_plots.release()
    post_plots.release()


def test_named_panes_project_paths_and_retire_original_collections(
    native_session: NativeSaveSession,
) -> None:
    session = native_session
    state, tab_id = session.state, "tab"
    original = Plots(NonPresentingHost())
    old_figure, _ = original.subplots("old")
    original.finish()
    state.replace_analysis_pane(tab_id, result=PaneSummary(), plots=original)
    replacement = Plots(NonPresentingHost())
    first, _ = replacement.subplots("fit")
    second, _ = replacement.subplots("diagnostic")
    replacement.finish()
    retired = state.replace_analysis_pane(
        tab_id, result=PaneSummary(), plots=replacement
    )
    assert retired.plots == (original,)
    retired.plots[0].release()
    png = BytesIO()
    old_figure.savefig(png, format="png")
    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    post = Plots(NonPresentingHost())
    post_figure, _ = post.subplots("fit")
    post.finish()
    state.replace_post_analysis_pane(tab_id, result=PostSummary(), plots=post)
    tabs = TabService(
        state, MagicMock(), MagicMock(), form_resources(state), MagicMock(), MagicMock()
    )
    snapshot = tabs.get_snapshot(tab_id)
    assert (
        snapshot.analysis is not None
        and snapshot.analysis.figures is not None
        and snapshot.analysis.figures is replacement
    )
    assert (
        snapshot.post_analysis is not None
        and snapshot.post_analysis.figures is not None
        and snapshot.post_analysis.figures is post
    )
    assert snapshot.analysis.figures["fit"] is first
    assert snapshot.analysis.figures["diagnostic"] is second
    assert snapshot.post_analysis.figures["fit"] is post_figure
    assert [
        member.key
        for section in snapshot.artifacts
        for artifact in section.artifacts
        for member in artifact.members
    ] == [
        ArtifactKey("run", "measurement_data", "data"),
        ArtifactKey("analysis", "results", "figure", "fit"),
        ArtifactKey("analysis", "results", "figure", "diagnostic"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
        ArtifactKey("post_analysis", "results", "figure", "fit"),
        ArtifactKey("post_analysis", "results", "analysis", "post_analysis"),
    ]
    session.save_artifacts()
    assert all(
        artifact.status is SaveStatus.SAVED
        for section in tabs.get_snapshot(tab_id).artifacts
        for artifact in section.artifacts
    )
    newer = Plots(NonPresentingHost())
    newer.subplots("fit")
    newer.finish()
    detached = state.replace_analysis_pane(tab_id, result=PaneSummary(), plots=newer)
    assert detached.plots == (replacement, post)
    assert state.get_tab(tab_id).post_analysis.plots is None
    current = tabs.get_snapshot(tab_id)
    assert [
        artifact.ref for section in current.artifacts for artifact in section.artifacts
    ] == [
        ArtifactRef("run", "measurement_data"),
        ArtifactRef("analysis", "results"),
    ]
    analysis = current.artifacts[1].artifacts[0]
    assert analysis.status is SaveStatus.NOT_SAVED
    assert [member.key for member in analysis.members] == [
        ArtifactKey("analysis", "results", "figure", "fit"),
        ArtifactKey("analysis", "results", "analysis", "analysis"),
    ]
    assert all(
        output.status == "reserved"
        for member in analysis.members
        for output in member.outputs
    )
    replacement.release()
    post.release()
    newer.release()


def test_primary_swap_returns_all_retired_dependents_and_invalidates_post(
    native_session: NativeSaveSession,
) -> None:
    state, tab_id = native_session.state, "tab"
    old_primary = object()
    old_post = object()
    old_primary_draft = object()
    old_post_draft = object()
    old_primary_plots = Plots(NonPresentingHost())
    old_primary_plots.subplots("fit")
    old_primary_plots.finish()
    old_post_plots = Plots(NonPresentingHost())
    old_post_plots.subplots("fit")
    old_post_plots.finish()
    new_plots = Plots(NonPresentingHost())
    new_plots.subplots("fit")
    new_plots.finish()
    state.replace_analysis_pane(
        tab_id,
        result=old_primary,
        plots=old_primary_plots,
        writeback_draft=old_primary_draft,
    )
    state.replace_post_analysis_pane(
        tab_id,
        result=old_post,
        plots=old_post_plots,
        writeback_draft=old_post_draft,
    )

    retired = state.replace_analysis_pane(
        tab_id,
        result="new-primary",
        plots=new_plots,
        writeback_draft="new-primary-draft",
    )

    tab = state.get_tab(tab_id)
    assert retired.analysis.result is old_primary
    assert retired.analysis.plots is old_primary_plots
    assert retired.analysis.writeback_draft is old_primary_draft
    assert retired.post_analysis.result is old_post
    assert retired.post_analysis.plots is old_post_plots
    assert retired.post_analysis.writeback_draft is old_post_draft
    assert retired.writeback_drafts == (old_primary_draft, old_post_draft)
    assert tab.analysis.result == "new-primary"
    assert tab.analysis.plots is new_plots
    assert tab.analysis.writeback_draft == "new-primary-draft"
    assert tab.post_analysis.result is None
    assert tab.post_analysis.writeback_draft is None


def test_post_swap_does_not_replace_primary() -> None:
    state, tab_id, _adapter, _ctx = _state()
    state.replace_analysis_pane(tab_id, result="primary", plots=None)
    primary_pane = state.get_tab(tab_id).analysis

    state.replace_post_analysis_pane(tab_id, result="post", plots=None)

    tab = state.get_tab(tab_id)
    assert tab.analysis is primary_pane
    assert tab.analysis.result == "primary"
    assert tab.post_analysis.result == "post"


def test_analyze_uses_captured_inputs_and_cleans_retired_after_commit() -> None:
    state, tab_id, adapter, _ctx = _state()
    old_primary_draft = _Draft(())
    old_post_draft = _Draft(())
    state.replace_analysis_pane(
        tab_id, result="old-primary", plots=None, writeback_draft=old_primary_draft
    )
    state.replace_post_analysis_pane(
        tab_id, result="old-post", plots=None, writeback_draft=old_post_draft
    )
    adapter.get_writeback_items.return_value = [ParameterWriteback("new", "new", 1.0)]
    writeback = _Writeback()
    service, bg = _analyze_service(state, writeback)
    plots = Plots(NonPresentingHost())
    plots.subplots("fit")
    captured_run_parameters = state.get_tab(tab_id).run.parameters
    assert captured_run_parameters is not None
    service.start_analyze(tab_id, "params", plots=plots)

    new_ctx = SessionEnv(parameters=parameter_capture(), soc=None, soccfg=None)
    state.set_context(new_ctx)
    state.get_tab(tab_id).run.result = "changed-after-start"
    result = object()
    bg.on_done(result)

    request = adapter.get_writeback_items.call_args.args[0]
    assert request.run_result == "run"
    assert request.ctx.parameters is captured_run_parameters
    assert state.get_tab(tab_id).analysis.result is result
    assert state.get_tab(tab_id).analysis.plots is plots
    assert tuple(plots) == ("fit",)
    assert state.get_tab(tab_id).post_analysis.result is None
    assert old_primary_draft in writeback.torn_down
    assert old_post_draft in writeback.torn_down


def _pane_source_with_excited_initial_state() -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(
        None, SavingResult(np.array([1.0, 2.0]), np.array([1 + 0j, 2 + 0j]))
    )


def _pane_capture(original: ParameterContext | None) -> ParameterContext:
    if original is None:
        raise RuntimeError("Native fixture must retain its Run parameter capture")
    return ParameterContext(
        point=original.document().model_copy(
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
    adapter = PaneAdapter()
    state.get_tab(tab_id).adapter = adapter
    state.update_tab_result(
        tab_id,
        _pane_source_with_excited_initial_state(),
        metadata=replace(
            native_session.metadata, experiment=SavingCore().persistence_spec().tag
        ),
        parameters=_pane_capture(state.get_tab(tab_id).run.parameters),
    )
    bus = EventBus()
    writeback = WritebackService(MagicMock(), MagicMock())
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
    post_params = adapter.get_post_analyze_params(primary, ctx)
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
            writeback.teardown_draft(current_draft)


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
    assert {
        item.target_name for item in panes.writeback.preview_draft(primary_draft)
    } == {
        "$qubit.ext.readout_cal.fidelity",
        "$qubit.ext.readout_cal.sigma",
        "$qubit.ext.readout_cal.g_center",
        "$qubit.ext.readout_cal.e_center",
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
    assert {item.target_name for item in panes.writeback.preview_draft(post_draft)} == {
        "$qubit.ext.readout_cal.radius",
        "$qubit.ext.readout_cal.confusion_matrix",
    }
    snapshot = TabService(
        panes.state,
        MagicMock(),
        panes.writeback,
        form_resources(panes.state),
        MagicMock(),
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
    state.replace_analysis_pane(tab_id, result="old-primary", plots=None)
    state.replace_post_analysis_pane(tab_id, result="old-post", plots=None)
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


def test_retired_teardown_failure_does_not_roll_back_committed_pane() -> None:
    state, tab_id, adapter, _ctx = _state()
    old_primary_draft = _Draft(())
    old_post_draft = _Draft(())
    state.replace_analysis_pane(
        tab_id, result="old-primary", plots=None, writeback_draft=old_primary_draft
    )
    state.replace_post_analysis_pane(
        tab_id, result="old-post", plots=None, writeback_draft=old_post_draft
    )
    adapter.get_writeback_items.return_value = []
    writeback = _Writeback()
    writeback.fail_teardown = True
    service, bg = _analyze_service(state, writeback)
    plots = Plots(NonPresentingHost())
    plots.subplots("diagnostic")
    service.start_analyze(tab_id, "params", plots=plots)
    new_result = object()

    bg.on_done(new_result)

    tab = state.get_tab(tab_id)
    assert tab.analysis.result is new_result
    assert tab.analysis.plots is plots
    assert tab.analysis.writeback_draft is writeback.created[-1]
    assert tab.post_analysis.result is None
    assert tab.post_analysis.writeback_draft is None
    assert writeback.torn_down == [old_primary_draft, old_post_draft]


def test_load_capability_gate_rejects_concrete_disabled_adapter() -> None:
    state, tab_id, _adapter, _ctx = _state()

    class DisabledAdapter:
        capabilities = AdapterCapabilities(load_data=False)

        def load(self, _request: object) -> object:
            return object()

    adapter = DisabledAdapter()
    state.get_tab(tab_id).adapter = adapter  # type: ignore[assignment]
    load = LoadService(
        state,
        MagicMock(),
        context=MagicMock(),
        captured_parameters=lambda _aliases: state.session_env.parameters,
    )

    with pytest.raises(LoadDataError) as exc_info:
        load.load_result(tab_id, "/tmp/x")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == "unsupported_load"


def test_guard_and_load_accept_enabled_concrete_adapter() -> None:
    state, tab_id, _adapter, _ctx = _state()

    class EnabledAdapter:
        capabilities = AdapterCapabilities(load_data=True)

        def load(self, _request: object) -> object:
            return object()

    state.get_tab(tab_id).adapter = EnabledAdapter()  # type: ignore[assignment]
    GuardService(state).check_load(tab_id)
    assert state.get_tab(tab_id).run.result == "run"
