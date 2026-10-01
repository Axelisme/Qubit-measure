from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.singleshot.ge import GE_Cfg, GE_Result
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.ge import (
    GEAdapter,
    GEAnalyzeParams,
    GEAnalyzeResult,
    GEPostAnalyzeResult,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ContextReadiness,
    MetaDictWriteback,
    SavePaths,
    SessionEnv,
)
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind
from zcu_tools.gui.app.measure.services.analyze import AnalyzeService
from zcu_tools.gui.app.measure.services.guard import (
    AnalyzePermit,
    GuardService,
    LoadPermit,
)
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.services.post_analyze import PostAnalyzeService
from zcu_tools.gui.app.measure.services.tab import TabService
from zcu_tools.gui.app.measure.state import (
    Session,
    State,
)
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import ExpectedErrorCategory
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import cfg_resources, make_cfg


@dataclass
class _Draft:
    items: tuple[Any, ...]


class _Writeback:
    def __init__(self) -> None:
        self.created: list[_Draft] = []
        self.torn_down: list[_Draft] = []
        self.fail_teardown = False
        self.fail_create = False

    def create_draft(self, items: list[Any]) -> _Draft:
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
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        database_path="/db",
        result_dir="/result",
        active_label="ctx",
        readiness=ContextReadiness.ACTIVE,
    )
    state = State(ctx)
    adapter = MagicMock()
    adapter.make_save_paths.return_value = SavePaths("/db/data.h5", "/result/base.png")
    state.add_tab(
        "tab",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg=make_cfg(CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue())),
        ),
    )
    state.update_tab_result("tab", "run")
    return state, "tab", adapter, ctx


def _analyze_service(
    state: State, writeback: _Writeback, bus: EventBus | None = None
) -> tuple[AnalyzeService, _Bg]:
    bus = bus or EventBus()
    handles = OperationHandles()
    bg = _Bg()
    runner = OperationRunner(
        MagicMock(), handles, ProgressService(DirectProgressTransport()), bg, bus
    )
    return AnalyzeService(state, runner, bus, cast(Any, writeback), handles), bg


def test_snapshot_exposes_independent_panes_and_paths() -> None:
    state, tab_id, adapter, _ctx = _state()
    primary = object()
    post = object()
    primary_plots = Plots(NonPresentingHost())
    primary_plots.subplots("fit")
    primary_plots.finish()
    post_plots = Plots(NonPresentingHost())
    post_plots.subplots("fit")
    post_plots.finish()
    state.replace_analysis_pane(
        tab_id,
        result=primary,
        plots=primary_plots,
        params="primary-params",
        writeback_draft="primary-draft",
    )
    state.replace_post_analysis_pane(
        tab_id,
        result=post,
        plots=post_plots,
        params="post-params",
        writeback_draft="post-draft",
    )
    state.update_tab_data_path_override(tab_id, "/custom/data.h5")
    state.update_tab_image_path_override(
        tab_id, ArtifactKey(ArtifactKind.ANALYSIS, "fit"), "/custom/a.png"
    )
    state.update_tab_image_path_override(
        tab_id, ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit"), "/custom/p.png"
    )

    # Final contract: TabService composes pane-owned drafts via preview_draft.
    primary_draft = MagicMock()
    post_draft = MagicMock()
    writeback = MagicMock()
    writeback.preview_draft.side_effect = lambda d: (
        ["primary-item"]
        if d is primary_draft
        else ["post-item"]
        if d is post_draft
        else []
    )
    # Attach drafts to panes so snapshot can preview them.
    state.get_tab(tab_id).analysis.writeback_draft = primary_draft  # type: ignore[assignment]
    state.get_tab(tab_id).post_analysis.writeback_draft = post_draft  # type: ignore[assignment]
    snapshot = TabService(
        state, MagicMock(), writeback, cfg_resources(state)
    ).get_snapshot(tab_id)

    assert snapshot.run is not None and snapshot.run.result == "run"
    assert snapshot.analysis is not None and snapshot.analysis.result is primary
    assert snapshot.post_analysis is not None and snapshot.post_analysis.result is post
    assert snapshot.analysis.writeback_items == ("primary-item",)
    assert snapshot.post_analysis.writeback_items == ("post-item",)
    assert snapshot.paths is not None
    assert snapshot.paths.data.path == "/custom/data.h5"
    assert snapshot.paths.analysis_images["fit"].path == "/custom/a.png"
    assert snapshot.paths.post_analysis_images["fit"].path == "/custom/p.png"
    assert adapter.make_save_paths.call_count == 0


def test_named_panes_project_paths_and_retire_original_collections() -> None:
    state, tab_id, _adapter, _ctx = _state()
    original = Plots(NonPresentingHost())
    old_figure, _ = original.subplots("old")
    original.finish()
    state.replace_analysis_pane(tab_id, result="original", plots=original)
    replacement = Plots(NonPresentingHost())
    first, _ = replacement.subplots("fit")
    second, _ = replacement.subplots("diagnostic")
    replacement.finish()
    retired = state.replace_analysis_pane(tab_id, result="updated", plots=replacement)
    assert retired.plots == (original,)
    retired.plots[0].release()
    png = BytesIO()
    old_figure.savefig(png, format="png")
    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")

    post = Plots(NonPresentingHost())
    post_figure, _ = post.subplots("fit")
    post.finish()
    state.replace_post_analysis_pane(tab_id, result="post", plots=post)
    fit = ArtifactKey(ArtifactKind.ANALYSIS, "fit")
    diagnostic = ArtifactKey(ArtifactKind.ANALYSIS, "diagnostic")
    post_fit = ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit")
    state.update_tab_image_path_override(tab_id, diagnostic, "/custom/diagnostic.png")
    snapshot = TabService(
        state, MagicMock(), MagicMock(), cfg_resources(state)
    ).get_snapshot(tab_id)
    assert snapshot.analysis is not None and snapshot.analysis.figures is replacement
    assert snapshot.post_analysis is not None and snapshot.post_analysis.figures is post
    assert snapshot.analysis.figures is not None
    assert snapshot.post_analysis.figures is not None
    assert snapshot.analysis.figures["fit"] is first
    assert snapshot.analysis.figures["diagnostic"] is second
    assert snapshot.post_analysis.figures["fit"] is post_figure
    assert snapshot.paths is not None
    assert (
        snapshot.paths.analysis_images["fit"].path == "/result/base__analysis__fit.png"
    )
    assert snapshot.paths.analysis_images["diagnostic"].path == "/custom/diagnostic.png"
    assert (
        snapshot.paths.post_analysis_images["fit"].path == "/result/base__post__fit.png"
    )
    assert [item.key for item in snapshot.artifacts] == [
        ArtifactKey(ArtifactKind.DATA),
        fit,
        diagnostic,
        post_fit,
    ]

    newer = Plots(NonPresentingHost())
    newer.subplots("fit")
    newer.finish()
    detached = state.replace_analysis_pane(tab_id, result="newest", plots=newer)
    assert detached.plots == (replacement, post)
    assert state.get_tab(tab_id).post_analysis.plots is None
    current = TabService(
        state, MagicMock(), MagicMock(), cfg_resources(state)
    ).get_snapshot(tab_id)
    assert current.paths is not None
    assert tuple(current.paths.analysis_images) == ("fit",)
    assert (
        current.paths.analysis_images["fit"].path == "/result/base__analysis__fit.png"
    )
    assert [item.key for item in current.artifacts] == [
        ArtifactKey(ArtifactKind.DATA),
        fit,
    ]
    with pytest.raises(KeyError, match="No current image artifact"):
        state.update_tab_image_path_override(tab_id, diagnostic, "stale.png")


def test_primary_swap_returns_all_retired_dependents_and_invalidates_post() -> None:
    state, tab_id, _adapter, _ctx = _state()
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
    state, tab_id, adapter, ctx = _state()
    old_primary_draft = _Draft(())
    old_post_draft = _Draft(())
    state.replace_analysis_pane(
        tab_id, result="old-primary", plots=None, writeback_draft=old_primary_draft
    )
    state.replace_post_analysis_pane(
        tab_id, result="old-post", plots=None, writeback_draft=old_post_draft
    )
    adapter.get_writeback_items.return_value = [MetaDictWriteback("new", "new", 1.0)]
    writeback = _Writeback()
    service, bg = _analyze_service(state, writeback)
    plots = Plots(NonPresentingHost())
    plots.subplots("fit")
    service.start_analyze(AnalyzePermit(tab_id), "params", plots=plots)

    new_ctx = SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    state.set_context(new_ctx)
    state.get_tab(tab_id).run.result = "changed-after-start"
    result = object()
    bg.on_done(result)

    request = adapter.get_writeback_items.call_args.args[0]
    assert request.run_result == "run"
    assert request.ctx is ctx
    assert state.get_tab(tab_id).analysis.result is result
    assert state.get_tab(tab_id).analysis.plots is plots
    assert tuple(plots) == ("fit",)
    assert state.get_tab(tab_id).post_analysis.result is None
    assert old_primary_draft in writeback.torn_down
    assert old_post_draft in writeback.torn_down


def _ge_source_with_excited_initial_state() -> RunRecord[GE_Cfg, GE_Result]:
    rng = np.random.default_rng(83)
    excited = rng.random((2, 6000)) < np.array([0.1, 0.9])[:, None]
    signals = np.asarray(
        np.where(excited, 1 + 0.4j, -1 - 0.4j)
        + 0.2 * (rng.normal(size=excited.shape) + 1j * rng.normal(size=excited.shape)),
        dtype=np.complex128,
    )
    # Probe-off/on rows are reversed for a predominantly excited initial state.
    return RunRecord[GE_Cfg, GE_Result](
        None, GE_Result(signals[::-1].copy(), np.arange(6000), np.array([0, 1]))
    )


def test_ge_real_fit_and_post_publish_separate_named_panes_and_writebacks() -> None:
    state, tab_id, _fake, ctx = _state()
    adapter = GEAdapter()
    state.get_tab(tab_id).adapter = adapter
    state.update_tab_result(tab_id, _ge_source_with_excited_initial_state())
    bus, writeback = EventBus(), _Writeback()
    primary_service, primary_bg = _analyze_service(state, writeback, bus)
    params = GEAnalyzeParams(initial_state="excited", length_ratio=0.01)
    fit_plots = Plots(NonPresentingHost())
    primary_service.start_analyze(AnalyzePermit(tab_id), params, plots=fit_plots)
    primary_bg.on_done(primary_bg.work())

    tab = state.get_tab(tab_id)
    primary = tab.analysis.result
    assert isinstance(primary, GEAnalyzeResult)
    assert primary.initial_state == "excited"
    assert tab.analysis.plots is fit_plots
    assert tuple(fit_plots) == ("fit",)
    assert {item.target_name for item in writeback.created[-1].items} == {
        "fid",
        "ge_s",
        "g_center",
        "e_center",
    }
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
    post_service = PostAnalyzeService(
        state, post_runner, bus, post_handles, cast(Any, writeback)
    )
    post_plots = Plots(NonPresentingHost())
    post_service.start_post_analyze(tab_id, post_params, plots=post_plots)
    post_bg.on_done(post_bg.work())

    post = tab.post_analysis.result
    assert isinstance(post, GEPostAnalyzeResult)
    np.testing.assert_allclose(post.confusion, np.eye(3), atol=0.06)
    assert tab.post_analysis.plots is post_plots
    assert tuple(post_plots) == ("post",)
    assert {item.target_name for item in writeback.created[-1].items} == {
        "ge_radius",
        "confusion_matrix",
    }
    snapshot = TabService(
        state, MagicMock(), cast(Any, writeback), cfg_resources(state)
    ).get_snapshot(tab_id)
    assert snapshot.analysis is not None and snapshot.post_analysis is not None
    assert snapshot.analysis.figures is fit_plots
    assert snapshot.post_analysis.figures is post_plots
    assert snapshot.paths is not None
    assert "fit" in snapshot.paths.analysis_images
    assert "post" in snapshot.paths.post_analysis_images
    assert {item.key for item in snapshot.artifacts} >= {
        ArtifactKey(ArtifactKind.ANALYSIS, "fit"),
        ArtifactKey(ArtifactKind.POST_ANALYSIS, "post"),
    }

    new_fit = Plots(NonPresentingHost())
    primary_service.start_analyze(
        AnalyzePermit(tab_id), GEAnalyzeParams(initial_state="excited"), plots=new_fit
    )
    primary_bg.on_done(primary_bg.work())
    assert tab.post_analysis.result is None
    assert tab.post_analysis.plots is None
    old_fit, old_post = BytesIO(), BytesIO()
    fit_plots["fit"].savefig(old_fit, format="png")
    post_plots["post"].savefig(old_post, format="png")
    assert old_fit.getvalue().startswith(b"\x89PNG")
    assert old_post.getvalue().startswith(b"\x89PNG")


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
    service.start_analyze(AnalyzePermit(tab_id), "params", plots=plots)

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
    service.start_analyze(AnalyzePermit(tab_id), "params", plots=plots)
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
    load = LoadService(state, MagicMock(), provide_options=lambda _kind: [])

    with pytest.raises(LoadDataError) as exc_info:
        load.load_result(LoadPermit(tab_id), "/tmp/x")

    assert exc_info.value.category is ExpectedErrorCategory.FAILED_PRECONDITION
    assert exc_info.value.reason_code == "unsupported_load"


def test_guard_and_load_accept_enabled_concrete_adapter() -> None:
    state, tab_id, _adapter, _ctx = _state()

    class EnabledAdapter:
        capabilities = AdapterCapabilities(load_data=True)

        def load(self, _request: object) -> object:
            return object()

    state.get_tab(tab_id).adapter = EnabledAdapter()  # type: ignore[assignment]
    permit = GuardService(state).acquire_load_permit(tab_id)
    assert permit.tab_id == tab_id
