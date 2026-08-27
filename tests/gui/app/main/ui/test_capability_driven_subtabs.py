"""Capability-driven subtabs tests for Ticket 03."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState


@dataclass
class DummyParams:
    x: int = 1
    y: str = "test"


@dataclass
class DummyPostParams:
    p: int = 2


def make_ctrl():
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.update_tab_data_path = MagicMock()
    ctrl.update_tab_analysis_image_path = MagicMock()
    ctrl.update_tab_post_analysis_image_path = MagicMock()
    ctrl.update_tab_save_paths = MagicMock()
    ctrl.open_seeded_cfg_editor.return_value = ("editor-id", ())
    ctrl.get_cfg_editor_draft.return_value = MagicMock()
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    return ctrl


def make_snapshot(
    tab_id: str,
    *,
    analysis=AnalysisMode.FIT,
    post=False,
    load=False,
    is_analyzing=False,
    has_run_result=True,
    has_analyze_result=True,
    has_post_result=False,
    has_figure=True,
    analysis_figure=None,
    post_analysis_figure=None,
    analysis_writeback_items=(),
    post_writeback_items=(),
):
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )

    # Allow explicit pane figure override or create based on flags
    if analysis_figure is None and has_figure and has_analyze_result:
        figure_obj = Figure()
    elif analysis_figure is not None:
        figure_obj = analysis_figure
    else:
        figure_obj = analysis_figure  # None or explicit
    # Post-analysis figure
    if post_analysis_figure is None and has_post_result:
        post_figure_obj = Figure()
    elif post_analysis_figure is not None:
        post_figure_obj = post_analysis_figure
    else:
        post_figure_obj = post_analysis_figure

    # Determine actual figure for has_figure flag: keep consistency
    # Paths
    data_path_snap = PathResourceSnapshot(
        override=None, path="/tmp/data.hdf5" if has_run_result else None
    )
    analysis_image_snap = PathResourceSnapshot(
        override=None,
        path="/tmp/image.png" if has_figure and has_analyze_result else None,
    )
    post_image_snap = PathResourceSnapshot(
        override=None, path="/tmp/post.png" if has_post_result else None
    )
    run_snap = RunPaneSnapshot(
        result=object() if has_run_result else None, source_path=None
    )
    analysis_snap = AnalysisPaneSnapshot(
        params=DummyParams() if has_analyze_result else None,
        result=object() if has_analyze_result else None,
        figure=figure_obj,
        writeback_items=tuple(analysis_writeback_items),
        image_path=analysis_image_snap,
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=DummyPostParams() if has_post_result else None,
        result=object() if has_post_result else None,
        figure=post_figure_obj,
        writeback_items=tuple(post_writeback_items),
        image_path=post_image_snap,
    )
    save_snap = SavePaneSnapshot(data_path=data_path_snap)
    paths_snap = TabPathsSnapshot(
        data=data_path_snap,
        analysis_image=analysis_image_snap,
        post_analysis_image=post_image_snap,
    )
    return TabSnapshot(
        adapter_name="fake",
        cfg_schema=MagicMock(),
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=is_analyzing,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=has_run_result,
            has_analyze_result=has_analyze_result,
            has_figure=has_figure,
            has_post_analyze_result=has_post_result,
        ),
        capabilities=AdapterCapabilities(
            analysis=analysis, post_analysis=post, load_data=load
        ),  # type: ignore[call-arg]
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=save_snap,
        paths=paths_snap,
    )


@pytest.fixture
def exp_tab_widget(qapp, monkeypatch):
    """Provide ExpTabWidget with cfg population stubbed for isolation."""
    import zcu_tools.gui.app.main.ui.exp_tab_widget as mod

    orig = mod.ExpTabWidget._populate_cfg

    def stub(self, schema, ctrl):
        self._cfg_editor_id = "probe-editor"
        self.cfg_form.is_valid = lambda: True  # type: ignore[method-assign]
        self.cfg_form.first_invalid_reason = lambda: None  # type: ignore[method-assign]

    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", stub)
    # Mock figure attachment to avoid host lifecycle
    orig_attach = mod.attach_existing_figure_to_container

    def mock_attach(fig, container):
        from qtpy.QtWidgets import QWidget

        w = QWidget()
        w.figure = fig  # type: ignore[attr-defined]
        container.attach_canvas(w)
        w.draw = lambda: None  # type: ignore[attr-defined]
        return w

    monkeypatch.setattr(mod, "attach_existing_figure_to_container", mock_attach)
    yield mod.ExpTabWidget
    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", orig)
    monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)


def test_visible_subtabs_follow_capabilities_in_fixed_order(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    snap_none = make_snapshot(
        "tab-1", analysis=AnalysisMode.NONE, post=False, load=False
    )
    assert snap_none.capabilities is not None
    tab_none = exp_tab_widget("tab-1", ctrl, snap_none.capabilities)
    tab_none.attach(snap_none, MagicMock())
    # Fixed order Run | Save | Guide — Analysis/Post not constructed
    visible_none = [
        tab_none._left_tabs.tabText(i) for i in range(tab_none._left_tabs.count())
    ]
    assert visible_none == ["Run", "Save", "Guide"]
    # Prove absent Analysis/Post pages and controls/containers were never constructed, not only hidden
    assert not hasattr(tab_none, "analyze_form")
    assert not hasattr(tab_none, "writeback_widget")
    assert not hasattr(tab_none, "_image_path_edit")
    assert not hasattr(tab_none, "_analysis_panel")
    assert not hasattr(tab_none, "_analysis_container")
    assert not hasattr(tab_none, "post_analyze_form")
    assert not hasattr(tab_none, "post_writeback_widget")
    assert not hasattr(tab_none, "_post_image_path_edit")
    assert not hasattr(tab_none, "_post_panel")
    assert not hasattr(tab_none, "_post_container")
    with pytest.raises(RuntimeError, match="does not support analysis"):
        tab_none.get_analysis_container()
    with pytest.raises(RuntimeError, match="does not support post-analysis"):
        tab_none.get_post_container()
    with pytest.raises(RuntimeError, match="does not support analysis"):
        tab_none.get_image_path()
    # Mismatch must be rejected
    bad_snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    with pytest.raises(RuntimeError, match="capability mismatch"):
        tab_none.attach(bad_snap, MagicMock())

    snap_analysis = make_snapshot(
        "tab-2", analysis=AnalysisMode.FIT, post=False, load=True
    )
    assert snap_analysis.capabilities is not None
    tab_analysis = exp_tab_widget("tab-2", ctrl, snap_analysis.capabilities)
    tab_analysis.attach(snap_analysis, MagicMock())
    visible_analysis = [
        tab_analysis._left_tabs.tabText(i)
        for i in range(tab_analysis._left_tabs.count())
    ]
    assert visible_analysis == ["Run", "Analysis", "Save", "Guide"]
    assert hasattr(tab_analysis, "analyze_form")
    assert hasattr(tab_analysis, "_image_path_edit")
    assert not hasattr(tab_analysis, "post_analyze_form")
    with pytest.raises(RuntimeError, match="does not support post-analysis"):
        tab_analysis.get_post_container()

    snap_both = make_snapshot("tab-3", analysis=AnalysisMode.FIT, post=True, load=True)
    assert snap_both.capabilities is not None
    tab_both = exp_tab_widget("tab-3", ctrl, snap_both.capabilities)
    tab_both.attach(snap_both, MagicMock())
    visible_both = [
        tab_both._left_tabs.tabText(i) for i in range(tab_both._left_tabs.count())
    ]
    assert visible_both == ["Run", "Analysis", "Post-Analysis", "Save", "Guide"]
    assert hasattr(tab_both, "analyze_form")
    assert hasattr(tab_both, "post_analyze_form")

    assert tab_none.load_data_btn is None
    assert tab_analysis.load_data_btn is not None
    assert tab_analysis.load_data_btn.isHidden() is False

    tab_none.detach()
    tab_analysis.detach()
    tab_both.detach()


def test_main_window_refresh_projects_post_writeback_into_post_widget(
    qapp, exp_tab_widget
):
    from zcu_tools.gui.app.main.adapter import MetaDictWriteback
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = make_ctrl()
    empty = make_snapshot(
        "tab-1", analysis=AnalysisMode.FIT, post=True, has_post_result=True
    )
    assert empty.capabilities is not None
    tab = exp_tab_widget("tab-1", ctrl, empty.capabilities)
    tab.attach(empty, MagicMock())
    assert tab.post_writeback_widget._items == []

    item = MetaDictWriteback(target_name="radius", description="post", proposed_value=1)
    item.session_id = "md-1"
    item.selected = True
    committed = make_snapshot(
        "tab-1",
        analysis=AnalysisMode.FIT,
        post=True,
        has_post_result=True,
        post_writeback_items=(item,),
    )
    host = MagicMock()
    host._tab_widgets = {"tab-1": tab}
    host._ctrl = ctrl

    MainWindow.refresh_tab_writeback(host, "tab-1", committed)

    assert tab.post_writeback_widget._items == [item]
    tab.detach()


def test_figure_containers_remain_stable_across_tab_switch_and_busy(
    qapp, exp_tab_widget
):
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=True)
    assert snap.capabilities is not None
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    run_c = tab.get_run_container()
    ana_c = tab.get_analysis_container()
    post_c = tab.get_post_container()
    captured = tab.prepare_analysis_container()
    assert captured is ana_c
    tab._left_tabs.setCurrentWidget(tab._save_panel)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    snap_busy = make_snapshot(
        "tab-1", analysis=AnalysisMode.FIT, post=True, is_analyzing=True
    )
    tab.update_interaction_state(snap_busy)
    assert tab.get_analysis_container() is ana_c
    assert tab.get_run_container() is run_c
    assert tab.get_post_container() is post_c
    # Busy tab cannot be closed
    from zcu_tools.gui.app.main.state import Session, State
    from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    exp_ctx = ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        database_path="/db",
        result_dir="/r",
        active_label="a",
    )
    state = State(exp_ctx)
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True
    )  # type: ignore[call-arg]
    state.add_tab(
        "tab-1",
        Session(
            adapter_name="fake",
            adapter=adapter,
            cfg_schema=CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue()),
        ),
    )
    state.set_tab_analyzing("tab-1", True)
    with pytest.raises(RuntimeError, match="busy"):
        state.remove_tab("tab-1")
    tab.detach()


def test_primary_analysis_lifecycle_clears_only_its_pane_and_restores_on_failure(
    qapp, exp_tab_widget
):
    ctrl = make_ctrl()
    snap = make_snapshot(
        "tab-1", analysis=AnalysisMode.FIT, post=True, has_post_result=True
    )
    assert snap.capabilities is not None
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    fig_a_old = Figure()
    fig_p_old = Figure()
    tab.show_analysis_figure(fig_a_old)
    tab.show_post_analysis_figure(fig_p_old)
    assert tab.get_current_figure_for_pane("analysis") is fig_a_old
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p_old
    tab.prepare_analysis_container()
    assert tab.get_current_figure_for_pane("analysis") is None
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p_old
    tab.show_analysis_figure(fig_a_old)
    assert tab.get_current_figure_for_pane("analysis") is fig_a_old
    fig_a_new = Figure()
    tab._post_container.clear_dynamic_canvases()
    tab.show_analysis_figure(fig_a_new)
    assert tab.get_current_figure_for_pane("analysis") is fig_a_new
    assert tab.get_current_figure_for_pane("post_analysis") is None
    fig_p_old2 = Figure()
    tab.show_post_analysis_figure(fig_p_old2)
    tab.prepare_post_container()
    assert tab.get_current_figure_for_pane("post_analysis") is None
    assert tab.get_current_figure_for_pane("analysis") is fig_a_new
    tab.detach()


@pytest.mark.parametrize(
    "fact_name",
    [
        "PRIMARY_ANALYZE_FAILED",
        "PRIMARY_ANALYZE_CANCELLED",
        "PRIMARY_ANALYZE_START_REJECTED",
        "POST_ANALYZE_FAILED",
        "POST_ANALYZE_START_REJECTED",
    ],
)
def test_analysis_terminal_restores_retained_figures_via_coordinator(
    qapp, exp_tab_widget, monkeypatch, fact_name
):
    """Terminal reactions restore retained primary then post figures."""
    from zcu_tools.gui.app.main.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.main.ui.main_window import MainWindow
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    ctrl = make_ctrl()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    # Initial snapshot with retained figures (State still holds them)
    fig_a_retained = Figure()
    fig_p_retained = Figure()
    ctrl.get_tab_snapshot.return_value = make_snapshot(
        "tab-1",
        analysis=AnalysisMode.FIT,
        post=True,
        has_post_result=True,
        analysis_figure=fig_a_retained,
        post_analysis_figure=fig_p_retained,
    )
    window = MainWindow(ctrl)
    # Tab must be constructed with the same capabilities as the snapshot the window will fetch
    # The mock's get_tab_snapshot returns a FIT+post snapshot; reuse those caps
    _tmp_snap = make_snapshot(
        "tab-1",
        analysis=AnalysisMode.FIT,
        post=True,
        has_post_result=True,
        analysis_figure=Figure(),
        post_analysis_figure=Figure(),
    )
    tab = exp_tab_widget("tab-1", ctrl, _tmp_snap.capabilities)
    tab.attach(ctrl.get_tab_snapshot.return_value, MagicMock())
    window._tab_widgets["tab-1"] = tab
    # Seed figures
    fig_a = Figure()
    fig_p = Figure()
    tab.show_analysis_figure(fig_a)
    tab.show_post_analysis_figure(fig_p)
    # Clear post to simulate start, then emit failure fact which should restore both from State
    tab.prepare_post_container()
    assert tab.get_current_figure_for_pane("post_analysis") is None
    # Snapshot on failure returns retained figures (same objects as above)
    # Emitting the domain fact exercises the subscribed coordinator reaction.
    bus.emit(
        TabInteractionChangedPayload("tab-1", getattr(TabInteractionFact, fact_name))
    )
    assert tab.get_current_figure_for_pane("analysis") is fig_a_retained
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p_retained
    tab.detach()


def test_save_and_image_ownership_and_placeholder_routing(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=True, load=True)
    assert snap.capabilities is not None
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())

    # Run pane does not contain image edits
    def contains(widget, target):
        return any(child is target for child in widget.findChildren(type(target)))

    assert not contains(tab._run_panel, tab._image_path_edit)
    assert contains(tab._analysis_panel, tab._image_path_edit)
    assert contains(tab._post_panel, tab._post_image_path_edit)
    assert contains(tab._save_panel, tab._data_path_edit)
    tab.set_data_path("/tmp/data.hdf5")
    tab.set_analysis_image_path("/tmp/a.png")
    tab.set_post_image_path("/tmp/p.png")
    assert tab.get_data_path() == "/tmp/data.hdf5"
    assert tab.get_image_path() == "/tmp/a.png"
    assert tab.get_post_image_path() == "/tmp/p.png"

    tab.set_data_path("")
    tab.set_analysis_image_path("")
    tab.set_post_image_path("")
    assert tab.get_data_path() == ""
    assert tab.get_image_path() == ""
    assert tab.get_post_image_path() == ""
    tab._left_tabs.setCurrentWidget(tab._save_panel)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._right_placeholder
    tab._left_tabs.setCurrentWidget(tab._guide_panel)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._right_placeholder
    fig_a = Figure()
    fig_p = Figure()
    tab.show_analysis_figure(fig_a)
    tab.show_post_analysis_figure(fig_p)
    tab._left_tabs.setCurrentWidget(tab._analysis_panel)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._analysis_stack
    tab._left_tabs.setCurrentWidget(tab._post_panel)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    assert tab._right_stack.currentWidget() is tab._post_stack
    assert tab.get_current_figure_for_pane("analysis") is fig_a
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p
    tab.detach()


def test_operation_gates_editing_for_affected_pane(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    snap_idle = make_snapshot(
        "tab-1",
        analysis=AnalysisMode.FIT,
        post=True,
        is_analyzing=False,
        has_analyze_result=True,
        has_post_result=True,
    )
    assert snap_idle.capabilities is not None
    tab = exp_tab_widget("tab-1", ctrl, snap_idle.capabilities)
    tab.attach(snap_idle, MagicMock())
    assert tab.analyze_form.isEnabled() is True
    assert tab.post_analyze_form.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab.post_writeback_widget.isEnabled() is True
    snap_busy = make_snapshot(
        "tab-1",
        analysis=AnalysisMode.FIT,
        post=True,
        is_analyzing=True,
        has_analyze_result=True,
        has_post_result=True,
    )
    tab.update_interaction_state(snap_busy)
    assert tab.analyze_form.isEnabled() is False
    assert tab.writeback_widget.isEnabled() is False
    assert tab.reset_btn.isEnabled() is False
    tab.update_interaction_state(snap_idle)
    assert tab.analyze_form.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    tab.detach()


def test_post_writeback_operates_on_its_own_draft(qapp, exp_tab_widget):
    ctrl = make_ctrl()
    # Setup controller mock to track pane-qualified calls
    ctrl.set_writeback_item_for_pane = MagicMock(return_value={"valid": True})
    ctrl.get_writeback_item_draft_for_pane = MagicMock(return_value=MagicMock())
    snap = make_snapshot(
        "tab-1", analysis=AnalysisMode.FIT, post=True, has_post_result=True
    )
    assert snap.capabilities is not None
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    # Populate post writeback with a dummy item
    from zcu_tools.gui.app.main.adapter import MetaDictWriteback

    item = MetaDictWriteback(target_name="x", description="d", proposed_value=1)
    item.session_id = "md-1"
    item.selected = True
    tab.post_writeback_widget.populate([item])
    # Toggle should route to post_analysis pane, not analysis
    # Find checkbox and toggle
    for child in tab.post_writeback_widget.findChildren(MagicMock):
        pass
    # Directly call handler via widget
    tab.post_writeback_widget._on_check_toggled(item)
    ctrl.set_writeback_item_for_pane.assert_called_with(
        "tab-1", "post_analysis", "md-1", selected=item.selected
    )
    # Edit should also route pane-qualified
    ctrl.set_writeback_item_for_pane.reset_mock()
    # Simulate edit via widget's edit path not needed; verify that primary widget still uses analysis pane
    primary_item = MetaDictWriteback(
        target_name="y", description="d2", proposed_value=2
    )
    primary_item.session_id = "md-1"
    primary_item.selected = True
    tab.writeback_widget.populate([primary_item])
    tab.writeback_widget._on_check_toggled(primary_item)
    ctrl.set_writeback_item_for_pane.assert_called_with(
        "tab-1", "analysis", "md-1", selected=primary_item.selected
    )
    tab.detach()


def test_render_host_routes_to_correct_pane_container(qapp):
    from zcu_tools.gui.app.main.services.run_analyze_control import (
        RunAnalyzeControlFacet,
    )

    log: list[str] = []

    class FakeHost:
        def make_run_container(self, tab_id: str) -> Any:
            log.append("run")
            return "run_c"

        def make_analysis_container(self, tab_id: str) -> Any:
            log.append("analysis")
            return "ana_c"

        def make_post_analysis_container(self, tab_id: str) -> Any:
            log.append("post")
            return "post_c"

        def mount_interactive_analysis(self, *a, **kw):
            pass

        def unmount_interactive_analysis(self, *a, **kw):
            pass

    host = FakeHost()
    state = MagicMock()
    state.running_tab_id = None
    state.has_tab.return_value = True
    state.get_tab.return_value = MagicMock(
        adapter=MagicMock(capabilities=MagicMock(analysis=AnalysisMode.FIT))
    )
    state.exp_context = MagicMock(md=MagicMock(), ml=MagicMock(), predictor=None)
    guard = MagicMock()
    guard.acquire_run_permit.return_value = MagicMock(
        tab_id="tab-1", adapter=MagicMock(), request=MagicMock(), schema=MagicMock()
    )
    guard.acquire_analyze_permit.return_value = MagicMock(tab_id="tab-1")
    bus = MagicMock()
    tab_svc = MagicMock()
    load_svc = MagicMock()
    run_svc = MagicMock()
    run_svc.start_run.return_value = 1
    analyze_svc = MagicMock()
    analyze_svc.start_analyze.return_value = 2
    post_svc = MagicMock()
    post_svc.start_post_analyze.return_value = 3
    facet = RunAnalyzeControlFacet(
        state=state,
        bus=bus,
        guard=guard,
        tab=tab_svc,
        load=load_svc,
        run=run_svc,
        analyze=analyze_svc,
        post_analyze=post_svc,
        render_host=lambda: host,
    )  # type: ignore[arg-type]
    facet.start_run("tab-1")
    assert log[-1] == "run"
    facet.analyze("tab-1", MagicMock())
    assert log[-1] == "analysis"
    facet.start_post_analyze("tab-1", MagicMock())
    assert log[-1] == "post"
