"""Capability-driven subtabs acceptance tests for Ticket 03 (A1-A5)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# Mock figure attachment to avoid host lifecycle flakiness in headless CI
import zcu_tools.gui.app.main.ui.exp_tab_widget as _exp_mod
from matplotlib.figure import Figure
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget


def _mock_attach(fig, container):
    from qtpy.QtWidgets import QWidget

    w = QWidget()
    w.figure = fig  # type: ignore[attr-defined]
    container.attach_canvas(w)
    w.draw = lambda: None  # type: ignore[attr-defined]
    return w


_exp_mod.attach_existing_figure_to_container = _mock_attach  # type: ignore[attr-defined]


@dataclass
class DummyParams:
    x: int = 1
    y: str = "test"


@dataclass
class DummyPostParams:
    p: int = 2


def _mock_ctrl():
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
    return ctrl


# Patch ExpTabWidget cfg population for test isolation
_orig_populate = _exp_mod.ExpTabWidget._populate_cfg


def _probe_populate(self, schema, ctrl):
    self._cfg_editor_id = "probe-editor"
    try:
        self.cfg_form.is_valid = lambda: True  # type: ignore[method-assign]
        self.cfg_form.first_invalid_reason = lambda: None  # type: ignore[method-assign]
    except Exception:
        pass


_exp_mod.ExpTabWidget._populate_cfg = _probe_populate  # type: ignore[attr-defined]


def _snapshot(
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
):
    return TabSnapshot(
        adapter_name="fake",
        cfg_schema=MagicMock(),
        save_paths_override=None,
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
        analyze_params=DummyParams() if has_analyze_result else None,
        post_analyze_params=DummyPostParams() if has_post_result else None,
        writeback_items=(),
        figure=Figure() if has_figure and has_analyze_result else None,
        save_paths=None,
        post_figure=Figure() if has_post_result else None,
        paths=None,
    )


def test_A1_capability_driven_composition_and_fixed_order(qapp):
    ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snap_none = _snapshot("tab-1", analysis=AnalysisMode.NONE, post=False, load=False)
    tab.attach(snap_none, MagicMock())
    visible = [
        tab._left_tabs.tabText(i)
        for i in range(tab._left_tabs.count())
        if tab._left_tabs.isTabVisible(i)
    ]
    assert visible == ["Run", "Save", "Guide"], f"got {visible}"
    snap_analysis = _snapshot("tab-1", analysis=AnalysisMode.FIT, post=False, load=True)
    tab2 = ExpTabWidget("tab-2", ctrl)
    tab2.attach(snap_analysis, MagicMock())
    visible2 = [
        tab2._left_tabs.tabText(i)
        for i in range(tab2._left_tabs.count())
        if tab2._left_tabs.isTabVisible(i)
    ]
    assert visible2 == ["Run", "Analysis", "Save", "Guide"], f"got {visible2}"
    snap_both = _snapshot("tab-1", analysis=AnalysisMode.FIT, post=True, load=True)
    tab3 = ExpTabWidget("tab-3", ctrl)
    tab3.attach(snap_both, MagicMock())
    visible3 = [
        tab3._left_tabs.tabText(i)
        for i in range(tab3._left_tabs.count())
        if tab3._left_tabs.isTabVisible(i)
    ]
    assert visible3 == ["Run", "Analysis", "Post-Analysis", "Save", "Guide"], (
        f"got {visible3}"
    )
    assert tab.load_data_btn.isHidden() is True
    assert tab2.load_data_btn.isHidden() is False
    assert tab3.load_data_btn.isHidden() is False
    tab.detach()
    tab2.detach()
    tab3.detach()


def test_A2_stable_container_and_busy_reject(qapp):
    ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snap = _snapshot("tab-1", analysis=AnalysisMode.FIT, post=True)
    tab.attach(snap, MagicMock())
    run_c = tab.get_run_container()
    ana_c = tab.get_analysis_container()
    post_c = tab.get_post_container()
    captured = tab.prepare_analysis_container()
    assert captured is ana_c
    tab._left_tabs.setCurrentWidget(tab._save_panel)
    tab._on_left_tab_changed(tab._left_tabs.currentIndex())
    snap_busy = _snapshot(
        "tab-1", analysis=AnalysisMode.FIT, post=True, is_analyzing=True
    )
    tab.update_interaction_state(snap_busy)
    assert tab.get_analysis_container() is ana_c
    assert tab.get_run_container() is run_c
    assert tab.get_post_container() is post_c
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
    try:
        state.remove_tab("tab-1")
        assert False, "should have raised"
    except RuntimeError as e:
        assert "busy" in str(e).lower()
    tab.detach()


def test_A3_primary_post_pane_reaction(qapp):
    ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snap = _snapshot(
        "tab-1", analysis=AnalysisMode.FIT, post=True, has_post_result=True
    )
    tab.attach(snap, MagicMock())
    fig_a_old = Figure()
    fig_p_old = Figure()
    tab.show_analysis_figure(fig_a_old)
    tab.show_post_analysis_figure(fig_p_old)
    assert tab.get_current_figure_for_pane("analysis") is fig_a_old
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p_old
    captured = tab.prepare_analysis_container()
    assert captured is tab.get_analysis_container()
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
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p_old2
    tab.prepare_post_container()
    assert tab.get_current_figure_for_pane("post_analysis") is None
    assert tab.get_current_figure_for_pane("analysis") is fig_a_new
    tab.detach()


def test_A4_save_image_ownership_and_placeholder(qapp):
    ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snap = _snapshot("tab-1", analysis=AnalysisMode.FIT, post=True, load=True)
    tab.attach(snap, MagicMock())

    # Run has no image path: check parent
    def contains(widget, target):
        for child in widget.findChildren(type(target)):
            if child is target:
                return True
        return False

    assert not contains(tab._run_panel, tab._image_path_edit)
    assert contains(tab._analysis_panel, tab._image_path_edit)
    assert contains(tab._post_panel, tab._post_image_path_edit)
    assert contains(tab._save_panel, tab._data_path_edit)
    tab.set_analysis_image_path("/tmp/a.png")
    tab.set_post_image_path("/tmp/p.png")
    assert tab.get_image_path() == "/tmp/a.png"
    assert tab.get_post_image_path() == "/tmp/p.png"
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


def test_A5_operation_gates_editing(qapp):
    ctrl = _mock_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snap_idle = _snapshot(
        "tab-1",
        analysis=AnalysisMode.FIT,
        post=True,
        is_analyzing=False,
        has_analyze_result=True,
        has_post_result=True,
    )
    tab.attach(snap_idle, MagicMock())
    assert tab.analyze_form.isEnabled() is True
    assert tab.post_analyze_form.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab.post_writeback_widget.isEnabled() is True
    snap_busy = _snapshot(
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


def test_controller_per_pane_routing(qapp):
    from unittest.mock import MagicMock as Mock

    from zcu_tools.gui.app.main.services.run_analyze_control import (
        RunAnalyzeControlFacet,
    )

    log = []

    class FakeHost:
        def make_live_container(self, tab_id):
            log.append("live")
            return "live"

        def make_run_container(self, tab_id):
            log.append("run")
            return "run_c"

        def make_analysis_container(self, tab_id):
            log.append("analysis")
            return "ana_c"

        def make_post_analysis_container(self, tab_id):
            log.append("post")
            return "post_c"

        def mount_interactive_analysis(self, *a, **kw):
            pass

        def unmount_interactive_analysis(self, *a, **kw):
            pass

    host = FakeHost()
    state = Mock()
    state.running_tab_id = None
    state.has_tab.return_value = True
    state.get_tab.return_value = Mock(
        adapter=Mock(capabilities=Mock(analysis=AnalysisMode.FIT))
    )
    state.exp_context = Mock(md=Mock(), ml=Mock(), predictor=None)
    guard = Mock()
    guard.acquire_run_permit.return_value = Mock(
        tab_id="tab-1", adapter=Mock(), request=Mock(), schema=Mock()
    )
    guard.acquire_analyze_permit.return_value = Mock(tab_id="tab-1")
    bus = Mock()
    tab_svc = Mock()
    load_svc = Mock()
    run_svc = Mock()
    run_svc.start_run.return_value = 1
    analyze_svc = Mock()
    analyze_svc.start_analyze.return_value = 2
    post_svc = Mock()
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
    )  # type: ignore
    facet.start_run("tab-1")
    assert log[-1] == "run"
    facet.analyze("tab-1", Mock())
    assert log[-1] == "analysis"
    facet.start_post_analyze("tab-1", Mock())
    assert log[-1] == "post"
