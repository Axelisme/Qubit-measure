"""MainWindow terminal figure projection and canvas ownership contracts."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QComboBox,
    QStackedWidget,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
)
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

from tests.gui.app.measure.artifacts._snapshots import ready_figures
from tests.gui.app.measure.ui._main_window_fakes import (
    apply_window_defaults,
    make_tab_snapshot,
    mock_ctrl,
    tab_pane_tabs,
)


def test_main_window_interaction_event_refreshes_finished_analysis_figure(
    main_window_control_factory, qapp, monkeypatch
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    writeback_item = MagicMock()
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1",
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
        writeback_items=(writeback_item,),
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        update_writeback_items = MagicMock(wraps=tab.update_writeback_items)
        monkeypatch.setattr(tab, "update_writeback_items", update_writeback_items)
        show_analysis_figures = MagicMock(wraps=tab.show_analysis_figures)
        monkeypatch.setattr(tab, "show_analysis_figures", show_analysis_figures)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(
            TabInteractionChangedPayload(
                "tab-1", TabInteractionFact.PRIMARY_ANALYZE_FAILED
            )
        )

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        update_writeback_items.assert_not_called()
        snapshot = ctrl.get_tab_snapshot.return_value
        assert snapshot.analysis is not None
        show_analysis_figures.assert_called_once_with(snapshot.analysis.figures)
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_does_not_restore_old_figure_on_analyze_start(
    main_window_control_factory,
    qapp,
    monkeypatch,
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1",
        is_analyzing=True,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        show_analysis_figures = MagicMock(wraps=tab.show_analysis_figures)
        monkeypatch.setattr(tab, "show_analysis_figures", show_analysis_figures)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(
            TabInteractionChangedPayload(
                "tab-1", TabInteractionFact.PRIMARY_ANALYZE_STARTED
            )
        )

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        show_analysis_figures.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_does_not_restore_old_figure_on_run_start(
    main_window_control_factory,
    qapp,
    monkeypatch,
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    snapshot = make_tab_snapshot(
        "tab-1",
        is_running=True,
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        figure=figure,
    )
    assert snapshot.analysis is not None
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    tab = MagicMock()
    try:
        window.add_tab_widget("tab-1", "fake")
        view = next(
            view for view in window.findChildren(ExpTabWidget) if view.tab_id == "tab-1"
        )
        monkeypatch.setattr(view.analyze_form, "sync", tab.analyze_form.sync)
        monkeypatch.setattr(view, "update_writeback_items", tab.update_writeback_items)
        monkeypatch.setattr(view, "show_analysis_figures", tab.show_analysis_figures)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(
            TabInteractionChangedPayload("tab-1", TabInteractionFact.RUN_START_REJECTED)
        )

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        tab.show_analysis_figures.assert_not_called()
        tab.analyze_form.sync.assert_called_once_with(snapshot.analysis.params)
        tab.update_writeback_items.assert_called_once_with(
            list(snapshot.analysis.writeback_items)
        )
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_interaction_event_shows_post_figure_after_primary(
    main_window_control_factory, qapp, monkeypatch
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    primary = Figure()
    post = Figure()
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1",
        is_analyzing=False,
        has_analyze_result=True,
        has_figure=True,
        has_post_analyze_result=True,
        supports_post_analysis=True,
        figure=primary,
        post_figure=post,
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        show_analysis_figures = MagicMock(wraps=tab.show_analysis_figures)
        monkeypatch.setattr(tab, "show_analysis_figures", show_analysis_figures)
        show_post_analysis_figures = MagicMock(wraps=tab.show_post_analysis_figures)
        monkeypatch.setattr(
            tab, "show_post_analysis_figures", show_post_analysis_figures
        )

        bus.emit(
            TabInteractionChangedPayload(
                "tab-1", TabInteractionFact.PRIMARY_ANALYZE_FAILED
            )
        )

        snapshot = ctrl.get_tab_snapshot.return_value
        assert snapshot.analysis is not None and snapshot.post_analysis is not None
        show_analysis_figures.assert_called_once_with(snapshot.analysis.figures)
        show_post_analysis_figures.assert_called_once_with(
            snapshot.post_analysis.figures
        )
    finally:
        window.remove_tab_widget("tab-1")


@pytest.mark.parametrize(
    "fact",
    [
        "primary_analyze_failed",
        "primary_analyze_cancelled",
        "primary_analyze_start_rejected",
        "post_analyze_failed",
        "post_analyze_start_rejected",
    ],
)
def test_analysis_terminal_restore_rebuilds_real_primary_then_post_canvas(
    main_window_control_factory, qapp, fact
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabInteractionChangedPayload,
        TabInteractionFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget, MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    primary = Figure()
    post = Figure()
    snapshot = make_tab_snapshot(
        "tab-1",
        has_analyze_result=True,
        has_post_analyze_result=True,
        has_figure=True,
        supports_post_analysis=True,
        figure=primary,
        post_figure=post,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        assert snapshot.analysis is not None and snapshot.post_analysis is not None
        tab.show_analysis_figures(snapshot.analysis.figures)
        tab.show_post_analysis_figures(snapshot.post_analysis.figures)

        # Simulate operation start clearing only its pane; here clear both figures to emulate stale state
        tab.get_analysis_container().clear_dynamic_canvases()
        tab.get_post_container().clear_dynamic_canvases()
        assert tab.get_current_figure_for_pane("analysis") is None
        assert tab.get_current_figure_for_pane("post_analysis") is None

        bus.emit(TabInteractionChangedPayload("tab-1", TabInteractionFact(fact)))

        # Coordinator failure/cancel re-renders retained figures per pane
        assert tab.get_current_figure_for_pane("analysis") is primary
        assert tab.get_current_figure_for_pane("post_analysis") is post
    finally:
        window.remove_tab_widget("tab-1")


def test_loaded_content_clears_stale_real_canvas_when_state_has_no_figure(
    main_window_control_factory, qapp
):
    from dataclasses import dataclass

    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget, MainWindow

    @dataclass
    class _Params:
        threshold: float = 0.5

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    snapshot = make_tab_snapshot(
        "tab-1",
        has_run_result=True,
        has_analyze_result=False,
        has_figure=False,
        supports_post_analysis=True,
        analyze_params=_Params(),
        figure=None,
        post_figure=None,
    )
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        tab.show_analysis_figures(ready_figures(Figure()))
        assert tab.get_current_figure_for_pane("analysis") is not None

        bus.emit(
            TabContentChangedPayload("tab-1", TabContentFact.LOADED_RESULT_COMMITTED)
        )

        assert tab.get_current_figure_for_pane("analysis") is None
        assert tab.get_current_figure_for_pane("post_analysis") is None
    finally:
        window.remove_tab_widget("tab-1")


def _emit_run_finished(bus, tab_id: str, outcome: str) -> None:
    from zcu_tools.gui.app.measure.events.run import RunFinishedPayload

    bus.emit(RunFinishedPayload(tab_id=tab_id, outcome=outcome))


def test_finished_run_preserves_selected_subtab(
    main_window_control_factory, qapp, monkeypatch
):
    """RUN_FINISHED refreshes state without choosing a result pane."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = tab_pane_tabs(tab)
        left_tabs.setCurrentIndex(2)
        assert left_tabs.currentIndex() == 2
        assert left_tabs.tabText(2) == "Data"
        focus_result_panel = MagicMock(wraps=tab.focus_result_panel)
        monkeypatch.setattr(tab, "focus_result_panel", focus_result_panel)

        _emit_run_finished(bus, "tab-1", outcome="finished")

        assert left_tabs.currentIndex() == 2
        focus_result_panel.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_stopped_run_does_not_auto_switch_to_analysis_tab(
    main_window_control_factory, qapp, monkeypatch
):
    """A stopped (cancelled) run may leave a partial result, but the user
    interrupted on purpose — must not yank them to the Analysis tab."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = tab_pane_tabs(tab)
        set_current_index = MagicMock(wraps=left_tabs.setCurrentIndex)
        monkeypatch.setattr(left_tabs, "setCurrentIndex", set_current_index)

        _emit_run_finished(bus, "tab-1", outcome="cancelled")

        set_current_index.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_non_analysis_adapter_run_preserves_selected_subtab(
    main_window_control_factory, qapp, monkeypatch
):
    """Non-analysis adapters also retain the user's selected pane after Run."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        left_tabs = tab_pane_tabs(tab)
        left_tabs.setCurrentIndex(1)
        assert left_tabs.currentIndex() == 1
        assert left_tabs.tabText(1) == "Data"
        focus_result_panel = MagicMock(wraps=tab.focus_result_panel)
        monkeypatch.setattr(tab, "focus_result_panel", focus_result_panel)

        _emit_run_finished(bus, "tab-1", outcome="finished")

        assert left_tabs.currentIndex() == 1
        focus_result_panel.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_refresh_analyze_form_skips_non_analysis_adapter_without_raising(
    main_window_control_factory, qapp, monkeypatch
):
    """A finished run on a non-analysis adapter has no analyze params; the
    content refresh must skip the analyze form rather than hit the Fast-Fail
    guard that demands initialized params (regression: it used to raise)."""
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        populate_analyze_params = MagicMock(wraps=tab.populate_analyze_params)
        monkeypatch.setattr(tab, "populate_analyze_params", populate_analyze_params)

        # Must not raise "Run result has no initialized analyze parameters".
        window.refresh_tab_analyze_form("tab-1")

        populate_analyze_params.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_show_analysis_figure_draws_canvas(
    main_window_control_factory, qapp, monkeypatch
):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
    )
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.ui.exp_tab_widget.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figures(ready_figures(Figure()))

    canvas.draw.assert_called_once_with()


def _figure_pane_stack(tab: ExpTabWidget, accessible_name: str) -> QStackedWidget:
    """Find the stack beside the selector named by accessible_name within tab.

    Assert a unique selector, a non-null parent pane, and a unique stack in that
    pane; fail if any part of this public Qt lookup is missing or ambiguous.
    """
    selectors = [
        selector
        for selector in tab.findChildren(QComboBox)
        if selector.accessibleName() == accessible_name
    ]
    assert len(selectors) == 1
    (selector,) = selectors
    pane = selector.parentWidget()
    assert pane is not None
    stacks = pane.findChildren(QStackedWidget)
    assert len(stacks) == 1
    (stack,) = stacks
    return stack


def test_show_analysis_figure_keeps_two_figures_coexisting(
    main_window_control_factory, qapp
):
    """Analysis and Post figures live in independent containers; each pane's history
    is distinct and showing one does not evict the other."""
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget(
        "tab-1",
        mock_ctrl(main_window_control_factory),
        AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True),
    )
    tab.show()
    QApplication.processEvents()

    fig_a1 = Figure()
    fig_p1 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a1))
    tab.show_post_analysis_figures(ready_figures(fig_p1))

    analysis_stack = _figure_pane_stack(tab, "Analysis figure")
    post_stack = _figure_pane_stack(tab, "Post-analysis figure")

    # Each pane has its own stack with placeholder + 1 canvas
    assert analysis_stack.count() == 2
    assert post_stack.count() == 2
    assert tab.get_current_figure_for_pane("analysis") is fig_a1
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p1

    # Re-showing analysis does not affect post
    fig_a2 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a2))
    assert tab.get_current_figure_for_pane("analysis") is fig_a2
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p1
    assert post_stack.count() == 2


# ---------------------------------------------------------------------------
