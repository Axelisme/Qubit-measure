"""Explicit singleton activation and Filter/Search context lifecycle."""

import numpy as np
from qtpy import QtWidgets
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow


def apply_button(widget):
    return next(
        button
        for button in widget.findChildren(QtWidgets.QPushButton)
        if button.text() == "Apply"
    )


def test_activate_detaches_old_view_reuses_context_and_filter_search_resets(
    qapp, cross_controller
):
    ctrl = cross_controller
    panel = AnalyzePanelWidget(ctrl)
    try:
        panel.activate()
        selector = panel.findChild(SelectorWidget)
        assert selector is not None
        context = ctrl.interactive.current_cross_selection()
        assert context is not None
        context.plugin.clear.execute(context.session, None)
        stale_apply = apply_button(selector)
        panel.activate()
        assert ctrl.interactive.current_cross_selection() is context
        assert selector.preview_view() is None
        stale_apply.click()
        assert ctrl.state.selection.selected is None
        tabs = panel.findChild(QtWidgets.QTabWidget)
        assert tabs is not None
        tabs.setCurrentIndex(1)
        assert ctrl.interactive.current_cross_selection() is None
        tabs.setCurrentIndex(0)
        fresh = ctrl.interactive.current_cross_selection()
        assert fresh is not None and fresh is not context
        assert fresh.session.snapshot().selected.all()
    finally:
        panel.quiesce()
        panel.detach()
        panel.deleteLater()
        qapp.processEvents()


def test_main_window_analyze_click_reactivates_singleton_after_external_change(
    qapp, cross_controller
):
    ctrl = cross_controller
    window = MainWindow(ctrl)
    try:
        analyze = next(
            button
            for button in window.findChildren(QtWidgets.QPushButton)
            if button.text().startswith("Analyze")
        )
        analyze.click()
        panel = window.findChild(AnalyzePanelWidget)
        assert panel is not None
        context = ctrl.interactive.current_cross_selection()
        assert context is not None
        ctrl.set_points("b", np.array([0.8]), np.array([4.8]))
        assert ctrl.interactive.current_cross_selection() is None
        analyze.click()
        assert window.findChild(AnalyzePanelWidget) is panel
        fresh = ctrl.interactive.current_cross_selection()
        assert fresh is not None and fresh is not context
        np.testing.assert_array_equal(fresh.plugin.inputs.fluxs, [0.0, 0.5, 1.0, 0.8])
    finally:
        window.close()
        qapp.processEvents()
