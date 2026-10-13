"""Explicit singleton activation and Filter/Search context lifecycle."""

import numpy as np
from qtpy import QtWidgets
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin

from tests.gui.app.fluxdep._window import current_page, editor, fit_panel, show_page


def apply_button(widget):
    return next(
        button
        for button in widget.findChildren(QtWidgets.QPushButton)
        if button.text() == "Apply"
    )


def test_agent_picker_reveals_same_session_while_reads_and_user_edits_do_not(
    qapp, cross_controller
):
    ctrl = cross_controller
    window = MainWindow(ctrl)
    window.show()
    try:
        show_page(window, "fit")
        panel = fit_panel(window)
        assert panel is not None
        panel.show_tab("search")

        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context = ctrl.interactive.begin_line_pick("a")
        # Agent work updates the Spectra page; the user stays on Fit.
        assert current_page(window) == "fit"
        assert panel.current_tab == "search"
        picker = editor(window)
        assert isinstance(picker, LinePickerWidget)
        active = ctrl.interactive.inspect()
        assert active is not None and active.context is context
        assert ctrl.state.spectrums["a"].points_completed

        show_page(window, "fit")
        assert fit_panel(window) is panel
        assert ctrl.interactive.inspect() == active
        context.plugin.actions.swap.execute(context.session, None)
        assert current_page(window) == "fit"
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            assert ctrl.interactive.inspect() == active
        assert current_page(window) == "fit"
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context.plugin.execute_command(context.session, "swap_lines", {})
        assert current_page(window) == "fit"
        shown = editor(window)
        assert isinstance(shown, LinePickerWidget)
        assert shown.get_result() == (
            context.session.snapshot().flux_half,
            context.session.snapshot().flux_int,
        )
        assert ctrl.interactive.inspect() == active
        assert context.session.can_undo()
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context.session.undo()
        assert editor(window) is shown
        assert not context.session.can_undo()

        ctrl.interactive.cancel()
        assert ctrl.interactive.inspect() is None
        assert editor(window) is not shown
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_agent_selection_updates_reuse_filter_view_and_undo(qapp, cross_controller):
    ctrl = cross_controller
    window = MainWindow(ctrl)
    window.show()
    try:
        assert window.findChild(AnalyzePanelWidget) is None
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context = ctrl.interactive.begin_cross_selection()
        # The agent's Filter waits on the Fit page until the user enters it.
        assert current_page(window) == "spectra"
        show_page(window, "fit")
        panel = fit_panel(window)
        assert panel is not None
        assert panel.current_tab == "filter"
        selector = panel.findChild(SelectorWidget)
        assert selector is not None
        active = ctrl.interactive.inspect()
        assert active is not None and active.kind == "selection"
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context.plugin.clear.execute(context.session, None)
        assert ctrl.interactive.inspect() == active
        assert panel.findChild(SelectorWidget) is selector
        assert not context.session.snapshot().selected.any()
        assert context.session.can_undo()
        with ctrl.bus.origin(EventOrigin(kind="agent")):
            context.session.undo()
        assert context.session.snapshot().selected.all()
        assert not context.session.can_undo()
        assert panel.findChild(SelectorWidget) is selector
        panel.show_tab("search")
        assert panel.current_tab == "search"
        assert ctrl.interactive.inspect() is None
        panel.show_tab("filter")
        fresh = ctrl.interactive.inspect()
        assert fresh is not None and fresh.context_id > active.context_id
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


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
        assert ctrl.state.selection_mask() is None
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


def test_fit_page_reentry_reactivates_singleton_after_external_change(
    qapp, cross_controller
):
    ctrl = cross_controller
    window = MainWindow(ctrl)
    try:
        show_page(window, "fit")
        panel = fit_panel(window)
        assert panel is not None
        context = ctrl.interactive.current_cross_selection()
        assert context is not None
        ctrl.set_points("b", np.array([0.8]), np.array([4.8]))
        assert ctrl.interactive.current_cross_selection() is None
        show_page(window, "spectra")
        show_page(window, "fit")
        assert fit_panel(window) is panel
        fresh = ctrl.interactive.current_cross_selection()
        assert fresh is not None and fresh is not context
        np.testing.assert_array_equal(fresh.plugin.inputs.fluxs, [0.0, 0.5, 1.0, 0.8])
    finally:
        window.close()
        qapp.processEvents()
