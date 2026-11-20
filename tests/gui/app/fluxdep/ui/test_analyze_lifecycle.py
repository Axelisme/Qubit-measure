"""Explicit singleton activation and Filter/Search context lifecycle."""

import copy

import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import QEvent
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.line_picker import LinePickerWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow

from tests.gui.app.fluxdep._window import (
    current_page,
    editor,
    filter_button,
    fit_panel,
    show_page,
)
from tests.gui.app.fluxdep.remote._route_harness import remote_owner


def _selectors(panel):
    """Selector views under panel once deferred Qt deletions have run."""
    QtWidgets.QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return panel.findChildren(SelectorWidget)


def apply_button(widget):
    return next(
        button
        for button in widget.findChildren(QtWidgets.QPushButton)
        if button.text() == "Apply"
    )


def test_agent_selection_updates_reuse_filter_view_and_undo(qapp, cross_controller):
    ctrl = cross_controller
    window = MainWindow(ctrl)
    window.show()
    try:
        with remote_owner(ctrl) as (route, link):
            assert window.findChild(AnalyzePanelWidget) is None
            opened = route.request(link, "selection.interactive.open")
            assert opened["ok"]
            assert current_page(window) == "fit"
            panel = fit_panel(window)
            assert panel is not None and panel.current_tab == "filter"
            selector = panel.findChild(SelectorWidget)
            assert selector is not None
            context = ctrl.interactive.current_cross_selection()
            active = ctrl.interactive.inspect_selection()
            assert context is not None and active is not None
            assert route.request(
                link,
                "selection.interactive.command",
                context_id=active.context_id,
                command="clear",
            )["ok"]
            assert ctrl.interactive.inspect_selection() == active
            assert panel.findChild(SelectorWidget) is selector
            assert not context.session.snapshot().selected.any()
            assert context.session.can_undo()
            assert route.request(
                link,
                "selection.interactive.command",
                context_id=active.context_id,
                command="undo",
            )["ok"]
            assert context.session.snapshot().selected.all()
            assert not context.session.can_undo()
            assert panel.findChild(SelectorWidget) is selector
            panel.show_tab("search")
            # Leaving Filter only retires its view; the Session stays open.
            assert _selectors(panel) == []
            assert ctrl.interactive.inspect_selection() == active
            panel.show_tab("filter")
            assert ctrl.interactive.inspect_selection() == active
            assert len(_selectors(panel)) == 1
    finally:
        ctrl.control_identity.reclaim_local()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_activate_detaches_old_view_reuses_context_and_discard_edit_resets(
    qapp, cross_controller
):
    ctrl = cross_controller
    panel = AnalyzePanelWidget(ctrl)
    try:
        panel.activate()
        # Activation shows the published masks; only Edit opens input.
        assert ctrl.interactive.current_cross_selection() is None
        assert _selectors(panel) == []
        filter_button(panel, "edit").click()
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
        filter_button(panel, "discard").click()
        assert ctrl.interactive.current_cross_selection() is None
        filter_button(panel, "edit").click()
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
        filter_button(panel, "edit").click()
        context = ctrl.interactive.current_cross_selection()
        assert context is not None
        ctrl.set_points("b", np.array([0.8]), np.array([4.8]))
        assert ctrl.interactive.current_cross_selection() is None
        show_page(window, "spectra")
        show_page(window, "fit")
        assert fit_panel(window) is panel
        # Re-entry shows the published masks; Edit captures the new cloud.
        assert ctrl.interactive.current_cross_selection() is None
        filter_button(panel, "edit").click()
        fresh = ctrl.interactive.current_cross_selection()
        assert fresh is not None and fresh is not context
        np.testing.assert_array_equal(fresh.plugin.inputs.fluxs, [0.0, 0.5, 1.0, 0.8])
    finally:
        window.close()
        qapp.processEvents()


def add_spectrum_named_selection(ctrl):
    entry = copy.deepcopy(ctrl.state.spectrums["a"])
    entry.name = "selection"
    ctrl.state.put_spectrum(entry)


def test_filter_view_attaches_global_filter_beside_spectrum_named_selection(
    qapp, cross_controller
):
    ctrl = cross_controller
    add_spectrum_named_selection(ctrl)
    draft = ctrl.interactive.begin_line_pick("selection")
    panel = AnalyzePanelWidget(ctrl)
    try:
        panel.activate()
        filter_button(panel, "edit").click()
        selector = panel.findChild(SelectorWidget)
        assert selector is not None
        filter_active = ctrl.interactive.inspect_selection()
        assert filter_active is not None and filter_active.kind == "selection"
        panel.show_tab("filter")
        assert panel.findChild(SelectorWidget) is selector
        assert ctrl.interactive.inspect_selection() == filter_active
        named = ctrl.interactive.inspect("selection")
        assert named is not None and named.context is draft
    finally:
        panel.quiesce()
        panel.detach()
        panel.deleteLater()
        qapp.processEvents()


def test_leaving_filter_keeps_global_filter_and_spectrum_named_selection(
    qapp, cross_controller
):
    ctrl = cross_controller
    add_spectrum_named_selection(ctrl)
    draft = ctrl.interactive.begin_line_pick("selection")
    panel = AnalyzePanelWidget(ctrl)
    try:
        joint = ctrl.interactive.begin_cross_selection()
        panel.show_tab("search")
        assert panel.current_tab == "search"
        filter_active = ctrl.interactive.inspect_selection()
        assert filter_active is not None and filter_active.context is joint
        named = ctrl.interactive.inspect("selection")
        assert named is not None and named.context is draft
    finally:
        panel.quiesce()
        panel.detach()
        panel.deleteLater()
        qapp.processEvents()


def test_agent_filter_reveals_beside_spectrum_named_selection_draft(
    qapp, cross_controller
):
    ctrl = cross_controller
    add_spectrum_named_selection(ctrl)
    window = MainWindow(ctrl)
    window.show()
    try:
        with remote_owner(ctrl) as (route, link):
            assert route.request(
                link, "spectrum.interactive.open", name="selection", kind="line"
            )["ok"]
            draft = ctrl.interactive.inspect("selection")
            assert draft is not None
            assert isinstance(editor(window), LinePickerWidget)
            assert route.request(link, "selection.interactive.open")["ok"]
            assert current_page(window) == "fit"
            panel = fit_panel(window)
            assert panel is not None and panel.current_tab == "filter"
            assert panel.findChild(SelectorWidget) is not None
            assert ctrl.interactive.inspect("selection") == draft
    finally:
        ctrl.control_identity.reclaim_local()
        window.close()
        window.deleteLater()
        qapp.processEvents()
