"""Fit page stages: Filter -> Search -> Show locks and the resident Filter stage."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QPushButton,
    QToolBar,
)
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.analysis.fluxdep.stroke import BrushPoint, BrushStroke
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.ui.analyze_panel import AnalyzePanelWidget
from zcu_tools.gui.app.fluxdep.ui.interactive.selector import SelectorWidget
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import EventOrigin

from tests.gui.app.fluxdep._search_result import (
    perfect_fit_result,
    synthetic_search_input,
)
from tests.gui.app.fluxdep._window import (
    click_fit_stage,
    current_page,
    filter_button,
    fit_panel,
    fit_stage_button,
    fit_stage_status,
    named,
    picker_button,
    show_page,
)


def _record_fit(ctrl: Controller, search_db: str) -> None:
    ctrl.set_fit_params(
        search_db,
        ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2)),
        TransitionDict({"transitions": [(0, 1)]}),
        None,
        None,
    )
    bounds = ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2))
    fluxs, freqs = ctrl.selected_pointcloud()
    ctrl.record_search_result(
        perfect_fit_result(fluxs, freqs, bounds),
        synthetic_search_input(ctrl.state, fluxs, freqs, bounds),
    )


def _statuses(panel: AnalyzePanelWidget) -> tuple[str, str, str]:
    return (
        fit_stage_status(panel, "filter"),
        fit_stage_status(panel, "search"),
        fit_stage_status(panel, "show"),
    )


def _selectors(panel: AnalyzePanelWidget) -> list[SelectorWidget]:
    """Selector views under panel once deferred Qt deletions have run."""
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return panel.findChildren(SelectorWidget)


def _scatter_offsets(figure: Figure) -> list[list[list[float]]]:
    return [
        np.asarray(artist.get_offsets()).tolist()
        for axes in figure.axes
        for artist in axes.collections
        if isinstance(artist, PathCollection)
    ]


def _published_view(panel: AnalyzePanelWidget) -> FigureCanvasQTAgg:
    return named(panel, FigureCanvasQTAgg, "fluxdep.filter.published")


def _shows_published_masks(panel: AnalyzePanelWidget, ctrl: Controller) -> bool:
    """Whether the Filter stage shows what Controller.render_selection draws."""
    view = _published_view(panel)
    expected = Figure()
    ctrl.render_selection(expected)
    return view.isVisibleTo(panel) and _scatter_offsets(
        view.figure
    ) == _scatter_offsets(expected)


@pytest.fixture
def panel(qapp, cross_controller):
    widget = AnalyzePanelWidget(cross_controller)
    widget.show()
    widget.show_tab("filter")
    yield widget, cross_controller
    widget.quiesce()
    widget.detach()
    widget.dispose()
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def window(qapp, cross_controller):
    view = MainWindow(cross_controller)
    view.show()
    yield view, cross_controller
    cross_controller.control_identity.reclaim_local()
    view.close()
    view.deleteLater()
    qapp.processEvents()


def test_search_locks_until_every_sampled_spectrum_has_a_published_mask(
    panel, search_database_hdf5
):
    widget, ctrl = panel
    assert _statuses(widget) == ("open", "locked", "locked")
    assert not fit_stage_button(widget, "search").isEnabled()
    ctrl.set_selection(np.array([True, False, True, True]))
    assert _statuses(widget) == ("done", "open", "locked")
    click_fit_stage(widget, "search")
    assert widget.current_tab == "search"
    assert fit_stage_button(widget, "search").isChecked()
    _record_fit(ctrl, search_database_hdf5)
    assert _statuses(widget) == ("done", "done", "open")
    click_fit_stage(widget, "show")
    assert widget.current_tab == "show"
    # New points for b drop b's mask: Filter is no longer complete, so Search
    # locks and Show locks with it, though the old result is still recorded.
    ctrl.set_points("b", np.array([0.8]), np.array([4.8]))
    assert ctrl.state.fit.has_result
    assert _statuses(widget) == ("open", "locked", "locked")
    assert not fit_stage_button(widget, "search").isEnabled()
    assert not fit_stage_button(widget, "show").isEnabled()
    # Show stays shown: Display is off, the local view toolbar stays on.
    assert widget.current_tab == "show"
    assert _show_inputs_enabled(widget) == (False, False)
    assert named(widget, QToolBar, "fluxdep.show.toolbar").isEnabled()


def test_live_filter_draft_locks_search_and_show_until_discard_or_finish(
    panel, search_database_hdf5
):
    widget, ctrl = panel
    ctrl.set_selection(np.array([True, True, True, True]))
    _record_fit(ctrl, search_database_hdf5)
    assert _statuses(widget) == ("done", "done", "open")
    filter_button(widget, "edit").click()
    assert _statuses(widget) == ("draft", "locked", "locked")
    filter_button(widget, "discard").click()
    assert ctrl.interactive.current_cross_selection() is None
    assert _statuses(widget) == ("done", "done", "open")
    filter_button(widget, "edit").click()
    assert _statuses(widget) == ("draft", "locked", "locked")
    (selector,) = _selectors(widget)
    picker_button(selector, "finish").click()
    assert ctrl.interactive.current_cross_selection() is None
    assert _statuses(widget) == ("done", "done", "open")


def test_an_agent_filter_draft_locks_the_stages_too(panel):
    widget, ctrl = panel
    ctrl.set_selection(np.array([True, True, True, True]))
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.begin_cross_selection()
    assert _statuses(widget) == ("draft", "locked", "locked")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.cancel_selection()
    assert _statuses(widget) == ("done", "open", "locked")


def test_filter_shows_published_masks_to_holder_and_viewer_without_a_session(
    window,
):
    view, ctrl = window
    ctrl.set_selection(np.array([True, False, True, False]))
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None and panel.current_tab == "filter"
    assert ctrl.interactive.current_cross_selection() is None
    assert _selectors(panel) == []
    assert _shows_published_masks(panel, ctrl)
    assert filter_button(panel, "edit").isEnabled()
    assert not filter_button(panel, "discard").isEnabled()

    ctrl.control_identity.connect("a", "flux-agent", "owner")
    show_page(view, "spectra")
    show_page(view, "fit")
    assert panel.current_tab == "filter"
    assert _shows_published_masks(panel, ctrl)
    assert not filter_button(panel, "edit").isEnabled()
    assert not filter_button(panel, "discard").isEnabled()
    filter_button(panel, "edit").click()
    assert ctrl.interactive.current_cross_selection() is None


def test_published_view_follows_a_new_publication(panel):
    widget, ctrl = panel
    assert _shows_published_masks(widget, ctrl)
    ctrl.set_selection(np.array([False, True, True, False]))
    assert _shows_published_masks(widget, ctrl)


def test_holder_edit_opens_a_session_and_leaving_filter_keeps_its_draft(window):
    view, ctrl = window
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    filter_button(panel, "edit").click()
    context = ctrl.interactive.current_cross_selection()
    assert context is not None
    (selector,) = _selectors(panel)
    assert not _published_view(panel).isVisibleTo(panel)
    assert not filter_button(panel, "edit").isEnabled()
    assert filter_button(panel, "discard").isEnabled()
    context.plugin.stroke.execute(
        context.session, BrushStroke((BrushPoint(0.0, 4.0),), 0.0, "erase")
    )
    draft = context.session.snapshot().selected.tolist()

    # A programmatic switch (as an agent's search does) leaves Filter.
    panel.show_tab("search")
    assert _selectors(panel) == []
    assert ctrl.interactive.current_cross_selection() is context
    panel.show_tab("filter")
    (selector,) = _selectors(panel)
    assert selector.get_result().selected.tolist() == [False, True, True, True]
    assert ctrl.interactive.current_cross_selection() is context
    assert context.session.snapshot().selected.tolist() == draft
    assert context.session.can_undo()

    # Leaving the Fit page keeps the same draft as well.
    show_page(view, "spectra")
    assert _selectors(panel) == []
    show_page(view, "fit")
    assert panel.current_tab == "filter"
    assert len(_selectors(panel)) == 1
    assert ctrl.interactive.current_cross_selection() is context
    assert context.session.snapshot().selected.tolist() == draft


def test_edit_starts_from_the_published_mask_and_discard_keeps_it(panel):
    widget, ctrl = panel
    ctrl.set_selection(np.array([True, False, True, False]), 0.025)
    filter_button(widget, "edit").click()
    context = ctrl.interactive.current_cross_selection()
    assert context is not None
    assert context.session.snapshot().selected.tolist() == [True, False, True, False]
    assert context.session.snapshot().min_distance == 0.025
    context.plugin.clear.execute(context.session, None)
    filter_button(widget, "discard").click()
    assert ctrl.interactive.current_cross_selection() is None
    assert ctrl.state.selection_mask().tolist() == [True, False, True, False]
    assert _selectors(widget) == []
    assert _shows_published_masks(widget, ctrl)


def test_finish_publishes_the_draft_and_returns_to_the_published_view(panel):
    widget, ctrl = panel
    filter_button(widget, "edit").click()
    context = ctrl.interactive.current_cross_selection()
    assert context is not None
    context.plugin.stroke.execute(
        context.session, BrushStroke((BrushPoint(1.0, 5.0),), 0.0, "erase")
    )
    (selector,) = _selectors(widget)
    picker_button(selector, "finish").click()
    assert ctrl.interactive.current_cross_selection() is None
    assert ctrl.state.selection_mask().tolist() == [True, True, False, True]
    assert _selectors(widget) == []
    assert _shows_published_masks(widget, ctrl)


def test_non_holder_views_the_agent_filter_but_cannot_edit_or_discard(window):
    view, ctrl = window
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        context = ctrl.interactive.begin_cross_selection()
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None and panel.current_tab == "filter"
    (selector,) = _selectors(panel)
    assert not picker_button(selector, "finish").isEnabled()
    assert not filter_button(panel, "edit").isEnabled()
    discard = filter_button(panel, "discard")
    assert not discard.isEnabled()
    discard.click()
    assert ctrl.interactive.current_cross_selection() is context
    show_page(view, "spectra")
    assert _selectors(panel) == []
    assert ctrl.interactive.current_cross_selection() is context


def test_agent_filter_on_the_shown_fit_page_switches_only_its_stage(window):
    view, ctrl = window
    ctrl.set_selection(np.array([True, True, True, True]))
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    click_fit_stage(panel, "search")
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        context = ctrl.interactive.begin_cross_selection()
    assert current_page(view) == "fit"
    assert panel.current_tab == "filter"
    (selector,) = _selectors(panel)
    assert ctrl.interactive.current_cross_selection() is context
    assert selector.get_result().selected.all()


def test_edit_without_points_stays_disabled(qapp):
    from zcu_tools.gui.app.fluxdep.state import FluxDepState

    ctrl = Controller(FluxDepState())
    widget = AnalyzePanelWidget(ctrl)
    try:
        widget.show_tab("filter")
        edit = filter_button(widget, "edit")
        assert not edit.isEnabled()
        edit.click()
        assert ctrl.interactive.current_cross_selection() is None
        assert _statuses(widget) == ("open", "locked", "locked")
    finally:
        widget.quiesce()
        widget.dispose()
        widget.deleteLater()
        ctrl.interactive.dispose()


def _search_inputs_enabled(panel: AnalyzePanelWidget) -> tuple[bool, bool]:
    """Whether the Search database field and bounds preset take input."""
    return (
        named(panel, QLineEdit, "fluxdep.fit.search_db").isEnabled(),
        named(panel, QComboBox, "fluxdep.fit.bounds_preset").isEnabled(),
    )


def _show_inputs_enabled(panel: AnalyzePanelWidget) -> tuple[bool, bool]:
    """Whether Show's Auto limits and Apply take input."""
    return (
        named(panel, QPushButton, "fluxdep.show.auto_limits").isEnabled(),
        named(panel, QPushButton, "fluxdep.show.apply").isEnabled(),
    )


def test_a_locked_search_is_shown_disabled_until_it_unlocks(panel):
    widget, ctrl = panel
    widget.show_tab("search")
    assert widget.current_tab == "search"
    assert fit_stage_status(widget, "search") == "locked"
    assert _search_inputs_enabled(widget) == (False, False)
    ctrl.set_selection(np.array([True, True, True, True]))
    assert _search_inputs_enabled(widget) == (True, True)


def test_search_stays_shown_but_disabled_when_it_locks_later(panel):
    widget, ctrl = panel
    ctrl.set_selection(np.array([True, True, True, True]))
    click_fit_stage(widget, "search")
    assert _search_inputs_enabled(widget) == (True, True)
    # An agent's Filter draft locks Search; the user keeps seeing Search.
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.begin_cross_selection()
    assert widget.current_tab == "search"
    assert _search_inputs_enabled(widget) == (False, False)
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.cancel_selection()
    assert _search_inputs_enabled(widget) == (True, True)
    # New points for b drop its mask: Search locks again, still shown.
    ctrl.set_points("b", np.array([0.8]), np.array([4.8]))
    assert widget.current_tab == "search"
    assert _search_inputs_enabled(widget) == (False, False)


def test_show_stays_shown_with_display_disabled_while_locked(
    panel, search_database_hdf5
):
    widget, ctrl = panel
    widget.show_tab("show")
    assert fit_stage_status(widget, "show") == "locked"
    assert _show_inputs_enabled(widget) == (False, False)
    ctrl.set_selection(np.array([True, True, True, True]))
    _record_fit(ctrl, search_database_hdf5)
    assert _show_inputs_enabled(widget) == (True, True)
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.begin_cross_selection()
    assert widget.current_tab == "show"
    assert _show_inputs_enabled(widget) == (False, False)
    # The toolbar only moves this viewer's view of the drawing.
    assert named(widget, QToolBar, "fluxdep.show.toolbar").isEnabled()
    with ctrl.bus.origin(EventOrigin(kind="agent")):
        ctrl.interactive.cancel_selection()
    assert _show_inputs_enabled(widget) == (True, True)


def test_an_unlocked_stage_still_follows_control_authority(
    window, search_database_hdf5
):
    view, ctrl = window
    ctrl.set_selection(np.array([True, True, True, True]))
    _record_fit(ctrl, search_database_hdf5)
    show_page(view, "fit")
    panel = fit_panel(view)
    assert panel is not None
    click_fit_stage(panel, "search")
    ctrl.control_identity.connect("a", "flux-agent", "owner")
    assert fit_stage_status(panel, "search") == "done"
    assert _search_inputs_enabled(panel) == (False, False)
    click_fit_stage(panel, "show")
    assert _show_inputs_enabled(panel) == (False, False)
    ctrl.control_identity.reclaim_local()
    assert _show_inputs_enabled(panel) == (True, True)
    click_fit_stage(panel, "search")
    assert _search_inputs_enabled(panel) == (True, True)
