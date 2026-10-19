"""ExpTabWidget Data gallery routing, Save dispatch, lifecycle, and geometry contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QCoreApplication, QEvent, QRect
from qtpy.QtWidgets import QApplication, QPushButton, QScrollArea, QTabWidget
from zcu_tools.experiment.artifacts import ArtifactRef
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.control_identity import ControlIdentity

from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure.artifacts._snapshots import ready_figures, with_artifacts
from tests.gui.app.measure.ui._figure_presentation_support import (
    gallery,
    right_page,
    right_stack,
)
from tests.gui.app.measure.ui._preview_gallery_support import (
    FAKE_PNG_BYTES,
    assert_aspect_fit,
    fake_renderer,
    gallery_geometry_cards,
    renderer_640x480,
)


@dataclass
class DummyParams:
    x: int = 1


@dataclass
class DummyPostParams:
    p: int = 2


@pytest.fixture
def make_ctrl() -> Iterator[Callable[[], MagicMock]]:
    """Retain every Local authority until this case retires its owned views."""
    owned_authorities: list[ControlIdentity] = []

    def create_ctrl() -> MagicMock:
        ctrl = MagicMock()
        ctrl.control_identity = ControlIdentity()
        owned_authorities.append(ctrl.control_identity)
        configure_form_lookup(ctrl)
        ctrl.get_left_panel_width.return_value = 500
        ctrl.get_tab_adapter_name.return_value = "fake"
        ctrl.get_adapter_guide.return_value = AdapterGuide(
            markdown="", resource_dir=None
        )
        ctrl.progress_control.attach_progress.return_value = lambda: None
        ctrl.progress_control.progress_bars.return_value = []
        ctrl.open_seeded_cfg_editor.return_value = ("editor-id", ())
        ctrl.get_cfg_editor_draft.return_value = MagicMock()
        ctrl.active_operation_count.return_value = 0

        return ctrl

    try:
        yield create_ctrl
    finally:
        owned_authorities.clear()


def make_snapshot(tab_id: str, *, analysis=AnalysisMode.FIT, post=False, load=False):
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
    )

    caps = AdapterCapabilities(analysis=analysis, post_analysis=post, load_data=load)
    run_snap = RunPaneSnapshot(result=object(), source_path=None)
    analysis_snap = AnalysisPaneSnapshot(
        params=DummyParams() if analysis is not AnalysisMode.NONE else None,
        result=object() if analysis is not AnalysisMode.NONE else None,
        figures=None,
        writeback_items=(),
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=DummyPostParams() if post else None,
        result=object() if post else None,
        figures=None,
        writeback_items=(),
    )
    save_snap = SavePaneSnapshot()
    snapshot = TabSnapshot(
        adapter_name="fake",
        cfg_input={},
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=analysis is not AnalysisMode.NONE,
            has_figure=False,
            has_post_analyze_result=post,
        ),
        capabilities=caps,
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=save_snap,
    )
    return with_artifacts(snapshot)


@pytest.fixture
def exp_tab_widget(qapp, make_ctrl):
    import zcu_tools.gui.app.measure.ui.exp_tab_widget as mod

    owned_tabs: list[mod.ExpTabWidget] = []

    def create_tab(tab_id, ctrl, capabilities, *, preview_renderer=None):
        tab = mod.ExpTabWidget(
            tab_id, ctrl, capabilities, preview_renderer=preview_renderer
        )
        owned_tabs.append(tab)
        return tab

    try:
        yield create_tab
    finally:
        for tab in owned_tabs:
            tab.close()
            tab.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        owned_tabs.clear()


def test_exp_data_subtab_shows_gallery_run_always(qapp, exp_tab_widget, make_ctrl):
    ctrl = make_ctrl()
    # NONE
    caps_none = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    snap_none = make_snapshot("t1", analysis=AnalysisMode.NONE, post=False)
    tab_none = exp_tab_widget("t1", ctrl, caps_none)
    tab_none.attach(snap_none, MagicMock())
    # Data tab exists
    left_tabs = tab_none.findChild(QTabWidget)
    assert left_tabs is not None
    assert any(left_tabs.tabText(i) == "Data" for i in range(left_tabs.count()))
    # Switch to Data
    tab_none.select_pane("data")
    assert right_stack(tab_none).currentWidget() is gallery(tab_none)
    assert gallery(tab_none).has_card("run")
    assert not gallery(tab_none).has_card("analysis")
    assert gallery(tab_none).card_state("run") == "empty"
    assert "No figure" in gallery(tab_none).card_text("run")
    assert isinstance(gallery(tab_none).findChild(QScrollArea), QScrollArea)
    tab_none.detach()

    # FIT + post
    caps_both = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    snap_both = make_snapshot("t2", analysis=AnalysisMode.FIT, post=True)
    tab_both = exp_tab_widget("t2", ctrl, caps_both)
    tab_both.attach(snap_both, MagicMock())
    tab_both.select_pane("data")
    assert gallery(tab_both).has_card("run")
    assert gallery(tab_both).has_card("analysis")
    assert gallery(tab_both).has_card("post_analysis")
    assert gallery(tab_both).card_count() == 3
    tab_both.detach()


def test_exp_data_routing_a4(qapp, exp_tab_widget, make_ctrl):
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=True)
    tab = exp_tab_widget("t1", ctrl, caps)
    tab.attach(snap, MagicMock())

    # Run
    tab.select_pane("run")
    assert right_stack(tab).currentWidget() is right_page(tab, "run")
    # Analysis
    tab.select_pane("analysis")
    assert right_stack(tab).currentWidget() is right_page(tab, "analysis")
    # Post
    tab.select_pane("post_analysis")
    assert right_stack(tab).currentWidget() is right_page(tab, "post_analysis")
    # Data -> gallery
    tab.select_pane("data")
    assert right_stack(tab).currentWidget() is gallery(tab)
    # Guide -> placeholder
    left_tabs = tab.findChild(QTabWidget)
    assert left_tabs is not None
    guide_idx = next(
        i for i in range(left_tabs.count()) if left_tabs.tabText(i) == "Guide"
    )
    left_tabs.setCurrentIndex(guide_idx)
    assert right_stack(tab).currentWidget() is right_page(tab, "guide")
    tab.detach()


def test_exp_data_activation_refreshes_and_lifecycle_sync(
    qapp, exp_tab_widget, make_ctrl
):
    ctrl = make_ctrl()
    # Use fake renderer so we can assert raster not empty after show
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    tab = exp_tab_widget("t1", ctrl, caps, preview_renderer=fake_renderer)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=True)
    tab.attach(snap, MagicMock())

    # Initially no figures -> empty
    tab.select_pane("data")
    assert gallery(tab).card_state("run") == "empty"
    assert gallery(tab).card_state("analysis") == "empty"

    # Data visible: show figures should sync immediately
    fig_run = Figure()
    fig_a = Figure()
    fig_p = Figure()
    # Capture container identities before
    run_c = tab.get_run_container()
    ana_c = tab.get_analysis_container()
    post_c = tab.get_post_container()
    tab.show_run_figure(fig_run)
    assert tab.get_current_figure_for_pane("run") is fig_run
    assert tab.get_run_container() is run_c
    assert gallery(tab).card_state("run") == "available"

    tab.show_analysis_figures(ready_figures(fig_a))
    assert tab.get_current_figure_for_pane("analysis") is fig_a
    assert tab.get_analysis_container() is ana_c
    assert gallery(tab).card_state("analysis") == "available"

    tab.show_post_analysis_figures(ready_figures(fig_p))
    assert tab.get_current_figure_for_pane("post_analysis") is fig_p
    assert tab.get_post_container() is post_c
    assert gallery(tab).card_state("post_analysis") == "available"

    # Data-visible clear should revert to empty
    tab.prepare_run_container()
    assert tab.get_current_figure_for_pane("run") is None
    assert tab.get_run_container() is run_c
    assert gallery(tab).card_state("run") == "empty"
    # downstream also cleared
    assert gallery(tab).card_state("analysis") == "empty"
    assert gallery(tab).card_state("post_analysis") == "empty"

    # Re-show
    fig_run2 = Figure()
    tab.show_run_figure(fig_run2)
    assert gallery(tab).card_state("run") == "available"
    # clear_post_figure only affects post
    fig_a2 = Figure()
    fig_p2 = Figure()
    tab.show_analysis_figures(ready_figures(fig_a2))
    tab.show_post_analysis_figures(ready_figures(fig_p2))
    assert gallery(tab).card_state("post_analysis") == "available"
    tab.clear_post_figure()
    assert gallery(tab).card_state("post_analysis") == "empty"
    assert gallery(tab).card_state("analysis") == "available"
    assert tab.get_current_figure_for_pane("analysis") is fig_a2
    assert tab.get_analysis_container() is ana_c
    tab.detach()


def test_exp_data_invisible_defers_render(qapp, exp_tab_widget, make_ctrl):
    ctrl = make_ctrl()
    call_log: list[str] = []

    def counting_renderer(fig):
        call_log.append("render")
        return bytes(FAKE_PNG_BYTES)

    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)
    tab = exp_tab_widget("t1", ctrl, caps, preview_renderer=counting_renderer)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=False)
    tab.attach(snap, MagicMock())

    # Start on Run (Data not visible)
    tab.select_pane("run")
    assert right_stack(tab).currentWidget() is right_page(tab, "run")
    call_log.clear()
    fig_run = Figure()
    tab.show_run_figure(fig_run)
    # No render while Data invisible
    assert call_log == []
    # Switch to Data -> should render once
    tab.select_pane("data")
    assert len(call_log) == 1
    assert gallery(tab).card_state("run") == "available"
    tab.detach()


def test_exp_production_reachability_data_gallery_is_active_right_widget(
    qapp, exp_tab_widget, make_ctrl
):
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    tab = exp_tab_widget("reach", ctrl, caps)
    snap = make_snapshot("reach", analysis=AnalysisMode.FIT, post=True)
    actions = MagicMock()
    tab.attach(snap, actions)
    # Select the real Data pane after checking its visible tab label.
    left_tabs = tab.findChild(QTabWidget)
    assert left_tabs is not None
    data_idx = None
    for i in range(left_tabs.count()):
        if left_tabs.tabText(i) == "Data":
            data_idx = i
            break
    assert data_idx is not None
    tab.select_pane("data")
    assert right_stack(tab).currentWidget() is gallery(tab)
    # The Data pane still submits a whole Run artifact through its Save button.
    buttons = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(buttons) == 3
    assert buttons[0].isEnabled()
    buttons[0].click()
    assert [call.args for call in actions.save_artifact.call_args_list] == [
        ("reach", ArtifactRef("run", "measurement_data"))
    ]
    tab.detach()


def test_exp_gallery_failure_isolated_and_save_controls_remain_usable(
    qapp, exp_tab_widget, make_ctrl
):
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    fig_run = Figure()
    fig_a = Figure()
    fig_p = Figure()

    def fail_analysis(fig):
        if fig is fig_a:
            raise RuntimeError("analysis boom")
        return bytes(FAKE_PNG_BYTES)

    tab = exp_tab_widget("t1", ctrl, caps, preview_renderer=fail_analysis)
    snap = make_snapshot("t1", analysis=AnalysisMode.FIT, post=True)
    actions = MagicMock()
    tab.attach(snap, actions)
    # Go to Data
    tab.select_pane("data")
    # Show figures while Data visible
    tab.show_run_figure(fig_run)
    tab.show_analysis_figures(ready_figures(fig_a))
    tab.show_post_analysis_figures(ready_figures(fig_p))
    # Analysis should be unavailable, others available
    assert gallery(tab).card_state("run") == "available"
    assert gallery(tab).card_state("analysis") == "unavailable"
    assert gallery(tab).card_state("post_analysis") == "available"
    # Raster failure must not disable whole-artifact Save or Save All commands.
    buttons = [
        button for button in tab.findChildren(QPushButton) if button.text() == "Save"
    ]
    assert len(buttons) == 3
    assert all(button.isEnabled() for button in buttons)
    for button in buttons:
        button.click()
    assert [call.args for call in actions.save_artifact.call_args_list] == [
        ("t1", ArtifactRef("run", "measurement_data")),
        ("t1", ArtifactRef("analysis", "results")),
        ("t1", ArtifactRef("post_analysis", "results")),
    ]
    save_all = next(
        button
        for button in tab.findChildren(QPushButton)
        if button.text() == "Save All"
    )
    assert save_all.isEnabled()
    save_all.click()
    assert [call.args for call in actions.save_all.call_args_list] == [("t1",)]
    tab.detach()


def test_gallery_aspect_fit_preserved_after_multiple_resizes_via_exp_tab(
    qapp, exp_tab_widget, make_ctrl
):
    """Via shipped ExpTabWidget Data path, verify bounds and ratio before/after resize."""
    ctrl = make_ctrl()
    caps = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    tab = exp_tab_widget("t-aspect", ctrl, caps, preview_renderer=renderer_640x480)
    snap = make_snapshot("t-aspect", analysis=AnalysisMode.NONE, post=False)
    tab.attach(snap, MagicMock())
    # Switch to Data so gallery is active right widget
    tab.select_pane("data")
    assert right_stack(tab).currentWidget() is gallery(tab)
    tab.show()
    tab.resize(500, 600)
    QApplication.processEvents()
    QApplication.processEvents()
    fig = Figure()
    tab.show_run_figure(fig)
    QApplication.processEvents()
    card = gallery(tab).find_card("run")
    assert card is not None
    vp = card.image_viewport_size()
    disp = card.displayed_pixmap()
    orig = card.original_pixmap()
    assert vp is not None and disp is not None and orig is not None
    assert_aspect_fit(vp, disp.size(), orig.size())
    # Resize larger — simulate user dragging splitter wider (still gallery viewport, not window)
    gallery(tab).setFixedWidth(700)
    QApplication.processEvents()
    QApplication.processEvents()
    vp2 = card.image_viewport_size()
    disp2 = card.displayed_pixmap()
    assert vp2 is not None and disp2 is not None
    assert_aspect_fit(vp2, disp2.size(), orig.size())
    tab.detach()


def _assert_narrow_gallery_geometries(
    geoms: dict[str, QRect], expected_count: int
) -> None:
    """Assert narrow layout geometry for the expected gallery card count."""
    if expected_count == 1:
        r = geoms["run"]
        assert r.x() <= 4
    elif expected_count == 2:
        r = geoms["run"]
        a = geoms["analysis"]
        assert abs(r.x() - a.x()) <= 2, (
            f"2-card narrow x not aligned {r.x()} vs {a.x()}"
        )
        assert r.y() < a.y(), f"2-card narrow y not monotonic {r.y()} vs {a.y()}"
    else:  # 3
        r = geoms["run"]
        a = geoms["analysis"]
        p = geoms["post_analysis"]
        assert abs(r.x() - a.x()) <= 2 and abs(a.x() - p.x()) <= 2, (
            "3-card narrow x not aligned"
        )
        assert r.y() < a.y() < p.y(), "3-card narrow y not monotonic"


def _assert_wide_gallery_geometries(
    geoms: dict[str, QRect], expected_count: int
) -> None:
    """Assert wide layout geometry for the expected gallery card count."""
    if expected_count == 1:
        r = geoms["run"]
        assert r.x() <= 4
        assert r.width() > 400  # wide single card expands
    elif expected_count == 2:
        r = geoms["run"]
        a = geoms["analysis"]
        assert abs(r.y() - a.y()) <= 4, f"2-card wide y not aligned {r.y()} vs {a.y()}"
        assert a.x() > r.x() + r.width() - 4, "2-card wide not side-by-side"
        assert abs(r.width() - a.width()) <= 20
    else:
        r = geoms["run"]
        a = geoms["analysis"]
        p = geoms["post_analysis"]
        assert r.x() < a.x() - 2, f"Run not left of Analysis {r.x()} vs {a.x()}"
        assert abs(a.x() - p.x()) <= 2
        assert abs(r.y() - a.y()) <= 4
        assert p.y() > a.y() + 2
        assert r.height() > a.height() + 2


def test_gallery_narrow_vs_wide_via_exp_tab_shipped_path(
    qapp, exp_tab_widget, make_ctrl
):
    """Geometry via shipped ExpTabWidget Data path for each capability combination (A1).

    Verifies automatic viewport-driven reflow (gallery viewport, not test-driven
    private call) across 1/2/3 cards through the shipped ExpTabWidget resize.
    """

    def _check_one(caps: AdapterCapabilities, expected_count: int) -> None:
        ctrl = make_ctrl()
        tag = f"t-geom-{expected_count}-{caps.analysis}-{caps.post_analysis}"
        tab = exp_tab_widget(tag, ctrl, caps, preview_renderer=fake_renderer)
        snap = make_snapshot(tag, analysis=caps.analysis, post=caps.post_analysis)
        tab.attach(snap, MagicMock())
        tab.select_pane("data")
        assert right_stack(tab).currentWidget() is gallery(tab)
        tab.show()
        QApplication.processEvents()
        QApplication.processEvents()
        g = gallery(tab)
        assert g.card_count() == expected_count

        # Narrow via ExpTabWidget resize — automatic reflow, no private call.
        tab.resize(650, 600)
        QApplication.processEvents()
        QApplication.processEvents()
        QApplication.processEvents()
        assert not g.is_wide_mode(), (
            f"expected narrow at 650 for {expected_count} cards"
        )
        geoms = gallery_geometry_cards(g)
        _assert_narrow_gallery_geometries(geoms, expected_count)

        # Wide via ExpTabWidget resize — automatic.
        tab.resize(1200, 600)
        QApplication.processEvents()
        QApplication.processEvents()
        QApplication.processEvents()
        assert g.is_wide_mode(), f"expected wide at 1200 for {expected_count} cards"
        geoms = gallery_geometry_cards(g)
        _assert_wide_gallery_geometries(geoms, expected_count)
        tab.detach()
        g.hide()
        QApplication.processEvents()

    # 1 card: Run only
    caps1 = AdapterCapabilities(analysis=AnalysisMode.NONE, post_analysis=False)
    _check_one(caps1, 1)
    # 2 cards: Run + Analysis
    caps2 = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)
    _check_one(caps2, 2)
    # 3 cards: Run + Analysis + Post
    caps3 = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
    _check_one(caps3, 3)
