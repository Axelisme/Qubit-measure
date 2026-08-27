"""UI tests for the post-analysis (second-layer) sub-tab.

Mirrors the primary-analyze UI tests (``test_main_window_ui.py``): the Post
sub-tab only appears for adapters declaring ``post_analysis``, its form/Run gate
on a primary analyze result, Run dispatches through the controller. The post
figure renders into the *shared* right-pane container (run/analyze/post all show
the most recent figure); the post sub-tab has its own Save Image button gated on
a post result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class _PostParams:
    """A minimal post-analysis params dataclass for form population."""

    backend: str = "pca"


@dataclass
class _AnalyzeParams:
    """A minimal primary-analyze params dataclass (the primary form refresh needs
    a real dataclass, not a MagicMock)."""

    backend: str = "pca"


@dataclass
class _EmptyPostParams:
    """A post-analysis params type with no operator-editable fields."""


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    return ctrl


def _snapshot(
    tab_id: str,
    *,
    is_running: bool = False,
    is_analyzing: bool = False,
    has_active_context: bool = True,
    has_run_result: bool = True,
    has_analyze_result: bool = True,
    has_post_analyze_result: bool = False,
    post_analysis: bool = True,
    post_analyze_params: object | None = None,
    post_figure: Figure | None = None,
    figure: Figure | None = None,
) -> TabSnapshot:
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        TabPathsSnapshot,
    )

    analysis_pane = AnalysisPaneSnapshot(
        params=_AnalyzeParams() if has_analyze_result else None,
        result=object() if has_analyze_result else None,
        figure=figure,
        writeback_items=(),
        image_path=PathResourceSnapshot(
            override=None, path="/tmp/a.png" if has_analyze_result else None
        ),
    )
    post_analysis_pane = PostAnalysisPaneSnapshot(
        params=post_analyze_params,
        result=object() if has_post_analyze_result else None,
        figure=post_figure,
        writeback_items=(),
        image_path=PathResourceSnapshot(
            override=None, path="/tmp/p.png" if has_post_analyze_result else None
        ),
    )
    paths = TabPathsSnapshot(
        data=PathResourceSnapshot(
            override=None, path="/tmp/data.h5" if has_run_result else None
        ),
        analysis_image=PathResourceSnapshot(
            override=None, path="/tmp/a.png" if has_analyze_result else None
        ),
        post_analysis_image=PathResourceSnapshot(
            override=None, path="/tmp/p.png" if has_post_analyze_result else None
        ),
    )
    return TabSnapshot(
        adapter_name="ge",
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=is_running,
            is_analyzing=is_analyzing,
            is_saving_data=False,
            has_context=True,
            has_active_context=has_active_context,
            has_soc=True,
            has_run_result=has_run_result,
            has_analyze_result=has_analyze_result,
            has_figure=True,
            has_post_analyze_result=has_post_analyze_result,
        ),
        cfg_schema=MagicMock(),
        save_paths_override=None,
        capabilities=AdapterCapabilities(
            analysis=AnalysisMode.FIT, post_analysis=post_analysis
        ),
        analyze_params=_AnalyzeParams(),
        post_analyze_params=post_analyze_params,
        post_figure=post_figure,
        writeback_items=(),
        save_paths=None,
        figure=figure,
        analysis=analysis_pane,
        post_analysis=post_analysis_pane,
        paths=paths,
    )


def test_post_tab_visible_only_for_post_analysis_adapter(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())

    tab.update_interaction_state(_snapshot("tab-1", post_analysis=True))
    assert tab._left_tabs.isTabVisible(tab._post_tab_index) is True

    tab.update_interaction_state(_snapshot("tab-1", post_analysis=False))
    assert tab._left_tabs.isTabVisible(tab._post_tab_index) is False


def test_post_form_disabled_without_primary_analyze_result(qapp):
    """No primary analyze result → the post form + Run are disabled and the gate
    hint is shown."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(_snapshot("tab-1", has_analyze_result=False))

    assert tab.post_analyze_form.isEnabled() is False
    assert tab.post_analyze_btn.isEnabled() is False
    # isHidden reflects the widget's own setVisible state independent of whether
    # an ancestor is shown (the tab is not .show()n here). Gate closed → shown.
    assert tab._post_gate_label.isHidden() is False


def test_post_form_enabled_with_primary_analyze_result(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.show()
    qapp.processEvents()
    tab.update_interaction_state(_snapshot("tab-1", has_analyze_result=True))

    assert tab.post_analyze_form.isEnabled() is True
    assert tab.post_analyze_btn.isEnabled() is True
    # Gate open (primary result present) → hint hidden.
    assert tab._post_gate_label.isHidden() is True


def test_initial_attach_hides_zero_field_post_parameter_section(qapp, monkeypatch):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    monkeypatch.setattr(tab, "_populate_cfg", MagicMock())
    monkeypatch.setattr(tab, "_bind_to_controller", MagicMock())

    tab.attach(
        _snapshot(
            "tab-1",
            has_analyze_result=True,
            post_analyze_params=_EmptyPostParams(),
        ),
        MagicMock(),
    )

    assert tab._post_analyze_section.isHidden() is True
    assert tab.post_analyze_btn.isEnabled() is True


def test_empty_post_params_hide_only_parameter_section_across_gate_states(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.populate_post_analyze_params(_EmptyPostParams())

    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_analyze_result=False,
            has_post_analyze_result=False,
            post_analyze_params=_EmptyPostParams(),
        )
    )
    assert tab._post_analyze_section.isHidden() is True
    assert tab.post_analyze_btn.isHidden() is False
    assert tab.post_analyze_btn.isEnabled() is False
    assert tab._post_gate_label.isHidden() is False
    assert tab.post_save_image_btn.isHidden() is False
    assert tab.post_save_image_btn.isEnabled() is False

    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_analyze_result=True,
            has_post_analyze_result=True,
            post_analyze_params=_EmptyPostParams(),
        )
    )
    assert tab._post_analyze_section.isHidden() is True
    assert tab.post_analyze_btn.isEnabled() is True
    assert tab._post_gate_label.isHidden() is True
    assert tab.post_save_image_btn.isEnabled() is True


def test_post_run_disabled_while_tab_busy(qapp):
    """A busy tab (analyzing) disables the post Run even with a primary result."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot("tab-1", has_analyze_result=True, is_analyzing=True)
    )

    assert tab.post_analyze_btn.isEnabled() is False


def test_post_run_click_starts_post_analyze(qapp):
    """Clicking Run Post-Analysis reads the post params and dispatches through the
    controller's start_post_analyze."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _mock_ctrl()
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)
    tab_w = MagicMock()
    tab_w.read_post_analyze_params.return_value = _PostParams(backend="center")
    window._tab_widgets["tab-1"] = tab_w

    window._on_post_analyze_clicked("tab-1")

    ctrl.start_post_analyze.assert_called_once()
    args = ctrl.start_post_analyze.call_args.args
    assert args[0] == "tab-1"
    assert args[1] == _PostParams(backend="center")


def test_post_content_refresh_populates_form_and_figure(qapp, monkeypatch):
    """A content event with a post result populates the post form and renders the
    post figure into its independent post pane container."""
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _mock_ctrl()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    post_fig = Figure()
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1",
        has_post_analyze_result=True,
        post_analyze_params=_PostParams(backend="pca"),
        post_figure=post_fig,
    )

    window = MainWindow(ctrl)
    tab_w = ExpTabWidget("tab-1", ctrl)
    # Seed the post form so populate_values has a matching cls.
    tab_w.populate_post_analyze_params(_PostParams(backend="pca"))
    window._tab_widgets["tab-1"] = tab_w

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.exp_tab_widget.attach_existing_figure_to_container",
        lambda fig, container: (
            captured.setdefault("container", container) or MagicMock()
        ),
    )

    bus.emit(TabContentChangedPayload("tab-1", TabContentFact.POST_ANALYSIS_COMMITTED))

    assert tab_w.has_post_analyze_params() is True
    # The post figure was attached to its independent post container.
    assert captured.get("container") is tab_w._post_container


def test_post_figure_refresh_is_noop_on_invalidation(qapp, monkeypatch):
    """A content event whose snapshot has no post figure must not touch any container."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _mock_ctrl()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_post_analyze_result=False, post_figure=None, figure=None
    )

    window = MainWindow(ctrl)
    tab_w = ExpTabWidget("tab-1", ctrl)
    window._tab_widgets["tab-1"] = tab_w

    attached: list[object] = []
    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.exp_tab_widget.attach_existing_figure_to_container",
        lambda fig, container: attached.append(container) or MagicMock(),
    )

    # The dedicated post-figure refresh is a no-op when there is no post figure.
    window.refresh_tab_post_figure("tab-1")

    assert attached == []


def test_make_live_container_is_independent_for_post(qapp):
    """Each pane has an independent FigureContainer with stable identity."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _mock_ctrl()
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)

    tab_w = ExpTabWidget("tab-1", ctrl)
    window._tab_widgets["tab-1"] = tab_w

    run_c = window.make_run_container("tab-1")
    ana_c = window.make_analysis_container("tab-1")
    post_c = window.make_post_analysis_container("tab-1")

    assert run_c is tab_w._run_container
    assert ana_c is tab_w._analysis_container
    assert post_c is tab_w._post_container
    # Containers are distinct and stable.
    assert run_c is not ana_c
    assert ana_c is not post_c
    assert run_c is not post_c
    # Legacy alias still points to analysis container for transitional screenshot.
    assert window.make_live_container("tab-1") is tab_w._run_container


def test_take_figure_screenshot_captures_post_figure(qapp):
    """The tab's visible-figure boundary exposes a post figure for screenshots."""
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _mock_ctrl()
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)

    tab_w = ExpTabWidget("tab-1", ctrl)
    window._tab_widgets["tab-1"] = tab_w

    post_fig = Figure()
    post_fig.add_subplot(111).plot([0, 1], [0, 1])
    # Render the post figure through the real shared-container path.
    window.show_post_analysis_image("tab-1", post_fig)

    png = window.take_figure_screenshot("tab-1")

    assert isinstance(png, bytes)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_post_save_image_button_gated_on_post_result(qapp):
    """The post Save Image button enables only with an active context + a post
    result (its figure is the thing it saves)."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.show()
    qapp.processEvents()

    tab.update_interaction_state(
        _snapshot("tab-1", has_analyze_result=True, has_post_analyze_result=False)
    )
    assert tab.post_save_image_btn.isEnabled() is False

    tab.update_interaction_state(
        _snapshot("tab-1", has_analyze_result=True, has_post_analyze_result=True)
    )
    assert tab.post_save_image_btn.isEnabled() is True


def test_post_save_image_button_disabled_without_active_context(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_analyze_result=True,
            has_post_analyze_result=True,
            has_active_context=False,
        )
    )
    assert tab.post_save_image_btn.isEnabled() is False


def test_post_save_image_click_saves_post_figure(qapp):
    """Clicking the post Save Image reads the post image path and dispatches
    through the controller's ``save_post_image`` (which saves ``tab.post_figure``)."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _mock_ctrl()
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)
    tab_w = MagicMock()
    tab_w.get_post_image_path.return_value = "/tmp/post.png"
    window._tab_widgets["tab-1"] = tab_w

    window._on_post_save_image_clicked("tab-1")

    ctrl.save_post_image.assert_called_once_with("tab-1", "/tmp/post.png")


def test_post_tab_uses_separate_qt_tab_index(qapp):
    """The Post sub-tab is a distinct QTabWidget tab (not Analysis at index 1)."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())

    assert tab._post_tab_index != 1
    assert tab._left_tabs.tabText(tab._post_tab_index) == "Post-Analysis"
    # Analysis stays at index 1 (the auto-switch + label logic depends on it).
    assert tab._left_tabs.tabText(1) == "Analysis"
