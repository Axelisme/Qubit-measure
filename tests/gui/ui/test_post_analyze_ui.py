"""UI tests for the post-analysis (second-layer) sub-tab.

Mirrors the primary-analyze UI tests (``test_main_window_ui.py``): the Post
sub-tab only appears for adapters declaring ``post_analysis``, its form/Run gate
on a primary analyze result, Run dispatches through the controller, and a
re-analyze invalidation clears the post figure.
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
) -> TabSnapshot:
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
        figure=None,
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
    """A content event with a post result populates the post form and shows the
    post figure in the separate post container."""
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload
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
        "zcu_tools.gui.app.main.ui.main_window.attach_existing_figure_to_container",
        lambda fig, container: captured.setdefault("canvas", MagicMock()),
    )

    bus.emit(TabContentChangedPayload(tab_id="tab-1"))

    assert tab_w.has_post_analyze_params() is True
    assert captured.get("canvas") is not None


def test_post_figure_cleared_on_invalidation(qapp):
    """A content event whose snapshot has no post figure (re-run / re-analyze
    invalidated the post result) drops the stale post canvas."""
    from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _mock_ctrl()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_post_analyze_result=False, post_figure=None
    )

    window = MainWindow(ctrl)
    tab_w = MagicMock()
    window._tab_widgets["tab-1"] = tab_w

    bus.emit(TabContentChangedPayload(tab_id="tab-1"))

    tab_w.reset_post_plot.assert_called()


def test_make_post_container_is_separate_from_primary(qapp):
    """The post container must be a different object than the primary live
    container, so the two figures never overwrite each other."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _mock_ctrl()
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab_w = ExpTabWidget("tab-1", ctrl)
    window._tab_widgets["tab-1"] = tab_w

    primary = window.make_live_container("tab-1")
    post = window.make_post_live_container("tab-1")

    assert primary is tab_w._figure_container
    assert post is tab_w._post_figure_container
    assert primary is not post


def test_post_tab_uses_separate_qt_tab_index(qapp):
    """The Post sub-tab is a distinct QTabWidget tab (not Analysis at index 1)."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())

    assert tab._post_tab_index != 1
    assert tab._left_tabs.tabText(tab._post_tab_index) == "Post-Analysis"
    # Analysis stays at index 1 (the auto-switch + label logic depends on it).
    assert tab._left_tabs.tabText(1) == "Analysis"
