"""Feedback panel mount and gate behavior in MainWindow."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.event_bus import BaseEventBus as EventBus


def _gate_window(
    qapp,
    *,
    op_count: int,
    agent_connected: bool,
    running_tab_id: str | None = None,
    active_tab_id: str | None = None,
):
    """Build a MainWindow + register a real ExpTabWidget per provided tab id.

    Returns (window, {tab_id: ExpTabWidget}). The C3 gate inputs and the
    running/active-tab resolution are stubbed on the mock controller; the tab
    widgets are real so mount_feedback_panel docks into a live plot_layout.
    """
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget, MainWindow

    del qapp
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_left_panel_width.return_value = 500
    ctrl.active_operation_count.return_value = op_count
    ctrl.has_agent_connected.return_value = agent_connected
    ctrl.can_cancel_active_operation.return_value = False
    ctrl.get_running_tab_id.return_value = running_tab_id
    ctrl.get_active_tab_id.return_value = active_tab_id
    ctrl.has_tab.side_effect = lambda tid: tid in tabs

    window = MainWindow(ctrl)

    tabs: dict[str, ExpTabWidget] = {}
    for tid in {t for t in (running_tab_id, active_tab_id) if t is not None}:
        tab_w = ExpTabWidget(
            tid,
            ctrl,
            AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
        )
        tabs[tid] = tab_w
        window._tab_widgets[tid] = tab_w
    return window, tabs


def _panel_docked_below_stack(window, tab_w) -> bool:
    """True iff the window's feedback panel sits at plot_layout index 1, i.e.
    directly below the figure host (index 0)."""
    layout = tab_w._plot_layout
    panel = _feedback_panel(window)
    return layout.indexOf(panel) == 1 and layout.indexOf(tab_w._right_stack) == 0


def _feedback_panel(window):
    return window._feedback_dock.panel


def _feedback_host_tab(window):
    return window._feedback_dock.host_tab


def test_feedback_panel_unmounted_when_op_active_but_no_agent(qapp):
    """C3 gate: active op alone is not enough — agent must also be connected."""
    window, tabs = _gate_window(
        qapp, op_count=1, agent_connected=False, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None
    assert tabs["tab-1"]._plot_layout.indexOf(_feedback_panel(window)) == -1


def test_feedback_panel_unmounted_when_agent_connected_but_no_op(qapp):
    """C3 gate: agent connected alone is not enough — op must also be live."""
    window, tabs = _gate_window(
        qapp, op_count=0, agent_connected=True, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None
    assert tabs["tab-1"]._plot_layout.indexOf(_feedback_panel(window)) == -1


def test_feedback_panel_unmounted_when_no_op_and_no_agent(qapp):
    """Both conditions false: panel stays unmounted."""
    window, tabs = _gate_window(
        qapp, op_count=0, agent_connected=False, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None


def test_feedback_panel_unmounted_when_no_tabs(qapp):
    """Edge case: gate true but no target tab → panel stays unmounted."""
    window, _ = _gate_window(qapp, op_count=1, agent_connected=True)
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None


def test_feedback_panel_mounted_below_figure_when_gate_true(qapp):
    """C3 gate satisfied: panel docks into the active tab's plot_layout at
    index 1 (directly below the plot stack), visible and expanded."""
    window, tabs = _gate_window(
        qapp, op_count=1, agent_connected=True, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()

    tab_w = tabs["tab-1"]
    panel = _feedback_panel(window)
    assert _feedback_host_tab(window) is tab_w
    assert _panel_docked_below_stack(window, tab_w)
    # Default EXPANDED: the collapsible body is not collapsed (toggle checked,
    # body visible relative to the panel — the window itself is not shown in the
    # test, so absolute isVisible() would be False).
    assert panel._toggle_btn is not None
    assert panel._toggle_btn.isChecked()
    assert panel._body.isVisibleTo(panel)


def test_feedback_panel_targets_running_tab_over_active(qapp):
    """Target tab = running tab if one is running, else active tab."""
    window, tabs = _gate_window(
        qapp,
        op_count=1,
        agent_connected=True,
        running_tab_id="run-tab",
        active_tab_id="act-tab",
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is tabs["run-tab"]
    assert tabs["act-tab"]._plot_layout.indexOf(_feedback_panel(window)) == -1


def test_feedback_panel_unmounts_and_clears_input_when_gate_drops(qapp):
    """Gate flips false (agent disconnects): panel unmounts and input clears."""
    window, tabs = _gate_window(
        qapp, op_count=1, agent_connected=True, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    panel = _feedback_panel(window)
    panel._input.setText("pending message")
    assert _feedback_host_tab(window) is tabs["tab-1"]

    cast(MagicMock, window._ctrl).has_agent_connected.return_value = False
    window.refresh_feedback_widget()

    assert _feedback_host_tab(window) is None
    assert tabs["tab-1"]._plot_layout.indexOf(panel) == -1
    assert panel._input.text() == ""


def test_feedback_panel_remounts_on_target_tab_change(qapp):
    """If the target tab changes while visible, the panel re-mounts under the
    new tab (and is removed from the old one)."""
    window, tabs = _gate_window(
        qapp,
        op_count=1,
        agent_connected=True,
        running_tab_id="tab-a",
        active_tab_id="tab-b",
    )
    # Add tab-b as a real tab too (active fallback target after run finishes).
    from zcu_tools.gui.app.measure.ui.main_window import ExpTabWidget

    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is tabs["tab-a"]

    # Run finishes: no running tab now, active tab becomes the target.
    cast(MagicMock, window._ctrl).get_running_tab_id.return_value = None
    if "tab-b" not in tabs:
        tab_b = ExpTabWidget(
            "tab-b",
            window._ctrl,
            AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False),
        )
        tabs["tab-b"] = tab_b
        window._tab_widgets["tab-b"] = tab_b
    window.refresh_feedback_widget()

    assert _feedback_host_tab(window) is tabs["tab-b"]
    assert tabs["tab-a"]._plot_layout.indexOf(_feedback_panel(window)) == -1
    assert _panel_docked_below_stack(window, tabs["tab-b"])
