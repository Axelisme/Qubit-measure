"""Focused visual corrections for run-analysis-visual-corrections (A1, A2, A4-A6).

Observable Qt behavior tests — no prose or stylesheet-source assertions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.control_identity import ControlIdentity

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure.artifacts._snapshots import ready_figures, with_artifacts


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
        exp_ctx = MagicMock()
        exp_ctx.parameters = parameter_capture(r_f=6000.0)
        ctrl.get_session_env.return_value = exp_ctx
        return ctrl

    try:
        yield create_ctrl
    finally:
        owned_authorities.clear()


def make_snapshot(tab_id, *, analysis=AnalysisMode.FIT, post=False):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
    )

    run_snap = RunPaneSnapshot(result=object(), source_path=None)
    analysis_snap = AnalysisPaneSnapshot(
        params=MagicMock(),
        result=object(),
        figures=ready_figures(Figure()),
        writeback_items=(),
        has_writeback_draft=False,
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=None,
        result=None,
        figures=None,
        writeback_items=(),
        has_writeback_draft=False,
    )

    @dataclasses.dataclass
    class P:
        thr: float = 0.5

    analysis_snap = dataclasses.replace(analysis_snap, params=P())
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
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=False,
        ),
        capabilities=AdapterCapabilities(analysis=analysis, post_analysis=post),
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=SavePaneSnapshot(),
    )
    return with_artifacts(snapshot)


@pytest.fixture
def exp_tab_widget(qapp, make_ctrl):
    import zcu_tools.gui.app.measure.ui.exp_tab_widget as mod

    owned_tabs: list[mod.ExpTabWidget] = []

    def create_tab(
        tab_id: str, ctrl: MagicMock, capabilities: AdapterCapabilities
    ) -> mod.ExpTabWidget:
        tab = mod.ExpTabWidget(tab_id, ctrl, capabilities)
        owned_tabs.append(tab)
        return tab

    try:
        yield create_tab
    finally:
        for tab in owned_tabs:
            tab.deleteLater()
        owned_tabs.clear()


def test_A5_Run_action_row_status_free_and_20_80_proportions(
    qapp, exp_tab_widget, make_ctrl
):
    """A5: Run action row has no status text; Reset 20% / Run 80% with retained treatments."""
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    qapp.processEvents()

    # No readiness/status label in the run panel
    assert not hasattr(tab, "_run_status_label")
    # No label with readyStatus objectName or "Ready" text inside run panel
    labels = tab._run_panel.findChildren(QLabel)
    for lb in labels:
        assert lb.objectName() != "readyStatus"
        assert "Ready" not in lb.text() or lb is tab.findChild(QLabel, "runActionBar")

    # Run action bar exists and is QFrame with 20/80 stretch
    assert hasattr(tab, "_run_action_bar")
    assert isinstance(tab._run_action_bar, QFrame)
    bar = tab._run_action_bar
    layout = bar.layout()
    assert isinstance(layout, QHBoxLayout)
    # Check stretch factors for the two buttons
    # layout has two widgets: reset_btn stretch 20, run_btn stretch 80
    found_reset = False
    found_run = False
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget() if item is not None else None
        if w is tab.reset_btn:
            assert layout.stretch(i) == 20, (
                f"Reset stretch should be 20, got {layout.stretch(i)}"
            )
            found_reset = True
        if w is tab.run_btn:
            assert layout.stretch(i) == 80, (
                f"Run stretch should be 80, got {layout.stretch(i)}"
            )
            found_run = True
    assert found_reset and found_run
    # Both buttons should be Expanding horizontally so they fill the proportion
    assert tab.reset_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert tab.run_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding

    # Reset and Run text present, Run has primaryButton, Stop semantics
    assert tab.reset_btn.text() == "Reset"
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.objectName() == "primaryButton"
    idle_style = tab.run_btn.styleSheet()
    assert not tab.reset_btn.isHidden()
    assert tab.reset_btn.height() == tab.run_btn.height()

    # When running, Reset disappears and Stop fills the action row.
    assert snap.interaction is not None
    busy = dataclasses.replace(
        snap,
        interaction=dataclasses.replace(snap.interaction, is_running=True),
    )
    tab.update_interaction_state(busy)
    qapp.processEvents()
    assert tab.run_btn.text() == "Stop"
    # style should have changed (Stop vs Run)
    assert tab.run_btn.styleSheet() != idle_style
    assert tab.reset_btn.isHidden()
    for i in range(layout.count()):
        item = layout.itemAt(i)
        assert item is not None
        if item.widget() is tab.reset_btn:
            assert layout.stretch(i) == 0
        if item.widget() is tab.run_btn:
            assert layout.stretch(i) == 100

    # back to idle restores primary and 20/80 geometry
    tab.update_interaction_state(snap)
    qapp.processEvents()
    assert not tab.reset_btn.isHidden()
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.objectName() == "primaryButton"
    for i in range(layout.count()):
        item = layout.itemAt(i)
        assert item is not None
        if item.widget() is tab.reset_btn:
            assert layout.stretch(i) == 20
        if item.widget() is tab.run_btn:
            assert layout.stretch(i) == 80
    tab.detach()


def test_A6_Analyze_full_width_below_params_before_writeback(
    qapp, exp_tab_widget, make_ctrl
):
    """A6: Analyze immediately below params and before Writeback, 100% width, availability preserved."""
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    qapp.processEvents()

    # Analyze should be directly inside scroll inner, not in a fixed bar
    scroll = tab._analysis_panel.findChild(QScrollArea)
    assert scroll is not None
    inner = scroll.widget()
    assert inner is not None

    def is_descendant(widget, ancestor):
        cur = widget
        while cur is not None:
            if cur is ancestor:
                return True
            cur = cur.parent()
        return False

    assert is_descendant(tab.analyze_btn, inner), "Analyze should be inside scroll"
    # Place Analyze after the parameters and before Writeback.
    layout = inner.layout()
    assert layout is not None
    widgets = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None:
            widgets.append(item.widget())
    idx_params = widgets.index(tab._analyze_section)
    # Analyze btn is directly a child of inner layout (100% width), not inside extra container
    # Check that analyze_btn is directly in widgets list
    assert tab.analyze_btn in widgets, (
        "Analyze should be directly in inner layout for 100% width"
    )
    idx_analyze, idx_writeback = (
        widgets.index(tab.analyze_btn),
        widgets.index(tab.writeback_section),
    )
    assert idx_params < idx_analyze < idx_writeback

    # 100% width: sizePolicy Expanding horizontally
    assert (
        tab.analyze_btn.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    )
    # Must expand to fill available width — check that minimum width is not the old 94 fixed right-aligned style
    # The button should not be inside a QHBoxLayout with stretch before it
    parent = tab.analyze_btn.parent()
    if parent is not None:
        parent_layout = parent.layout()
        if isinstance(parent_layout, QHBoxLayout):
            # Should not have a stretch before the button
            has_stretch_before = False
            for i in range(parent_layout.count()):
                item = parent_layout.itemAt(i)
                assert item is not None
                if item.widget() is tab.analyze_btn:
                    # check if previous item is spacer
                    if i > 0:
                        previous_item = parent_layout.itemAt(i - 1)
                        assert previous_item is not None
                        if previous_item.spacerItem() is not None:
                            has_stretch_before = True
                    break
            assert not has_stretch_before, (
                "Analyze should not have stretch before it (should be 100% width)"
            )

    # Availability: when idle with context & run result, enabled
    assert tab.analyze_btn.isEnabled() is True
    # Busy disables
    assert snap.interaction is not None
    busy = dataclasses.replace(
        snap,
        interaction=dataclasses.replace(snap.interaction, is_analyzing=True),
    )
    tab.update_interaction_state(busy)
    assert tab.analyze_btn.isEnabled() is False
    assert tab.analyze_form.isEnabled() is False
    # Also verify that writeback availability still governed correctly
    assert tab.writeback_widget.isEnabled() is False
    tab.detach()
