"""Focused visual corrections for run-analysis-visual-corrections (A1, A2, A4-A6).

Observable Qt behavior tests — no prose or stylesheet-source assertions.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QWidget,
)
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.widgets.cfg import CfgFormWidget, TreeCfgWidget
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure._cfg_fakes import configure_cfg_lookup
from tests.gui.app.measure.ui._artifact_snapshots import ready_figures, with_artifacts


def make_ctrl():
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    md = MetaDict()
    md.r_f = 6000.0
    ml = ModuleLibrary()
    exp_ctx = MagicMock()
    exp_ctx.md = md
    exp_ctx.ml = ml
    ctrl.get_session_env.return_value = exp_ctx
    return ctrl


def make_snapshot(tab_id, *, analysis=AnalysisMode.FIT, post=False):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )

    data_path = PathResourceSnapshot(override=None, path="/tmp/data.hdf5")
    image = PathResourceSnapshot(override=None, path="/tmp/img.png")
    run_snap = RunPaneSnapshot(result=object(), source_path=None)
    analysis_snap = AnalysisPaneSnapshot(
        params=MagicMock(),
        result=object(),
        figures=ready_figures(Figure()),
        writeback_items=(),
        image_paths={"fit": image},
        has_writeback_draft=False,
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=None,
        result=None,
        figures=None,
        writeback_items=(),
        image_paths={},
        has_writeback_draft=False,
    )
    spec = CfgSectionSpec(
        label="root", fields={"reps": ScalarSpec(label="Reps", type=int)}
    )
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(100)})
    )

    @dataclasses.dataclass
    class P:
        thr: float = 0.5

    analysis_snap = dataclasses.replace(analysis_snap, params=P())
    snapshot = TabSnapshot(
        adapter_name="fake",
        cfg_schema=schema,
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
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
        save=SavePaneSnapshot(data_path=data_path),
        paths=TabPathsSnapshot(
            data=data_path, analysis_images={"fit": image}, post_analysis_images={}
        ),
    )
    return with_artifacts(snapshot)


@pytest.fixture
def exp_tab_widget(qapp, monkeypatch):
    import zcu_tools.gui.app.measure.ui.exp_tab_widget as mod

    orig_attach = mod.attach_existing_figure_to_container

    def mock_attach(fig, container):
        from qtpy.QtWidgets import QWidget

        w = QWidget()
        w.figure = fig
        container.attach_canvas(w)
        w.draw = lambda: None
        return w

    monkeypatch.setattr(mod, "attach_existing_figure_to_container", mock_attach)
    yield mod.ExpTabWidget
    monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)


def test_A2_sole_tree_has_no_structure_selector(qapp):
    """A2: CfgFormWidget sole tree — no public structure selector, default is tree."""
    # CfgFormWidget must reject a structure kwarg
    with pytest.raises(TypeError):
        CfgFormWidget(structure=object())  # type: ignore[call-arg]
    import zcu_tools.gui.widgets.cfg as cfg_pkg

    assert not hasattr(cfg_pkg, "form_structure")
    assert not hasattr(cfg_pkg, "FormStructure")
    assert not hasattr(cfg_pkg, "tree_structure")
    # Default construction is tree
    ctrl = MagicMock()
    configure_cfg_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = MetaDict()
    ctrl.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    ctrl.arb_waveforms.list_data_keys.return_value = []
    ctrl.list_device_names.return_value = []
    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)}),
        value=CfgSectionValue(fields={"reps": DirectValue(10)}),
    )
    w = CfgFormWidget()
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    assert isinstance(w._root_widget, TreeCfgWidget)
    w.detach()
    draft.close()


def test_A4_cfg_viewport_expands_with_panel_height(qapp, exp_tab_widget):
    """A4: Run cfg tree viewport follows panel height, scrolls only when content exceeds viewport."""
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    qapp.processEvents()
    assert tab.cfg_form.maximumHeight() >= 10000
    assert tab.cfg_form.minimumHeight() <= 50

    ctrl2 = MagicMock()
    ctrl2.get_bus.return_value = EventBus()
    ctrl2.get_current_md.return_value = MetaDict()
    ctrl2.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    ctrl2.arb_waveforms.list_data_keys.return_value = []
    ctrl2.list_device_names.return_value = []
    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)}),
        value=CfgSectionValue(fields={"reps": DirectValue(10)}),
    )
    w = CfgFormWidget()
    draft = MeasureCfgBindings(ctrl2).new_draft(schema)
    w.attach(draft)
    assert w._scroll.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert w._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    tree_w = w._root_widget
    assert isinstance(tree_w, TreeCfgWidget)
    assert tree_w._tree.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    assert tree_w.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert tree_w._tree.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert w._inner_layout.count() == 2
    assert w._inner_layout.stretch(0) == 1
    assert w._inner_layout.stretch(1) == 0
    parent = QWidget()
    layout = QHBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(w, stretch=1)
    parent.resize(400, 300)
    parent.show()
    qapp.processEvents()
    h_small = w.height()
    parent.resize(400, 600)
    qapp.processEvents()
    h_large = w.height()
    assert h_large > h_small
    parent.close()
    w.detach()
    draft.close()
    tab.detach()


def test_A5_Run_action_row_status_free_and_20_80_proportions(qapp, exp_tab_widget):
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
        assert "Ready" not in lb.text() or lb is tab.findChild(QLabel, "runActionBar")  # type: ignore

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
    reset_style = tab.reset_btn.styleSheet()
    assert "background-color" in reset_style
    assert ":hover" in reset_style
    assert ":pressed" in reset_style
    assert ":disabled" in reset_style
    assert not tab.reset_btn.isHidden()
    assert tab.reset_btn.sizeHint().height() == tab.run_btn.sizeHint().height()
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


def test_A6_Analyze_full_width_below_params_before_writeback(qapp, exp_tab_widget):
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
    idx_analyze = widgets.index(tab.analyze_btn)
    idx_writeback = widgets.index(tab.writeback_section)
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
                if item.widget() is tab.analyze_btn:
                    # check if previous item is spacer
                    if i > 0 and parent_layout.itemAt(i - 1).spacerItem() is not None:
                        has_stretch_before = True
                    break
            assert not has_stretch_before, (
                "Analyze should not have stretch before it (should be 100% width)"
            )

    # Availability: when idle with context & run result, enabled
    assert tab.analyze_btn.isEnabled() is True
    # Busy disables
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
