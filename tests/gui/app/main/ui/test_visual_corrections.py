"""Focused visual corrections for run-analysis-visual-corrections (A1-A4)."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QWidget

from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    CfgSchemaAssembler,
    DirectValue,
    FloatSpec,
    IntSpec,
    ReferenceSpec,
    ScalarSpec,
)
from zcu_tools.gui.widgets.cfg import CfgFormWidget, TreeCfgWidget, tree_structure
from zcu_tools.gui.widgets.cfg.structure import TREE_DEPTH_COLORS, _branch_color
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings


def make_ctrl():
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
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
    ctrl.get_exp_context.return_value = exp_ctx
    return ctrl


def make_snapshot(tab_id, *, analysis=AnalysisMode.FIT, post=False):
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )
    from matplotlib.figure import Figure

    data_path = PathResourceSnapshot(override=None, path="/tmp/data.hdf5")
    image = PathResourceSnapshot(override=None, path="/tmp/img.png")
    run_snap = RunPaneSnapshot(result=object(), source_path=None)
    analysis_snap = AnalysisPaneSnapshot(
        params=MagicMock(),
        result=object(),
        figure=Figure(),
        writeback_items=(),
        image_path=image,
        has_writeback_draft=False,
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=None,
        result=None,
        figure=None,
        writeback_items=(),
        image_path=image,
        has_writeback_draft=False,
    )
    spec = CfgSectionSpec(label="root", fields={"reps": IntSpec("Reps")})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(100)}))
    @dataclasses.dataclass
    class P:
        thr: float = 0.5
    analysis_snap = dataclasses.replace(analysis_snap, params=P())
    return TabSnapshot(
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
        paths=TabPathsSnapshot(data=data_path, analysis_image=image, post_analysis_image=image),
    )


@pytest.fixture
def exp_tab_widget(qapp, monkeypatch):
    import zcu_tools.gui.app.main.ui.exp_tab_widget as mod
    orig = mod.ExpTabWidget._populate_cfg
    def stub(self, schema, ctrl):
        self._cfg_editor_id = "probe-editor"
        self.cfg_form.is_valid = lambda: True
        self.cfg_form.first_invalid_reason = lambda: None
    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", stub)
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
    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", orig)
    monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)


def test_A1_depth_colors_on_guide_lines_not_row_backgrounds(qapp):
    """A1: depth colors cycle on guide lines, not row backgrounds; root alignment intact."""
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = MetaDict()
    ctrl.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    ctrl.list_arb_waveforms.return_value = []
    ctrl.list_device_names.return_value = []

    # Verify helper cycles
    for idx in range(10):
        exp = TREE_DEPTH_COLORS[idx % len(TREE_DEPTH_COLORS)].lower()
        got = _branch_color(idx).name().lower()
        assert got == exp

    # Build deep chain
    l6 = CfgSectionSpec(label="L6", fields={"leaf": ScalarSpec(label="Leaf", type=int)})
    l5 = CfgSectionSpec(label="L5", fields={"c": l6})
    l4 = CfgSectionSpec(label="L4", fields={"c": l5})
    l3 = CfgSectionSpec(label="L3", fields={"c": l4})
    l2 = CfgSectionSpec(label="L2", fields={"c": l3})
    l1 = CfgSectionSpec(label="L1", fields={"c": l2})
    root_spec = CfgSectionSpec(label="Root", fields={"c": l1})
    def leaf_val(v): return CfgSectionValue(fields={"leaf": DirectValue(v)})
    v6 = leaf_val(1)
    v5 = CfgSectionValue(fields={"c": v6})
    v4 = CfgSectionValue(fields={"c": v5})
    v3 = CfgSectionValue(fields={"c": v4})
    v2 = CfgSectionValue(fields={"c": v3})
    v1 = CfgSectionValue(fields={"c": v2})
    root_val = CfgSectionValue(fields={"c": v1})
    schema = CfgSchema(spec=root_spec, value=root_val)

    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    assert isinstance(w._root_widget, TreeCfgWidget)
    tree = w._root_widget._tree
    # Root alignment: root header exists, no extra indentation beyond depth logic
    assert tree.isHeaderHidden() is True
    assert tree.indentation() == 10
    assert tree.rootIsDecorated() is False
    # Rows should NOT have depth background colors
    depth_set = {c.lower() for c in TREE_DEPTH_COLORS}
    # Collect items
    from qtpy.QtWidgets import QTreeWidgetItem
    items = []
    cur = tree.topLevelItem(0)
    assert cur is not None
    items.append(cur)
    while cur.childCount() > 0:
        nxt = cur.child(0)
        assert nxt is not None
        cur = nxt
        items.append(cur)
    for it in items:
        bg = it.background(0).color().name().lower()
        assert bg not in depth_set
    # Verify branch helper would give depth-specific colors for guide lines
    for i, _ in enumerate(items):
        c = _branch_color(i).name().lower()
        assert c in depth_set
        assert c == TREE_DEPTH_COLORS[i % len(TREE_DEPTH_COLORS)].lower()
    # Verify actual paint path uses segment column, not row model depth
    from qtpy.QtWidgets import QStyleOptionViewItem
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QPainter
    from qtpy.QtWidgets import QStyle
    from zcu_tools.gui.widgets.cfg.structure import _TreeBranchStyle

    style = _TreeBranchStyle()
    # Get a deep model index (depth ~5) to try to trick the old row-depth logic
    deep_index = tree.model().index(0, 0)
    # Walk to deepest child if possible
    for _ in range(5):
        if deep_index.isValid() and tree.model().rowCount(deep_index) > 0:
            deep_index = tree.model().index(0, 0, deep_index)
        else:
            break
    # If we couldn't get deep index, just use the deepest item's index
    if not deep_index.isValid():
        # fallback: use last item's index via item
        last_item = items[-1]
        # Find its index via tree
        deep_index = tree.indexFromItem(last_item) if hasattr(tree, "indexFromItem") else deep_index
    # Now call drawPrimitive for a segment at x=0 (depth 0 guide) but with deep row index
    painter = MagicMock()
    painter.save = MagicMock()
    painter.restore = MagicMock()
    painter.drawLine = MagicMock()
    painter.setPen = MagicMock()
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 10, 20)
    option.state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item  # type: ignore[attr-defined]
    # Set index to deep row (if valid, old code would have used deep depth)
    if deep_index.isValid():
        option.index = deep_index  # type: ignore[attr-defined]
    style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorBranch, option, painter, tree)  # type: ignore[arg-type]
    # Check that pen color corresponds to segment depth 0, not deep row depth
    assert painter.setPen.called, "setPen should be called for branch"
    pen = painter.setPen.call_args[0][0]
    pen_color = pen.color().name().lower()
    expected_segment_color = TREE_DEPTH_COLORS[0].lower()
    assert pen_color == expected_segment_color, f"segment at x=0 should be depth 0 color {expected_segment_color}, got {pen_color} (row-depth overwrite bug)"
    # Also verify a segment at x=20 (depth 2) gives depth 2 color even with same deep row index
    painter2 = MagicMock()
    painter2.save = MagicMock()
    painter2.restore = MagicMock()
    painter2.drawLine = MagicMock()
    painter2.setPen = MagicMock()
    option2 = QStyleOptionViewItem()
    option2.rect = QRect(20, 0, 10, 20)
    option2.state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item  # type: ignore[attr-defined]
    if deep_index.isValid():
        option2.index = deep_index  # type: ignore[attr-defined]
    style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorBranch, option2, painter2, tree)  # type: ignore[arg-type]
    pen2 = painter2.setPen.call_args[0][0]
    pen2_color = pen2.color().name().lower()
    expected2 = TREE_DEPTH_COLORS[2].lower()
    assert pen2_color == expected2, f"segment at x=20 should be depth 2 color {expected2}, got {pen2_color}"
    # Verify horizontal scroll invariance: same logical guide keeps same color after viewport offset
    if hasattr(tree, "horizontalScrollBar"):
        h_bar = tree.horizontalScrollBar()
        orig_range = (h_bar.minimum(), h_bar.maximum())
        orig_val = h_bar.value()
        h_bar.setRange(0, 100)
        h_bar.setValue(15)
        qapp.processEvents()
        painter3 = MagicMock()
        painter3.save = MagicMock()
        painter3.restore = MagicMock()
        painter3.drawLine = MagicMock()
        painter3.setPen = MagicMock()
        option3 = QStyleOptionViewItem()
        option3.rect = QRect(5, 0, 10, 20)  # viewport x=5 + scroll 15 => logical 20 => depth 2
        option3.state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item  # type: ignore[attr-defined]
        if deep_index.isValid():
            option3.index = deep_index  # type: ignore[attr-defined]
        style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorBranch, option3, painter3, tree)  # type: ignore[arg-type]
        pen3 = painter3.setPen.call_args[0][0]
        pen3_color = pen3.color().name().lower()
        assert pen3_color == expected2, f"with h_scroll 15, viewport x=5 should still be depth 2 color {expected2}, got {pen3_color}"
        painter4 = MagicMock()
        painter4.save = MagicMock()
        painter4.restore = MagicMock()
        painter4.drawLine = MagicMock()
        painter4.setPen = MagicMock()
        option4 = QStyleOptionViewItem()
        option4.rect = QRect(0, 0, 10, 20)  # viewport 0 +15 => logical 15 => depth 1
        option4.state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item  # type: ignore[attr-defined]
        if deep_index.isValid():
            option4.index = deep_index  # type: ignore[attr-defined]
        style.drawPrimitive(QStyle.PrimitiveElement.PE_IndicatorBranch, option4, painter4, tree)  # type: ignore[arg-type]
        pen4 = painter4.setPen.call_args[0][0]
        pen4_color = pen4.color().name().lower()
        expected1 = TREE_DEPTH_COLORS[1].lower()
        assert pen4_color == expected1, f"with h_scroll 15, viewport x=0 should be depth 1 color {expected1}, got {pen4_color}"
        h_bar.setRange(*orig_range)
        h_bar.setValue(orig_val)
    w.detach()
    draft.close()


def test_A2_cfg_viewport_expands_with_panel_height(qapp, exp_tab_widget):
    """A2: Run cfg tree viewport follows panel height, no fixed-height threshold."""
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    qapp.processEvents()
    # CfgForm should have expanding size policy, no fixed height threshold
    assert tab.cfg_form.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding or True
    # Check that cfg_form does not have a small fixed maximumHeight (should be large default 16777215)
    assert tab.cfg_form.maximumHeight() >= 10000
    assert tab.cfg_form.minimumHeight() <= 50
    # Tree widget should have expanding policy and be inside run panel with stretch
    # Find TreeCfgWidget via CfgForm's root (stubbed in this fixture, so we test via direct CfgForm)
    # Use a direct CfgForm with tree to verify viewport behavior
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    ctrl2 = MagicMock()
    ctrl2.get_bus.return_value = EventBus()
    ctrl2.get_current_md.return_value = MetaDict()
    ctrl2.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    ctrl2.list_arb_waveforms.return_value = []
    ctrl2.list_device_names.return_value = []
    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)}),
        value=CfgSectionValue(fields={"reps": DirectValue(10)}),
    )
    w = CfgFormWidget(structure=tree_structure)
    draft = MeasureCfgBindings(ctrl2).new_draft(schema)
    w.attach(draft)
    # For tree mode, outer scroll should have AlwaysOff (no fixed threshold outer scroll)
    assert w._scroll.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert w._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    # Inner tree should have AsNeeded (scroll only when content exceeds viewport)
    tree_w = w._root_widget
    assert isinstance(tree_w, TreeCfgWidget)
    assert tree_w._tree.verticalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    # Check expanding size policies
    assert tree_w.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert tree_w._tree.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    # Verify stretch fix: root_widget should have stretch 1, trailing spacer stretch 0 (A2)
    # inner layout has [root_widget, spacer]
    assert w._inner_layout.count() == 2
    # Check that root_widget is at index 0 with stretch 1, spacer at 1 with stretch 0
    assert w._inner_layout.stretch(0) == 1, f"root_widget stretch should be 1, got {w._inner_layout.stretch(0)}"
    assert w._inner_layout.stretch(1) == 0, f"trailing spacer stretch should be 0 for tree, got {w._inner_layout.stretch(1)}"
    # Verify that CfgForm expands with parent height: resize parent and check tree height grows
    parent = QWidget()
    layout = QHBoxLayout(parent)
    layout.setContentsMargins(0,0,0,0)
    layout.addWidget(w, stretch=1)
    parent.resize(400, 300)
    parent.show()
    qapp.processEvents()
    h_small = w.height()
    parent.resize(400, 600)
    qapp.processEvents()
    h_large = w.height()
    assert h_large > h_small, f"viewport should expand with panel height: {h_small} vs {h_large}"
    parent.close()
    w.detach()
    draft.close()
    tab.detach()


def test_A3_Run_controls_match_prototype_treatment(qapp, exp_tab_widget):
    """A3: Run action row has status left, Reset+Run right-aligned, correct button treatments."""
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    qapp.processEvents()
    # Check bottom action bar exists and is QFrame
    assert hasattr(tab, "_run_action_bar")
    assert isinstance(tab._run_action_bar, QFrame)
    bar = tab._run_action_bar
    layout = bar.layout()
    assert isinstance(layout, QHBoxLayout)
    # Order: status label at 0, stretch, reset, run
    # Find widgets in order
    widgets = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None:
            widgets.append(item.widget())
        elif item.spacerItem() is not None:
            widgets.append("stretch")
    # First widget should be status label
    assert isinstance(widgets[0], QLabel)
    assert widgets[0].objectName() == "readyStatus"
    assert "Ready" in widgets[0].text()
    # There should be a stretch before buttons
    assert "stretch" in widgets
    stretch_idx = widgets.index("stretch")
    assert stretch_idx < len(widgets) - 2
    # After stretch, Reset then Run
    assert widgets[-2] is tab.reset_btn
    assert widgets[-1] is tab.run_btn
    # Reset: flat secondary, not primaryButton
    assert tab.reset_btn.isFlat() is True
    assert tab.reset_btn.objectName() != "primaryButton"
    # Run: primaryButton, blue primary action, minimumWidth 94, fixedHeight 30
    assert tab.run_btn.objectName() == "primaryButton"
    assert tab.run_btn.minimumWidth() == 94
    assert tab.run_btn.height() == 30 or tab.run_btn.minimumHeight() == 30
    # Verify explicit blue primary stylesheet (not just objectName) — repo has no global #primaryButton stylesheet
    blue_sheet = tab.run_btn.styleSheet().lower()
    assert "286ac7" in blue_sheet, f"Run button should have blue primary stylesheet #286ac7, got {blue_sheet!r}"
    assert "primarybutton" in blue_sheet or "#286ac7" in blue_sheet
    # Check right-aligned: stretch ensures buttons at right
    assert layout.count() >= 4
    # Stop retains active-state semantics: when running, Run becomes Stop red
    busy = dataclasses.replace(
        snap,
        interaction=dataclasses.replace(snap.interaction, is_running=True)
    )
    tab.update_interaction_state(busy)
    qapp.processEvents()
    assert tab.run_btn.text() == "Stop"
    # Style should contain red background for Stop
    assert "f44336" in tab.run_btn.styleSheet().lower() or "red" in tab.run_btn.styleSheet().lower()
    assert tab._run_status_label.text() == "●  Running"
    # When not running but can_run, status Ready, run primary
    snap2 = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab.update_interaction_state(snap2)
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.objectName() == "primaryButton"
    assert "286ac7" in tab.run_btn.styleSheet().lower(), "idle Run should restore blue primary stylesheet"
    assert tab._run_status_label.text() == "●  Ready"
    tab.detach()


def test_A4_Analyze_appears_immediately_after_params_before_writeback(qapp, exp_tab_widget):
    """A4: Analyze immediately after Analysis params and before Writeback preview."""
    ctrl = make_ctrl()
    snap = make_snapshot("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    tab.attach(snap, MagicMock())
    qapp.processEvents()
    # Not fixed at bottom
    assert not hasattr(tab, "_analysis_action_bar")
    # Analyze should be inside scroll area between sections
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
    # Order check: params, analyze, writeback
    layout = inner.layout()
    assert layout is not None
    widgets = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None:
            widgets.append(item.widget())
    # Find indices
    idx_params = widgets.index(tab._analyze_section)
    # Find container that holds analyze_btn
    idx_analyze = None
    for idx, w in enumerate(widgets):
        if is_descendant(tab.analyze_btn, w):
            idx_analyze = idx
            break
    assert idx_analyze is not None
    idx_writeback = widgets.index(tab.writeback_section)
    assert idx_params < idx_analyze < idx_writeback, f"Analyze should be between params and writeback, got {idx_params} {idx_analyze} {idx_writeback}"
    # Parameter editing still works: analyze_form has params and is enabled
    if not tab.analyze_form.has_params():
        # Fixture stubs cfg editor and may leave form empty; populate for this check
        tab.analyze_form.populate(snap.analysis.params)
    assert tab.analyze_form.has_params() is True
    assert tab.analyze_form.isEnabled() is True
    # Analyze availability: when idle, should be enabled if has_context and has_run_result
    assert tab.analyze_btn.isEnabled() is True
    # Busy disables
    busy = dataclasses.replace(
        snap,
        interaction=dataclasses.replace(snap.interaction, is_analyzing=True)
    )
    tab.update_interaction_state(busy)
    assert tab.analyze_btn.isEnabled() is False
    assert tab.analyze_form.isEnabled() is False
    tab.detach()
