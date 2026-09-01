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
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.widgets.cfg import CfgFormWidget, TreeCfgWidget
from zcu_tools.gui.widgets.cfg.structure import TREE_DEPTH_COLORS, _branch_color
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # perceived luminance
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# Candidate 466122625829ff779e721127ed15bd08fe69f58c pastel palette (light)
_CANDIDATE_PASTELS = ("#e2ebf6", "#e3f0e6", "#f4e9d2", "#eadff1", "#dceeee")


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
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.services.ports import (
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
        paths=TabPathsSnapshot(
            data=data_path, analysis_image=image, post_analysis_image=image
        ),
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


def test_A1_depth_colors_dark_stable_and_not_row_background(qapp):
    """A1: depth colors cycle on guide-line segments, darker than candidate, stable under h-scroll, not row backgrounds."""
    # Verify palette is darker than candidate 466122 pastel set
    for new in TREE_DEPTH_COLORS:
        new_lum = _luminance(new)
        # Every new color must be visibly darker than the lightest candidate (≈ 235 avg)
        # Candidate luminance range ~ 230-235; new should be < 180 to be clearly darker
        assert new_lum < 180, (
            f"{new} luminance {new_lum:.1f} not < 180 (darker than candidate)"
        )
        # Also ensure darker than each candidate individually
        for pastel in _CANDIDATE_PASTELS:
            assert new_lum < _luminance(pastel) - 40, (
                f"{new} not sufficiently darker than candidate {pastel}"
            )

    # Verify helper cycles correctly
    for idx in range(10):
        exp = TREE_DEPTH_COLORS[idx % len(TREE_DEPTH_COLORS)].lower()
        got = _branch_color(idx).name().lower()
        assert got == exp

    # Build deep chain to check painting
    l6 = CfgSectionSpec(label="L6", fields={"leaf": ScalarSpec(label="Leaf", type=int)})
    l5 = CfgSectionSpec(label="L5", fields={"c": l6})
    l4 = CfgSectionSpec(label="L4", fields={"c": l5})
    l3 = CfgSectionSpec(label="L3", fields={"c": l4})
    l2 = CfgSectionSpec(label="L2", fields={"c": l3})
    l1 = CfgSectionSpec(label="L1", fields={"c": l2})
    root_spec = CfgSectionSpec(label="Root", fields={"c": l1})

    def leaf_val(v):
        return CfgSectionValue(fields={"leaf": DirectValue(v)})

    v6 = leaf_val(1)
    v5 = CfgSectionValue(fields={"c": v6})
    v4 = CfgSectionValue(fields={"c": v5})
    v3 = CfgSectionValue(fields={"c": v4})
    v2 = CfgSectionValue(fields={"c": v3})
    v1 = CfgSectionValue(fields={"c": v2})
    root_val = CfgSectionValue(fields={"c": v1})
    schema = CfgSchema(spec=root_spec, value=root_val)

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = MetaDict()
    ctrl.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    ctrl.list_arb_waveforms.return_value = []
    ctrl.list_device_names.return_value = []

    w = CfgFormWidget()
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    qapp.processEvents()
    assert isinstance(w._root_widget, TreeCfgWidget)
    tree = w._root_widget._tree
    assert tree.isHeaderHidden() is True
    assert tree.indentation() == 10
    assert tree.rootIsDecorated() is False

    # Rows should NOT have depth background colors
    depth_set = {c.lower() for c in TREE_DEPTH_COLORS}
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

    # Guide-line painting: depth is per-segment column, normalized for h-scroll
    from qtpy.QtCore import QRect
    from qtpy.QtWidgets import QStyle, QStyleOptionViewItem
    from zcu_tools.gui.widgets.cfg.structure import _TreeBranchStyle

    style = _TreeBranchStyle()
    deep_index = tree.model().index(0, 0)
    for _ in range(5):
        if deep_index.isValid() and tree.model().rowCount(deep_index) > 0:
            deep_index = tree.model().index(0, 0, deep_index)
        else:
            break
    if not deep_index.isValid():
        last_item = items[-1]
        deep_index = (
            tree.indexFromItem(last_item)
            if hasattr(tree, "indexFromItem")
            else deep_index
        )

    # segment at viewport x=0 => depth 0 even with deep row
    painter = MagicMock()
    painter.save = MagicMock()
    painter.restore = MagicMock()
    painter.drawLine = MagicMock()
    painter.setPen = MagicMock()
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 10, 20)
    option.state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item  # type: ignore[attr-defined]
    if deep_index.isValid():
        option.index = deep_index  # type: ignore[attr-defined]
    style.drawPrimitive(
        QStyle.PrimitiveElement.PE_IndicatorBranch, option, painter, tree
    )  # type: ignore[arg-type]
    assert painter.setPen.called
    pen = painter.setPen.call_args[0][0]
    pen_color = pen.color().name().lower()
    expected_segment_color = TREE_DEPTH_COLORS[0].lower()
    assert pen_color == expected_segment_color

    # x=20 => depth 2
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
    style.drawPrimitive(
        QStyle.PrimitiveElement.PE_IndicatorBranch, option2, painter2, tree
    )  # type: ignore[arg-type]
    pen2 = painter2.setPen.call_args[0][0]
    assert pen2.color().name().lower() == TREE_DEPTH_COLORS[2].lower()

    # stable under horizontal scroll
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
        option3.rect = QRect(5, 0, 10, 20)  # viewport 5 + 15 => logical 20 => depth 2
        option3.state = QStyle.StateFlag.State_Sibling | QStyle.StateFlag.State_Item  # type: ignore[attr-defined]
        if deep_index.isValid():
            option3.index = deep_index  # type: ignore[attr-defined]
        style.drawPrimitive(
            QStyle.PrimitiveElement.PE_IndicatorBranch, option3, painter3, tree
        )  # type: ignore[arg-type]
        pen3 = painter3.setPen.call_args[0][0]
        assert pen3.color().name().lower() == TREE_DEPTH_COLORS[2].lower()
        h_bar.setRange(*orig_range)
        h_bar.setValue(orig_val)
    w.detach()
    draft.close()


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
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = MetaDict()
    ctrl.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    ctrl.list_arb_waveforms.return_value = []
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
    ctrl2.list_arb_waveforms.return_value = []
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
    # Order: params < analyze < writeback
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
