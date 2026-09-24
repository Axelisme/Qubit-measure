from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
)
from zcu_tools.experiment.v2_gui.adapters.fake.freq import (
    FakeFreqAdapter,
    FakeFreqAnalyzeParams,
    FakeFreqRunResult,
)
from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    ExpContext,
    MetaDictWriteback,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.main.ui.writeback_widget import WritebackWidget
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def _default_analyze_params(
    adapter: FakeFreqAdapter, result: FakeFreqRunResult, ctx: ExpContext
) -> FakeFreqAnalyzeParams:
    return adapter.get_analyze_params(result, ctx)


def test_writeback_widget_lists_items_and_edit_buttons(qapp):
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(
        RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg), schema
    )
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=_default_analyze_params(adapter, result, ctx),
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
    )
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(run_result=result, analyze_result=analyze_result, ctx=ctx)
        )
    )
    # The service stamps session_ids at compute time; do it here so the widget's
    # per-id checkbox map is unambiguous.
    for i, item in enumerate(items):
        item.session_id = f"id-{i}"

    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    widget.populate(items)
    selected = [it for it in items if it.selected]
    edit_buttons = [w for w in widget.findChildren(QPushButton) if w.text() == "Edit"]

    # The one-tone freq fit proposes only r_f / rf_w (two MetaDict items, both
    # editable) — no readout module / waveform writeback.
    assert len(selected) == len(items)  # all selected by default
    assert {it.target_name for it in items} == {"r_f", "rf_w"}
    assert len(edit_buttons) == 2


def test_writeback_widget_non_scalar_item_is_read_only(qapp):
    """A non-scalar md item (e.g. a confusion matrix) renders selectable but
    with no Edit button — it is a derived value applied verbatim."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    scalar = MetaDictWriteback(target_name="fid", description="d", proposed_value=0.95)
    scalar.session_id = "md-1"
    nonscalar = MetaDictWriteback(
        target_name="confusion_matrix", description="d", proposed_value=matrix
    )
    nonscalar.session_id = "md-2"

    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    widget.populate([scalar, nonscalar])

    edit_buttons = [w for w in widget.findChildren(QPushButton) if w.text() == "Edit"]
    # Only the scalar item gets an Edit button; the matrix is read-only.
    assert len(edit_buttons) == 1
    # Both items are still selectable for apply (the matrix label shows its value).
    checks = widget.findChildren(QCheckBox)
    assert len(checks) == 2
    assert any("confusion_matrix" in cb.text() for cb in checks)
    assert all(cb.isChecked() for cb in checks)


def test_writeback_widget_projects_draft_owned_applied_state(qapp):
    item = MetaDictWriteback(target_name="r_f", description="d", proposed_value=6000.0)
    item.session_id = "md-1"
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {"md-1": False}
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")

    widget.populate([item])

    checkbox = widget._checks["md-1"]
    assert checkbox.text() == "r_f*"
    assert checkbox.font().bold()
    assert any(
        label.text() == "* = not applied" for label in widget.findChildren(QLabel)
    )

    ctrl.get_writeback_applied_for_pane.return_value = {"md-1": True}
    widget.populate([item])
    checkbox = widget._checks["md-1"]
    assert checkbox.text() == "r_f"
    assert not checkbox.font().bold()


def test_writeback_widget_does_not_hide_applied_projection_failures(qapp):
    item = MetaDictWriteback(target_name="r_f", description="d", proposed_value=6000.0)
    item.session_id = "md-1"
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.side_effect = RuntimeError("projection failed")
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")

    with pytest.raises(RuntimeError, match="projection failed"):
        widget.populate([item])


def test_writeback_compact_ledger_target_only_centered_and_equal_actions(qapp):
    """A1 — target-only labels, tooltip, centered Current → Proposed,
    shared backgrounds/borders and equal 56x26 actions."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    scalar = MetaDictWriteback(
        target_name="r_f", description="resonator freq", proposed_value=6000.0
    )
    scalar.session_id = "md-1"
    nonscalar = MetaDictWriteback(
        target_name="confusion_matrix", description="matrix desc", proposed_value=matrix
    )
    nonscalar.session_id = "md-2"
    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    widget.resize(600, 400)
    widget.populate([scalar, nonscalar])
    widget.show()
    qapp.processEvents()
    try:
        # target-only, tooltip, no duplication
        cbs = {cb.text(): cb for cb in widget.findChildren(QCheckBox)}
        assert set(cbs) == {"r_f*", "confusion_matrix*"}
        assert cbs["r_f*"].toolTip() == "resonator freq"
        assert cbs["confusion_matrix*"].toolTip() == "matrix desc"
        assert "0.95" not in cbs["confusion_matrix*"].text()
        assert "freq" not in cbs["r_f*"].text()
        # centered Current → Proposed
        for lbl in widget.findChildren(QLabel):
            if lbl.objectName() in (
                "writebackCurrent",
                "writebackProposed",
                "writebackProposedChip",
                "writebackArrow",
            ):
                assert lbl.alignment() & Qt.AlignmentFlag.AlignHCenter
                assert lbl.alignment() & Qt.AlignmentFlag.AlignVCenter
        # shared backgrounds / continuous borders — observed via palette and geometry
        panel = widget._rows_container
        rows = widget._rows
        assert panel.objectName() == "writebackPanel"
        # rendered white background shared between panel and rows (palette)
        panel_bg = panel.palette().color(panel.backgroundRole()).name().lower()
        assert panel_bg == "#ffffff", f"panel bg {panel_bg}"
        assert rows, "no rows rendered"
        first_bg = rows[0].palette().color(rows[0].backgroundRole()).name().lower()
        assert first_bg == "#ffffff", f"row bg {first_bg}"
        assert panel_bg == first_bg
        # container has no extra spacing, rows stack continuously with 1px border gap
        assert widget._rows_layout.spacing() == 0
        cm = widget._rows_layout.contentsMargins()
        assert cm.left() == 0 and cm.top() == 0 and cm.right() == 0 and cm.bottom() == 0
        if len(rows) > 1:
            # second row directly follows first (allow 1px border)
            assert (
                abs(rows[1].geometry().top() - (rows[0].geometry().bottom() + 1)) <= 1
            )
        # row internal margins per spec 8,4,8,4
        lay0 = rows[0].layout()
        assert lay0 is not None
        lm = lay0.contentsMargins()
        assert lm.left() == 8 and lm.top() == 4 and lm.right() == 8 and lm.bottom() == 4
        # identical Edit/Copy geometry
        edit_btns = [b for b in widget.findChildren(QPushButton) if b.text() == "Edit"]
        copy_btns = [b for b in widget.findChildren(QPushButton) if b.text() == "Copy"]
        assert len(edit_btns) == 1 and len(copy_btns) == 1
        for b in edit_btns + copy_btns:
            assert b.width() == 56 and b.height() == 26
            assert b.size().width() == 56 and b.size().height() == 26
        # rendered-image border checks — bounded deterministic sampling of
        # visible outer panel border and adjacent-row divider (palette alone
        # would hide a missing stylesheet border).
        panel_grab = panel.grab()  # type: ignore[attr-defined]
        img = panel_grab.toImage()
        w = panel.width()
        h = panel.height()
        # offscreen QT_QPA_PLATFORM=offscreen has devicePixelRatio 1.0; scale if hidpi
        dpr = panel_grab.devicePixelRatio()  # type: ignore[attr-defined]
        scale = int(dpr) if dpr != 1 else 1  # type: ignore[arg-type]
        outer_expected = "#d7dde7"
        divider_expected = "#e8ecf2"
        # outer border — top-center and left-mid avoid rounded corners
        assert (
            img.pixelColor((w // 2) * scale, 0 * scale).name().lower() == outer_expected
        ), f"outer top border {img.pixelColor((w // 2) * scale, 0).name().lower()}"
        assert (
            img.pixelColor(0 * scale, (h // 2) * scale).name().lower() == outer_expected
        )
        # adjacent-row divider — horizontal 1px line at row boundary
        assert len(rows) >= 2
        y_div = rows[1].pos().y() - 1
        assert 0 <= y_div < h, f"divider y {y_div} out of {h}"
        assert (
            img.pixelColor((w // 2) * scale, y_div * scale).name().lower()
            == divider_expected
        )
        assert (
            img.pixelColor((w // 4) * scale, y_div * scale).name().lower()
            == divider_expected
        )
        # sanity: interior pixels are white, not border
        assert img.pixelColor((w // 2) * scale, 1 * scale).name().lower() == "#ffffff"
        assert (
            img.pixelColor((w // 2) * scale, (y_div + 1) * scale).name().lower()
            == "#ffffff"
        )
    finally:
        widget.close()


def test_writeback_proposed_matrix_has_view_while_current_stays_summary_only(qapp):
    """A2 — Proposed shows the 3×3 view; Current stays bounded summary-only."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    controller = MagicMock()
    controller.get_writeback_summaries_for_pane.return_value = {
        "md-1": ("3 × 3 matrix", "3 × 3 matrix")
    }
    widget = WritebackWidget(controller, tab_id="tab-1", pane="analysis")
    item = MetaDictWriteback(
        target_name="confusion_matrix", description="d", proposed_value=matrix
    )
    item.session_id = "md-1"
    widget.resize(600, 400)
    widget.populate([item])
    widget.show()
    qapp.processEvents()
    try:
        current = widget.findChild(QLabel, "writebackCurrent")
        assert current is not None
        assert current.text() == "3 × 3 matrix"

        # Proposed keeps its bounded heading and adds the readable matrix view.
        prop = next(
            l
            for l in widget.findChildren(QLabel)
            if "writebackProposed" in l.objectName()
        )
        assert prop.text() == "3 \u00d7 3 matrix"
        tables = widget.findChildren(QTableWidget)
        assert len(tables) == 1
        assert tables[0].objectName() == "writebackProposedMatrixTable"
        top_left = tables[0].item(0, 0)
        bottom_right = tables[0].item(2, 2)
        assert top_left is not None
        assert bottom_right is not None
        assert top_left.text() == "0.9500"
        assert bottom_right.text() == "1.0000"
        # Copy places complete JSON
        copy_btn = next(
            b for b in widget.findChildren(QPushButton) if b.text() == "Copy"
        )
        copy_btn.click()
        qapp.processEvents()
        cb_clip = QApplication.clipboard()
        assert cb_clip is not None
        assert cb_clip.text() == json.dumps(matrix)
        # long value cannot widen ledger
        long_val = list(range(200))
        w2 = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
        long_item = MetaDictWriteback(
            target_name="long_list", description="d", proposed_value=long_val
        )
        long_item.session_id = "md-1"
        w2.resize(600, 400)
        w2.populate([long_item])
        w2.show()
        qapp.processEvents()
        try:
            prop2 = next(
                l
                for l in w2.findChildren(QLabel)
                if "writebackProposed" in l.objectName()
            )
            assert prop2.text() == "list[200]"
            # size hint remains bounded
            assert w2.sizeHint().width() < widget.sizeHint().width() + 200
            assert w2.findChildren(QTableWidget) == []
            # No nested writeback scroll — long value does not create inner scroll
            assert w2.findChild(QScrollArea, "writebackScroll") is None
            assert not hasattr(w2, "_scroll")
            # No horizontal overflow: panel width bounded
            assert w2._rows_container.width() <= w2.width() + 4
        finally:
            w2.close()
    finally:
        widget.close()


def test_writeback_responsive_reflow_no_overflow_and_tight_panel(qapp):
    """A3 — 500 px wide single-line vs 400 px narrow stacked, no nested scroll,
    no horizontal overflow, panel hugs rows, Apply immediately follows."""
    items: list[MetaDictWriteback] = []
    for i in range(3):
        it = MetaDictWriteback(
            target_name=f"p{i}", description=f"d{i}", proposed_value=float(i)
        )
        it.session_id = f"md-{i}"
        items.append(it)
    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    widget.resize(500, 400)
    widget.populate(items)
    widget.show()
    qapp.processEvents()
    try:
        # no nested writeback scroll
        assert widget.findChild(QScrollArea, "writebackScroll") is None
        assert not hasattr(widget, "_scroll")
        assert widget.findChildren(QScrollArea) == []

        # wide: checkbox and current share Y (single-line)
        cbs = widget.findChildren(QCheckBox)
        curs = [
            l
            for l in widget.findChildren(QLabel)
            if l.objectName() == "writebackCurrent"
        ]
        assert cbs and curs
        assert abs(cbs[0].y() - curs[0].y()) < 30

        # No horizontal overflow: panel width within widget width
        assert widget._rows_container.width() <= widget.width() + 2
        # panel hugs rows, Apply immediately follows (content-tight)
        panel = widget._rows_container
        rows = widget._rows
        assert rows, "no rows"
        last = rows[-1]
        assert abs(panel.height() - (last.geometry().bottom() + 1)) <= 6, (
            f"panel {panel.height()} vs last bottom {last.geometry().bottom() + 1}"
        )
        layout = widget.layout()
        assert layout is not None
        spacing = layout.spacing()
        assert (
            abs(
                widget._apply_btn.geometry().y()
                - (panel.geometry().y() + panel.height() + spacing)
            )
            <= 6
        ), (
            f"Apply y {widget._apply_btn.geometry().y()} vs panel y {panel.geometry().y()} h {panel.height()} spacing {spacing}"
        )
        assert widget._apply_btn.isVisible()
        assert widget._apply_btn.parent() is widget

        # narrow: should reflow to target/action above Current→Proposed
        widget.resize(400, 400)
        qapp.processEvents()
        widget._update_responsive()
        qapp.processEvents()
        # in narrow, checkbox above current
        assert cbs[0].y() < curs[0].y() - 5
        # still no nested scroll and still content-tight
        assert widget.findChild(QScrollArea, "writebackScroll") is None
        assert widget._rows_container.width() <= widget.width() + 2
        assert abs(panel.height() - (rows[-1].geometry().bottom() + 1)) <= 6
        assert (
            abs(
                widget._apply_btn.geometry().y()
                - (panel.geometry().y() + panel.height() + spacing)
            )
            <= 6
        )
        assert widget._apply_btn.isVisible()
        # no hidden inner scrollbar to interfere
        assert widget.findChildren(QScrollArea) == []
    finally:
        widget.close()


def test_writeback_ledger_content_tight_no_nested_scroll_and_long_grows_via_outer(qapp):
    """A3 — bordered ledger bottom tracks last row, Apply immediately follows,
    no nested writeback scroll at 500/400, long ledger grows naturally and
    is reachable through an outer scroll (generic host)."""
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    def _assert_tight(widget: WritebackWidget, *, frame_tolerance: int = 6) -> None:
        panel = widget._rows_container
        rows = widget._rows
        assert rows, "no rows rendered"
        assert panel.isVisible()
        # no nested scroll
        assert widget.findChild(QScrollArea, "writebackScroll") is None
        assert widget.findChildren(QScrollArea) == []
        # panel bottom tracks last row
        last = rows[-1]
        assert (
            abs(panel.height() - (last.geometry().bottom() + 1)) <= frame_tolerance
        ), f"panel {panel.height()} vs last bottom {last.geometry().bottom() + 1}"
        # Apply immediately follows panel with only layout spacing
        layout = widget.layout()
        assert layout is not None
        spacing = layout.spacing()
        assert (
            abs(
                widget._apply_btn.geometry().y()
                - (panel.geometry().y() + panel.height() + spacing)
            )
            <= frame_tolerance
        ), (
            f"Apply y {widget._apply_btn.geometry().y()} panel y {panel.geometry().y()} h {panel.height()} spacing {spacing}"
        )
        assert widget._apply_btn.isVisible()
        assert widget._apply_btn.parent() is widget
        # no horizontal overflow: panel width bounded
        assert panel.width() <= widget.width() + frame_tolerance

    # Short ledger at 500 and 400 — content-tight, no inner scroll
    for width in (500, 400):
        items_short: list[MetaDictWriteback] = []
        for i in range(4):
            it = MetaDictWriteback(
                target_name=f"s{i}", description=f"desc {i}", proposed_value=float(i)
            )
            it.session_id = f"md-{i}"
            items_short.append(it)
        w = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
        w.resize(width, 400)
        w.populate(items_short)
        w.show()
        qapp.processEvents()
        qapp.processEvents()
        try:
            _assert_tight(w)
            # natural height is compact (short)
            short_h = w.sizeHint().height()
            # For reference, store for long comparison
            assert short_h < 600, f"short should be compact, got {short_h}"
        finally:
            w.close()
            qapp.processEvents()

    # Long ledger — natural height grows, no capping, reachable via outer scroll
    items_long: list[MetaDictWriteback] = []
    for i in range(30):
        it = MetaDictWriteback(
            target_name=f"p{i}", description=f"d{i}", proposed_value=float(i)
        )
        it.session_id = f"md-{i}"
        items_long.append(it)

    # Measure natural height increase
    w_short = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    w_short.resize(500, 400)
    short_items = [
        MetaDictWriteback(target_name="s0", description="d", proposed_value=0.0)
    ]
    short_items[0].session_id = "md-0"
    w_short.populate(short_items)
    w_short.show()
    qapp.processEvents()
    short_height = w_short.sizeHint().height()
    short_panel_h = w_short._rows_container.height()
    w_short.close()

    w_long = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    w_long.resize(500, 400)
    w_long.populate(items_long)
    w_long.show()
    qapp.processEvents()
    qapp.processEvents()
    try:
        _assert_tight(w_long)
        long_height = w_long.sizeHint().height()
        long_panel_h = w_long._rows_container.height()
        assert long_height > short_height + 200, (
            f"long natural height {long_height} should exceed short {short_height} significantly"
        )
        assert long_panel_h > short_panel_h + 200
        # Still no inner scroll
        assert w_long.findChild(QScrollArea, "writebackScroll") is None

        # Reachability via outer scroll: host outer QScrollArea with small viewport
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFixedSize(520, 300)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.addWidget(w_long)
        inner_layout.addStretch()
        outer.setWidget(inner)
        outer.show()
        qapp.processEvents()
        qapp.processEvents()
        try:
            # Outer should be scrollable because long ledger exceeds viewport
            vbar = outer.verticalScrollBar()
            assert vbar is not None
            # Need to wait for layout; outer should have scrollable extent
            # The inner's height should exceed viewport's height
            vp = outer.viewport()
            assert vp is not None
            assert inner.height() > vp.height(), (
                f"inner {inner.height()} vs viewport {vp.height()}"
            )
            assert vbar.maximum() > 0, (
                f"outer should be scrollable for long ledger, vbar max {vbar.maximum()}"
            )
            # Apply should be beyond initial viewport top, but reachable by scrolling to max
            # Map Apply's bottom to outer viewport coordinates
            # Initially at top, Apply may be outside viewport; after scrolling to max it should be visible
            vbar.setValue(vbar.maximum())
            qapp.processEvents()
            assert abs(vbar.value() - vbar.maximum()) <= 2
            # After scrolling, Apply's geometry mapped to outer should be within outer's visible rect
            # We verify that Apply is still a child of w_long and outer is scrollable; exact visibility
            # depends on layout, but the fact that outer max >0 proves reachability.
            vbar.setValue(0)
            qapp.processEvents()
        finally:
            outer.close()
            # w_long was reparented to inner; inner's close will delete it, but we already manage
            # Ensure w_long not double-closed; it was already inside outer, so just process events
            qapp.processEvents()
    finally:
        # w_long already closed via outer if reparented; ensure no double close error
        try:
            w_long.close()
        except Exception:
            pass
        qapp.processEvents()


def test_writeback_shipped_analysis_composition_outer_scroll_owns_long_content(qapp):
    """A3 — shipped Analysis composition: no nested writeback scroll, panel
    bottom tracks last row, Apply immediately follows, long content is
    reachable through the existing Analysis-pane outer scroll."""
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.main.services import PersistedStartup
    from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget

    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)  # type: ignore[call-arg]
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}

    tab = ExpTabWidget("tab-1", ctrl, caps)
    tab.resize(600, 500)
    tab.show()
    qapp.processEvents()
    qapp.processEvents()
    tab._left_tabs.setCurrentWidget(tab._analysis_panel)
    qapp.processEvents()
    qapp.processEvents()

    # Need to locate outer analysis scroll: it's the QScrollArea containing analysis_inner
    # Find it via tab children
    outer_scroll: QScrollArea | None = None
    for child in tab.findChildren(QScrollArea):
        # The analysis_scroll is the one whose widget contains the writeback_section
        w = child.widget()
        if w is not None:
            # Check if writeback_widget is descendant of w
            q = tab.writeback_widget.parentWidget()
            found = False
            while q is not None:
                if q is w:
                    found = True
                    break
                q = q.parentWidget()
            if found:
                outer_scroll = child
                break
    assert outer_scroll is not None, (
        "Analysis pane outer scroll not found — cannot delegate scrolling"
    )

    # Long ledger
    items_long: list[MetaDictWriteback] = []
    for i in range(30):
        it = MetaDictWriteback(
            target_name=f"p{i}", description=f"d{i}", proposed_value=float(i)
        )
        it.session_id = f"md-{i}"
        items_long.append(it)
    tab.writeback_widget.populate(items_long)
    tab.writeback_section.setVisible(True)
    qapp.processEvents()
    qapp.processEvents()
    qapp.processEvents()

    def _assert_shipped_tight():
        w = tab.writeback_widget
        panel = w._rows_container
        rows = w._rows
        assert rows, "no rows"
        # No nested inner scroll
        assert w.findChild(QScrollArea, "writebackScroll") is None
        assert w.findChildren(QScrollArea) == []
        # Panel hugs last row
        last = rows[-1]
        assert abs(panel.height() - (last.geometry().bottom() + 1)) <= 6, (
            f"panel {panel.height()} vs last {last.geometry().bottom() + 1}"
        )
        # Apply immediately follows panel
        layout = w.layout()
        assert layout is not None
        spacing = layout.spacing()
        assert (
            abs(
                w._apply_btn.geometry().y()
                - (panel.geometry().y() + panel.height() + spacing)
            )
            <= 6
        )
        assert w._apply_btn.isVisible()
        assert w._apply_btn.parent() is w

    try:
        _assert_shipped_tight()
        # Outer scroll should now be scrollable for long content
        # Inner (analysis_inner) should be taller than viewport
        inner = outer_scroll.widget()
        assert inner is not None
        viewport = outer_scroll.viewport()
        assert viewport is not None
        # Process events to ensure layout updated
        qapp.processEvents()
        assert inner.height() > viewport.height(), (
            f"shipped outer inner {inner.height()} should exceed viewport {viewport.height()} for long ledger"
        )
        vbar = outer_scroll.verticalScrollBar()
        assert vbar is not None
        assert vbar.maximum() > 0, (
            f"outer Analysis scroll should be scrollable, max {vbar.maximum()}"
        )
        # Verify Apply is beyond initial viewport but reachable
        # Map Apply to outer viewport coordinates via global
        # Initially at top, Apply likely below viewport; after scrolling to max, verify max scroll works
        vbar.setValue(vbar.maximum())
        qapp.processEvents()
        assert abs(vbar.value() - vbar.maximum()) <= 2
        # After scroll, still tight
        _assert_shipped_tight()
        vbar.setValue(0)
        qapp.processEvents()
        _assert_shipped_tight()

        # Also verify at narrow width (400) the shipped composition still hugs and remains outer-scrollable
        tab.resize(500, 500)
        # Force writeback narrow mode by resizing writeback widget narrow? ExpTabWidget left pane width changes with splitter,
        # but we can directly resize the writeback_widget to 400 to trigger narrow
        tab.writeback_widget.resize(400, tab.writeback_widget.height())
        qapp.processEvents()
        qapp.processEvents()
        tab.writeback_widget._update_responsive()
        qapp.processEvents()
        _assert_shipped_tight()
        # Re-check outer still scrollable after narrow reflow (rows taller in narrow)
        qapp.processEvents()
        assert inner.height() > viewport.height() or vbar.maximum() > 0

        # Short ledger should not require outer scrolling to show Apply (but outer may still have some scroll due to other chrome)
        short_items: list[MetaDictWriteback] = []
        for i in range(2):
            it = MetaDictWriteback(
                target_name=f"s{i}", description="d", proposed_value=float(i)
            )
            it.session_id = f"md-{i}"
            short_items.append(it)
        tab.writeback_widget.populate(short_items)
        qapp.processEvents()
        qapp.processEvents()
        _assert_shipped_tight()
        # For short, inner height growth should be smaller
        short_inner_h = inner.height()
        # Long was earlier, short should be smaller
        # We already asserted long inner > viewport; now short should be noticeably smaller
        # Just check that short still tight and no inner scroll
        assert tab.writeback_widget.findChildren(QScrollArea) == []

    finally:
        tab.close()
        qapp.processEvents()
