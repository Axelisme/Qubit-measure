from __future__ import annotations

import json
from unittest.mock import MagicMock

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QPushButton, QTableWidget
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
        assert set(cbs) == {"r_f", "confusion_matrix"}
        assert cbs["r_f"].toolTip() == "resonator freq"
        assert cbs["confusion_matrix"].toolTip() == "matrix desc"
        assert "0.95" not in cbs["confusion_matrix"].text()
        assert "freq" not in cbs["r_f"].text()
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


def test_writeback_structured_matrix_bounded_table_copy_and_long_bounded(qapp):
    """A2 — 3×3 shows 3 \u00d7 3 matrix + read-only table, Copy JSON, long bounded."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    item = MetaDictWriteback(
        target_name="confusion_matrix", description="d", proposed_value=matrix
    )
    item.session_id = "md-1"
    widget.resize(600, 400)
    widget.populate([item])
    widget.show()
    qapp.processEvents()
    try:
        # bounded summary
        prop = next(
            l
            for l in widget.findChildren(QLabel)
            if "writebackProposed" in l.objectName()
        )
        assert prop.text() == "3 \u00d7 3 matrix"
        # compact read-only table
        tables = widget.findChildren(QTableWidget)
        assert len(tables) == 1
        tbl = tables[0]
        assert tbl.rowCount() == 3 and tbl.columnCount() == 3
        assert tbl.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers
        assert tbl.selectionMode() == QTableWidget.SelectionMode.NoSelection
        first = tbl.item(0, 0)
        assert first is not None
        assert first.text() == "0.9500"
        assert first.textAlignment() & Qt.AlignmentFlag.AlignCenter
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
            hbar2 = w2._scroll.horizontalScrollBar()
            assert hbar2 is not None
            assert (
                w2._scroll.horizontalScrollBarPolicy()
                == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
        finally:
            w2.close()
    finally:
        widget.close()


def test_writeback_responsive_reflow_scroll_and_fixed_apply(qapp):
    """A3 — 500 px wide single-line vs 400 px narrow stacked, vertical scroll, fixed Apply."""
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
        # wide: checkbox and current share Y
        cbs = widget.findChildren(QCheckBox)
        curs = [
            l
            for l in widget.findChildren(QLabel)
            if l.objectName() == "writebackCurrent"
        ]
        assert cbs and curs
        assert abs(cbs[0].y() - curs[0].y()) < 30
        # vertical scroll, no horizontal overflow, Apply fixed visible
        assert (
            widget._scroll.horizontalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        assert (
            widget._scroll.verticalScrollBarPolicy()
            == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        hbar = widget._scroll.horizontalScrollBar()
        assert hbar is not None
        assert hbar.maximum() == 0
        assert widget._apply_btn.isVisible()
        assert widget._apply_btn.parent() is widget
        # narrow
        widget.resize(400, 400)
        qapp.processEvents()
        widget._update_responsive()
        qapp.processEvents()
        # in narrow, checkbox above current
        assert cbs[0].y() < curs[0].y() - 5
        hbar_n = widget._scroll.horizontalScrollBar()
        assert hbar_n is not None
        assert hbar_n.maximum() == 0
        assert widget._apply_btn.isVisible()
        # Apply remains outside scroll viewport
        assert widget._apply_btn.geometry().y() > widget._scroll.geometry().y()
    finally:
        widget.close()
