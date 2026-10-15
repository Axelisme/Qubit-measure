from __future__ import annotations

import json
from typing import Literal
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QPoint, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.services.writeback import (
    WritebackValues,
)
from zcu_tools.gui.app.measure.ui.writeback_widget import WritebackWidget
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._components import ResonatorSchema, WiringSchema
from tests.gui.app.measure._synthetic import PairAdapter, SampleParams, SampleRun


def _make_ctx() -> SessionEnv:
    parameters = parameter_capture(
        {
            "R1": ResonatorSchema(
                kind="resonator",
                freq=6000.0,
                kappa=2.0,
                wiring=WiringSchema(ch=0, ro_ch=0),
                ext={"time_of_fly": 0.0},
            )
        }
    ).bind_aliases((AliasRow("resonator", "R1"),), {})
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


def _parameter_values(items: list[WritebackItem]) -> dict[str, WritebackValues]:
    values: dict[str, WritebackValues] = {}
    for item in items:
        if not isinstance(item, ParameterWriteback):
            raise TypeError("This widget fixture supplies parameter proposals")
        values[item.session_id] = WritebackValues(
            actual_path=item.target_name,
            current=None,
            proposed=item.proposed_value,
            unit=None,
        )
    return values


def _default_analyze_params(
    adapter: PairAdapter, result: SampleRun, ctx: SessionEnv
) -> SampleParams:
    return adapter.get_analyze_params(result, ctx)


def test_writeback_widget_lists_items_and_edit_buttons(qapp):
    ctx = _make_ctx()
    adapter = PairAdapter()
    from zcu_tools.program.v2 import SweepCfg

    from tests.gui.app.measure._synthetic import SampleCfg

    run_plots = Plots(NonPresentingHost())
    result = adapter.run(
        RunRequest(soc=ctx.soc, soccfg=ctx.soccfg, cfg_context=None),
        SampleCfg(sweep=SweepCfg(start=5.0, stop=6.0, expts=11, step=0.1)),
        context=RunContext(
            ctx.soc, ctx.soccfg, run_plots, devices={}, cancel_signal=StopSignal()
        ),
    )
    run_plots.finish()
    run_plots.release()
    analysis_plots = Plots(NonPresentingHost())
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=_default_analyze_params(adapter, result, ctx),
            parameters=ctx.parameters,
        ),
        plots=analysis_plots,
    )
    analysis_plots.finish()
    analysis_plots.release()
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
    widget.populate(items, _parameter_values(items))
    selected = [it for it in items if it.selected]
    edit_buttons = [w for w in widget.findChildren(QPushButton) if w.text() == "Edit"]

    # The fixture supplies two editable scalar proposals.
    assert len(selected) == len(items)  # all selected by default
    assert {it.target_name for it in items} == {"r_f", "rf_w"}
    assert len(edit_buttons) == 2


@pytest.mark.parametrize(
    "value",
    [None, [], [[], []], [1, 2], [[1], [2, 3]], [[{}]], [[[1]]]],
)
def test_non_matrix_proposals_do_not_render_matrix_table(qapp, value):
    item = ParameterWriteback(
        target_name="proposal", description="d", proposed_value=value
    )
    item.session_id = "param-1"
    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    try:
        widget.populate([item], _parameter_values([item]))
        assert widget.findChildren(QTableWidget) == []
        checks = widget.findChildren(QCheckBox)
        assert len(checks) == 1
        assert checks[0].isChecked()
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_writeback_widget_non_scalar_item_is_read_only(qapp):
    """A non-scalar md item (e.g. a confusion matrix) renders selectable but
    with no Edit button — it is a derived value applied verbatim."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    scalar = ParameterWriteback(target_name="fid", description="d", proposed_value=0.95)
    scalar.session_id = "param-1"
    nonscalar = ParameterWriteback(
        target_name="confusion_matrix", description="d", proposed_value=matrix
    )
    nonscalar.session_id = "param-2"

    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    widget.populate([scalar, nonscalar], _parameter_values([scalar, nonscalar]))

    edit_buttons = [w for w in widget.findChildren(QPushButton) if w.text() == "Edit"]
    # Only the scalar item gets an Edit button; the matrix is read-only.
    assert len(edit_buttons) == 1
    # Both items are still selectable for apply (the matrix label shows its value).
    checks = widget.findChildren(QCheckBox)
    assert len(checks) == 2
    assert any("confusion_matrix" in cb.text() for cb in checks)
    assert all(cb.isChecked() for cb in checks)


def test_writeback_widget_projects_draft_owned_applied_state(qapp):
    item = ParameterWriteback(target_name="r_f", description="d", proposed_value=6000.0)
    item.session_id = "param-1"
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {"param-1": False}
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")

    widget.populate([item], _parameter_values([item]))

    checkbox = widget.findChildren(QCheckBox)[0]
    assert checkbox.text() == "r_f*"
    assert checkbox.font().bold()
    assert any(
        label.text() == "* = not applied" for label in widget.findChildren(QLabel)
    )

    ctrl.get_writeback_applied_for_pane.return_value = {"param-1": True}
    widget.populate([item], _parameter_values([item]))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    checkbox = widget.findChildren(QCheckBox)[0]
    assert checkbox.text() == "r_f"
    assert not checkbox.font().bold()


def test_writeback_widget_does_not_hide_applied_projection_failures(qapp):
    item = ParameterWriteback(target_name="r_f", description="d", proposed_value=6000.0)
    item.session_id = "param-1"
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.side_effect = RuntimeError("projection failed")
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")

    with pytest.raises(RuntimeError, match="projection failed"):
        widget.populate([item], _parameter_values([item]))


def _assert_ledger_panel(widget: WritebackWidget) -> None:
    # shared backgrounds / continuous borders — observed via palette and geometry
    panel = widget.findChild(QFrame, "writebackPanel")
    assert panel is not None
    rows = panel.findChildren(QFrame, "writebackRow")
    panel_layout = panel.layout()
    if panel_layout is None:
        raise RuntimeError("Writeback panel has no Qt layout")
    assert panel.objectName() == "writebackPanel"
    # Rendered Base background is shared between panel and rows.
    panel_bg = panel.palette().color(panel.backgroundRole())
    assert panel_bg == widget.palette().color(QPalette.ColorRole.Base)
    assert rows, "no rows rendered"
    first_bg = rows[0].palette().color(rows[0].backgroundRole())
    assert first_bg == widget.palette().color(QPalette.ColorRole.Base)
    assert panel_bg == first_bg
    # container has no extra spacing, rows stack continuously with 1px border gap
    assert panel_layout.spacing() == 0
    cm = panel_layout.contentsMargins()
    assert cm.left() == 0 and cm.top() == 0 and cm.right() == 0 and cm.bottom() == 0
    if len(rows) > 1:
        # second row directly follows first (allow 1px border)
        assert abs(rows[1].geometry().top() - (rows[0].geometry().bottom() + 1)) <= 1
    # row internal margins per spec 8,4,8,4
    lay0 = rows[0].layout()
    assert lay0 is not None
    lm = lay0.contentsMargins()
    assert lm.left() == 8 and lm.top() == 4 and lm.right() == 8 and lm.bottom() == 4
    # rendered-image border checks — bounded deterministic sampling of
    # visible outer panel border and adjacent-row divider (palette alone
    # would hide a missing stylesheet border).
    panel_grab = panel.grab()
    img = panel_grab.toImage()
    w = panel.width()
    h = panel.height()
    # offscreen QT_QPA_PLATFORM=offscreen has devicePixelRatio 1.0; scale if hidpi
    dpr = panel_grab.devicePixelRatio()
    scale = int(dpr) if dpr != 1 else 1
    # QImage pixels quantize the palette's 16-bit channels to 8 bits.
    outer_expected = QColor.fromRgba(
        widget.palette().color(QPalette.ColorRole.Mid).rgba()
    )
    divider_expected = QColor.fromRgba(
        widget.palette().color(QPalette.ColorRole.Midlight).rgba()
    )
    # outer border — top-center and left-mid avoid rounded corners
    assert img.pixelColor((w // 2) * scale, 0 * scale) == outer_expected, (
        f"outer top border {img.pixelColor((w // 2) * scale, 0).name().lower()}"
    )
    assert img.pixelColor(0 * scale, (h // 2) * scale) == outer_expected
    # adjacent-row divider — horizontal 1px line at row boundary
    assert len(rows) >= 2
    y_div = rows[1].pos().y() - 1
    assert 0 <= y_div < h, f"divider y {y_div} out of {h}"
    assert img.pixelColor((w // 2) * scale, y_div * scale) == divider_expected
    assert img.pixelColor((w // 4) * scale, y_div * scale) == divider_expected
    # Interior pixels use Base rather than the border colour.
    assert img.pixelColor((w // 2) * scale, 1 * scale) == widget.palette().color(
        QPalette.ColorRole.Base
    )
    assert img.pixelColor(
        (w // 2) * scale, (y_div + 1) * scale
    ) == widget.palette().color(QPalette.ColorRole.Base)


def test_writeback_compact_ledger_target_only_centered_and_equal_actions(qapp):
    """A1 — target-only labels, tooltip, centered Current → Proposed,
    shared backgrounds/borders and equal 56x26 actions."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    scalar = ParameterWriteback(
        target_name="r_f", description="resonator freq", proposed_value=6000.0
    )
    scalar.session_id = "param-1"
    nonscalar = ParameterWriteback(
        target_name="confusion_matrix", description="matrix desc", proposed_value=matrix
    )
    nonscalar.session_id = "param-2"
    widget = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    widget.resize(600, 400)
    widget.populate([scalar, nonscalar], _parameter_values([scalar, nonscalar]))
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
        _assert_ledger_panel(widget)
        # identical Edit/Copy geometry
        edit_btns = [b for b in widget.findChildren(QPushButton) if b.text() == "Edit"]
        copy_btns = [b for b in widget.findChildren(QPushButton) if b.text() == "Copy"]
        assert len(edit_btns) == 1 and len(copy_btns) == 1
        for b in edit_btns + copy_btns:
            assert b.width() == 56 and b.height() == 26
            assert b.size().width() == 56 and b.size().height() == 26
    finally:
        widget.close()


def _assert_long_value_bounded(qapp, reference: WritebackWidget) -> None:
    # long value cannot widen ledger
    long_val = list(range(200))
    w2 = WritebackWidget(MagicMock(), tab_id="tab-1", pane="analysis")
    long_item = ParameterWriteback(
        target_name="long_list", description="d", proposed_value=long_val
    )
    long_item.session_id = "param-1"
    w2.resize(600, 400)
    w2.populate([long_item], _parameter_values([long_item]))
    w2.show()
    qapp.processEvents()
    try:
        prop2 = next(
            label
            for label in w2.findChildren(QLabel)
            if "writebackProposed" in label.objectName()
        )
        assert prop2.text() == "list[200]"
        # size hint remains bounded
        assert w2.sizeHint().width() < reference.sizeHint().width() + 200
        assert w2.findChildren(QTableWidget) == []
        # No nested writeback scroll — long value does not create inner scroll
        assert w2.findChild(QScrollArea, "writebackScroll") is None
        # No horizontal overflow: panel width bounded
        assert w2.findChild(QFrame, "writebackPanel").width() <= w2.width() + 4
    finally:
        w2.close()


def test_writeback_proposed_matrix_has_view_while_current_stays_summary_only(qapp):
    """A2 — Proposed shows the 3×3 view; Current stays bounded summary-only."""
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    controller = MagicMock()
    controller.get_writeback_summaries_for_pane.return_value = {
        "param-1": ("3 × 3 matrix", "3 × 3 matrix")
    }
    widget = WritebackWidget(controller, tab_id="tab-1", pane="analysis")
    item = ParameterWriteback(
        target_name="confusion_matrix", description="d", proposed_value=matrix
    )
    item.session_id = "param-1"
    widget.resize(600, 400)
    widget.populate(
        [item],
        {
            item.session_id: WritebackValues(
                actual_path=item.target_name, current=matrix, proposed=matrix, unit=None
            )
        },
    )
    widget.show()
    qapp.processEvents()
    try:
        current = widget.findChild(QLabel, "writebackCurrent")
        assert current is not None
        assert current.text() == "3 × 3 matrix"

        # Proposed keeps its bounded heading and adds the readable matrix view.
        prop = next(
            label
            for label in widget.findChildren(QLabel)
            if "writebackProposed" in label.objectName()
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
        _assert_long_value_bounded(qapp, widget)
    finally:
        widget.close()


def _assert_matrix_row_reflows(row: QFrame, *, narrow: bool) -> None:
    checks = row.findChildren(QCheckBox)
    currents = row.findChildren(QLabel, "writebackCurrent")
    paths = row.findChildren(QLabel, "writebackActualPath")
    actions = row.findChildren(QPushButton)
    tables = row.findChildren(QTableWidget)
    assert (
        len(checks) == len(currents) == len(paths) == len(actions) == len(tables) == 1
    )
    check, current, path, action, table = (
        checks[0],
        currents[0],
        paths[0],
        actions[0],
        tables[0],
    )
    assert all(child.isVisible() for child in (check, current, path, action, table))
    check_y = check.mapTo(row, QPoint()).y() + check.height() / 2
    current_y = current.mapTo(row, QPoint()).y() + current.height() / 2
    action_y = action.mapTo(row, QPoint()).y() + action.height() / 2
    assert abs(check_y - action_y) < (check.height() + action.height()) / 2
    if narrow:
        assert check_y + 10 < current_y
    else:
        assert abs(check_y - current_y) < (check.height() + current.height()) / 2
    assert path.mapTo(row, QPoint()).y() > current.mapTo(row, QPoint()).y()
    assert table.mapTo(row, QPoint()).y() > path.mapTo(row, QPoint()).y()
    assert table.rowCount() == table.columnCount() == 2
    assert table.editTriggers() == QTableWidget.EditTrigger.NoEditTriggers
    for (r, c), expected in (((0, 1), 0.1), ((1, 0), 0.2)):
        cell = table.item(r, c)
        assert cell is not None
        assert float(cell.text()) == expected
        assert not cell.flags() & Qt.ItemFlag.ItemIsEditable


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_static_units_display_once_after_stderr_without_converting_values(
    qapp, pane: Literal["analysis", "post_analysis"]
) -> None:
    fixed = ParameterWriteback(
        "driver.frequency", "Frequency", proposed_value=20.09, stderr=0.18
    )
    unset = ParameterWriteback("driver.duration", "Duration", proposed_value=0.25)
    ext = ParameterWriteback("general.ext.value", "Value", proposed_value=9.0)
    items = [fixed, unset, ext]
    for index, item in enumerate(items, 1):
        item.session_id = f"param-{index}"
    facts = {
        "param-1": WritebackValues(
            actual_path="Q1.frequency", current=19.8, proposed=20.09, unit="MHz"
        ),
        "param-2": WritebackValues(
            actual_path="Q1.duration", current=None, proposed=0.25, unit="us"
        ),
        "param-3": WritebackValues(
            actual_path="general.ext.value", current=7.0, proposed=9.0, unit=None
        ),
    }
    ctrl = MagicMock()
    ctrl.writeback_control.get_writeback_applied_for_pane.return_value = {}
    widget = WritebackWidget(ctrl, tab_id="captured", pane=pane)
    try:
        widget.populate(items, facts)
        texts = [label.text() for label in widget.findChildren(QLabel)]
        for expected in ("19.8 MHz", "20.09 ± 0.18 MHz", "—", "0.25 us", "7.0", "9.0"):
            assert expected in texts
        assert "— us" not in texts
        assert fixed.proposed_value == 20.09 and fixed.stderr == 0.18
        assert all(item.selected for item in items)
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_captured_paths_stderr_and_matrix_copy_are_visible_in_each_pane(
    qapp, pane: Literal["analysis", "post_analysis"]
) -> None:
    matrix = [[0.9, 0.1], [0.2, 0.8]]
    scalar = ParameterWriteback(
        "driver.ext.value", "Value", proposed_value=20.09, stderr=0.18
    )
    scalar.session_id = "param-1"
    table = ParameterWriteback("driver.ext.matrix", "Matrix", proposed_value=matrix)
    table.session_id = "param-2"
    values = {
        "param-1": WritebackValues(
            actual_path="Q1.ext.value", current=19.8, proposed=20.09, unit=None
        ),
        "param-2": WritebackValues(
            actual_path="Q1.ext.matrix", current=None, proposed=matrix, unit=None
        ),
    }
    ctrl = MagicMock()
    applied = {"param-1": False, "param-2": False}
    ctrl.get_writeback_applied_for_pane.return_value = applied
    ctrl.writeback_control.get_writeback_applied_for_pane.return_value = applied
    ctrl.get_writeback_summaries_for_pane.return_value = {
        "param-1": ("18.0", "19.0"),  # stale baseline, not the supplied live facts
        "param-2": ("—", "2 × 2 matrix"),
    }
    widget = WritebackWidget(ctrl, tab_id="captured", pane=pane)
    try:
        widget.populate([scalar, table], values)
        widget.show()
        for width in (620, 360, 620):
            widget.resize(width, 600)
            qapp.processEvents()
            texts = [label.text() for label in widget.findChildren(QLabel)]
            assert any("Q1.ext.value" in text for text in texts)
            assert any("Q1.ext.matrix" in text for text in texts)
            assert any("±" in text and "0.18" in text for text in texts)
            assert any("19.8" in text for text in texts)
            assert any("20.09" in text for text in texts)
            matrix_row = next(
                row
                for row in widget.findChildren(QFrame, "writebackRow")
                if any(
                    "Q1.ext.matrix" in label.text()
                    for label in row.findChildren(QLabel)
                )
            )
            _assert_matrix_row_reflows(matrix_row, narrow=width == 360)
            assert all(
                "±" not in label.text() for label in matrix_row.findChildren(QLabel)
            )
            assert any(
                "driver.ext.value" in check.text()
                for check in widget.findChildren(QCheckBox)
            )
        copy_button = next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "Copy"
        )
        copy_button.click()
        clipboard = QApplication.clipboard()
        assert clipboard is not None
        assert json.loads(clipboard.text()) == matrix
        assert scalar.selected and table.selected
        unapplied = next(
            check
            for check in widget.findChildren(QCheckBox)
            if check.text() == "driver.ext.value*"
        )
        assert unapplied.font().bold()
        applied["param-1"] = True
        widget.populate([scalar, table], values)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        written = next(
            check
            for check in widget.findChildren(QCheckBox)
            if check.text() == "driver.ext.value"
        )
        assert not written.font().bold()
        widget.populate([], {})
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        assert not widget.findChildren(QCheckBox)
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_nonempty_preview_requires_each_service_fact(qapp) -> None:
    item = ParameterWriteback("driver.ext.value", "Value", proposed_value=1.0)
    item.session_id = "param-1"
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {}
    widget = WritebackWidget(ctrl, tab_id="captured")
    try:
        with pytest.raises(KeyError, match="param-1"):
            widget.populate([item], {})
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_retarget_edit_refreshes_captured_path_current_and_proposal(
    qapp, pane: Literal["analysis", "post_analysis"]
) -> None:
    item = ParameterWriteback("driver.frequency", "Value", proposed_value=20.09)
    item.session_id = "param-1"
    original = WritebackValues(
        actual_path="Q1.frequency", current=19.8, proposed=20.09, unit="MHz"
    )
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {}
    ctrl.get_writeback_summaries_for_pane.return_value = {"param-1": ("19.8", "20.09")}

    def commit_edit(
        tab_id: str,
        requested_pane: str,
        session_id: str,
        *,
        target_name: str,
        proposed_value: object,
    ) -> None:
        assert (tab_id, requested_pane, session_id) == ("captured", pane, "param-1")
        assert target_name == "driver.ext.other"
        ctrl.writeback_control.get_writeback_values_for_pane.return_value = {
            "param-1": WritebackValues(
                actual_path="Q1.ext.other",
                current=7.0,
                proposed=proposed_value,
                unit=None,
            )
        }

    ctrl.set_writeback_item_for_pane.side_effect = commit_edit
    ctrl.writeback_control.set_writeback_item_for_pane.side_effect = commit_edit
    widget = WritebackWidget(ctrl, tab_id="captured", pane=pane)
    try:
        widget.populate([item], {"param-1": original})
        initial_texts = [label.text() for label in widget.findChildren(QLabel)]
        assert "19.8 MHz" in initial_texts
        assert "20.09 MHz" in initial_texts
        widget.show()
        next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "Edit"
        ).click()
        qapp.processEvents()
        dialog = next(
            candidate
            for candidate in qapp.topLevelWidgets()
            if isinstance(candidate, QDialog)
            and candidate.isVisible()
            and candidate.windowTitle() == "Edit Value: driver.frequency"
        )
        inputs = dialog.findChildren(QLineEdit)
        assert len(inputs) == 2
        inputs[0].setText("driver.ext.other")
        inputs[1].setText("9.0")
        next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "Save"
        ).click()
        qapp.processEvents()
        texts = [label.text() for label in widget.findChildren(QLabel)]
        assert any("Q1.ext.other" in text for text in texts)
        assert any(text == "7.0" for text in texts)
        assert any(text == "9.0" for text in texts)
        assert all("MHz" not in text for text in texts)
        assert item.target_name == "driver.ext.other"
        assert item.proposed_value == 9.0
        assert item.selected
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()
