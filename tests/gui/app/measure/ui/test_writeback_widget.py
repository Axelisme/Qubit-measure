from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QPoint, Qt
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
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    RunRequest,
    SessionEnv,
    WaveformWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.cfg_schemas import waveform_cfg_to_value
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import (
    WritebackService,
    WritebackValues,
)
from zcu_tools.gui.app.measure.services.writeback_control import WritebackControlFacet
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.app.measure.ui.writeback_widget import WritebackWidget
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.binding.fields import ScalarField
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv as ParameterSessionEnv
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry
from zcu_lab.v2.fake.freq.core import FakeFreqRunResult
from zcu_lab.v2.fake.freq.gui import FakeFreqAdapter, FakeFreqAnalyzeParams


@pytest.fixture
def waveform_writeback(tmp_path: Path):
    before = registry_state()
    draft = None
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "Result",
            database_root=tmp_path / "Database",
        )
        bus = BaseEventBus()
        context = ContextService(
            SessionState(ParameterSessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        control = ContextControlFacet(context=context)
        control.add_component("Q1", "fake/drive/a", {})
        target = control.read_context().target
        control.set_waveform(target, "Q1.waveforms.base", ConstWaveformCfg(length=0.1))
        editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
        service = WritebackService(editor, control, entry_ledger=context.entry_ledger)
        spec, value = waveform_cfg_to_value(ConstWaveformCfg(length=0.1))
        draft = service.create_draft(
            [
                WaveformWriteback(
                    "Q1.waveforms.base", "Waveform", edit_schema=CfgSchema(spec, value)
                )
            ],
            parameters=control.read_context(),
        )
        service.apply_draft(draft)
        yield control, service, draft
        assert registry_state() == installed
    finally:
        if draft is not None:
            draft.teardown()
        restore_registry(before)


@pytest.fixture
def multiple_waveform_writeback(waveform_writeback):
    control, service, _original = waveform_writeback
    names = ("Q1.waveforms.base", "Q1.waveforms.other")
    control.set_waveform(
        control.read_context().target, names[1], ConstWaveformCfg(length=0.2)
    )
    proposals = []
    for name, length in zip(names, (0.1, 0.2), strict=True):
        spec, value = waveform_cfg_to_value(ConstWaveformCfg(length=length))
        proposals.append(
            WaveformWriteback(name, name, edit_schema=CfgSchema(spec, value))
        )
    draft = service.create_draft(proposals, parameters=control.read_context())
    try:
        service.apply_draft(draft)
        yield control, service, draft, names
    finally:
        draft.teardown()


def _waveform_length_field(service, draft, item_id: str) -> ScalarField:
    field = service.get_item_draft(draft, item_id).root.fields["length"]
    assert isinstance(field, ScalarField)
    return field


def _make_ctx() -> SessionEnv:
    return SessionEnv(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def _default_analyze_params(
    adapter: FakeFreqAdapter, result: FakeFreqRunResult, ctx: SessionEnv
) -> FakeFreqAnalyzeParams:
    return adapter.get_analyze_params(result, ctx)


def test_writeback_widget_lists_items_and_edit_buttons(qapp):
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict

    run_plots = Plots(NonPresentingHost())
    result = adapter.run(
        RunRequest(soc=ctx.soc, soccfg=ctx.soccfg, device_snapshot={}),
        schema_to_raw_dict(schema, ctx.md, ctx.ml),
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
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
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
    fixed = MetaDictWriteback(
        "driver.frequency", "Frequency", proposed_value=20.09, stderr=0.18
    )
    unset = MetaDictWriteback("driver.duration", "Duration", proposed_value=0.25)
    ext = MetaDictWriteback("general.ext.value", "Value", proposed_value=9.0)
    items = [fixed, unset, ext]
    for index, item in enumerate(items, 1):
        item.session_id = f"md-{index}"
    facts = {
        "md-1": WritebackValues(
            actual_path="Q1.frequency", current=19.8, proposed=20.09, unit="MHz"
        ),
        "md-2": WritebackValues(
            actual_path="Q1.duration", current=None, proposed=0.25, unit="us"
        ),
        "md-3": WritebackValues(
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
    scalar = MetaDictWriteback(
        "driver.ext.value", "Value", proposed_value=20.09, stderr=0.18
    )
    scalar.session_id = "md-1"
    table = MetaDictWriteback("driver.ext.matrix", "Matrix", proposed_value=matrix)
    table.session_id = "md-2"
    values = {
        "md-1": WritebackValues(
            actual_path="Q1.ext.value", current=19.8, proposed=20.09, unit=None
        ),
        "md-2": WritebackValues(
            actual_path="Q1.ext.matrix", current=None, proposed=matrix, unit=None
        ),
    }
    ctrl = MagicMock()
    applied = {"md-1": False, "md-2": False}
    ctrl.get_writeback_applied_for_pane.return_value = applied
    ctrl.writeback_control.get_writeback_applied_for_pane.return_value = applied
    ctrl.get_writeback_summaries_for_pane.return_value = {
        "md-1": ("18.0", "19.0"),  # stale baseline, not the supplied live facts
        "md-2": ("—", "2 × 2 matrix"),
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
        applied["md-1"] = True
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
    item = MetaDictWriteback("driver.ext.value", "Value", proposed_value=1.0)
    item.session_id = "md-1"
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {}
    widget = WritebackWidget(ctrl, tab_id="captured")
    try:
        with pytest.raises(KeyError, match="md-1"):
            widget.populate([item], {})
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def _assert_cfg_row_invalid(row: QFrame, name: str) -> None:
    assert row.findChildren(QCheckBox)[0].text() == f"{name}*"
    assert (
        row.findChildren(QLabel, "writebackProposed")[0].text()
        == "Invalid configuration"
    )
    for label_name in ("writebackCurrent", "writebackActualPath"):
        assert (
            row.findChildren(QLabel, label_name)[0].text()
            == "Unavailable (invalid configuration)"
        )


def _assert_cfg_row_repaired(row: QFrame, name: str) -> None:
    assert row.findChildren(QCheckBox)[0].text() == f"{name}*"
    assert row.findChildren(QLabel, "writebackActualPath")[0].text() == name
    assert row.findChildren(QLabel, "writebackCurrent")[0].text() == "map[2]"
    assert row.findChildren(QLabel, "writebackProposed")[0].text() == "map[2]"


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
@pytest.mark.parametrize("repair_order", [(0, 1), (1, 0)])
def test_multiple_invalid_cfg_items_repair_independently_in_each_pane(
    qapp,
    multiple_waveform_writeback,
    pane: Literal["analysis", "post_analysis"],
    repair_order,
) -> None:
    control, service, draft, names = multiple_waveform_writeback
    state = State(SessionEnv(parameters=control.read_context(), soc=None, soccfg=None))
    state.add_tab(
        "captured", Session(adapter_name="fake", adapter=MagicMock(), cfg=MagicMock())
    )
    getattr(state.get_tab("captured"), pane).writeback_draft = draft
    facet = WritebackControlFacet(
        state=state,
        guard=MagicMock(),
        writeback=service,
        resource_versions=lambda: {},
        bus=BaseEventBus(),
    )
    ctrl = MagicMock()
    ctrl.writeback_control = facet
    ctrl.get_writeback_item_draft_for_pane = facet.get_writeback_item_draft_for_pane
    ctrl.get_writeback_applied_for_pane = facet.get_writeback_applied_for_pane
    widget = WritebackWidget(ctrl, tab_id="captured", pane=pane)
    try:
        widget.populate(service.preview_draft(draft), service.preview_values(draft))
        widget.show()
        rows = widget.findChildren(QFrame, "writebackRow")
        assert len(rows) == 2
        fields = [
            _waveform_length_field(service, draft, item.session_id)
            for item in draft.items
        ]
        rows[1].findChildren(QCheckBox)[0].setChecked(False)
        for row, field, name in zip(rows, fields, names, strict=True):
            row.findChildren(QPushButton)[0].click()
            field.set_text("")
            _assert_cfg_row_invalid(row, name)
            widget.findChildren(QDialog)[0].reject()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        with pytest.raises(RuntimeError):
            facet.get_writeback_values_for_pane("captured", pane)
        repaired = []
        for index in repair_order:
            row, field = rows[index], fields[index]
            row.findChildren(QPushButton)[0].click()
            field.set_text(str(0.5 + index / 10))
            _assert_cfg_row_repaired(row, names[index])
            sibling = 1 - index
            if sibling not in repaired:
                _assert_cfg_row_invalid(rows[sibling], names[sibling])
            widget.findChildren(QDialog)[0].reject()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            repaired.append(index)
        for row, name in zip(rows, names, strict=True):
            _assert_cfg_row_repaired(row, name)
        assert [row.findChildren(QCheckBox)[0].isChecked() for row in rows] == [
            True,
            False,
        ]
        facts = facet.get_writeback_values_for_pane("captured", pane)
        assert [
            ConstWaveformCfg.model_validate(facts[item.session_id].proposed).length
            for item in draft.items
        ] == [0.5, 0.6]
        assert service.get_all_applied(draft) == {
            item.session_id: False for item in draft.items
        }
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
@pytest.mark.parametrize("invalid_text", ["", "not-a-number"])
def test_invalid_cfg_edit_can_close_and_repair_without_stale_applied_state(
    qapp,
    waveform_writeback,
    pane: Literal["analysis", "post_analysis"],
    invalid_text: str,
) -> None:
    _control, service, draft = waveform_writeback
    item = draft.items[0]
    model = service.get_item_draft(draft, item.session_id)
    field = model.root.fields["length"]
    assert isinstance(field, ScalarField)
    ctrl = MagicMock()
    ctrl.get_writeback_item_draft_for_pane.return_value = model
    ctrl.get_writeback_applied_for_pane.side_effect = lambda *_: (
        service.get_all_applied(draft)
    )
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = (
        lambda *_, **kwargs: service.preview_values(draft, **kwargs)
    )
    widget = WritebackWidget(ctrl, tab_id="captured", pane=pane)
    try:
        widget.populate(service.preview_draft(draft), service.preview_values(draft))
        widget.show()
        edit = next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "Edit"
        )
        edit.click()
        field.set_text(invalid_text)
        assert not model.is_valid()
        assert widget.findChildren(QCheckBox)[0].text() == "Q1.waveforms.base*"
        assert (
            widget.findChildren(QLabel, "writebackProposed")[0].text()
            == "Invalid configuration"
        )
        for name in ("writebackCurrent", "writebackActualPath"):
            assert (
                widget.findChildren(QLabel, name)[0].text()
                == "Unavailable (invalid configuration)"
            )
        with pytest.raises(RuntimeError):
            service.preview_values(draft)
        dialog = widget.findChildren(QDialog)[0]
        dialog.reject()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        edit.click()
        field.set_text("0.5")
        assert model.is_valid()
        facts = service.preview_values(draft)[item.session_id]
        assert facts.proposed["length"] == 0.5
        assert (
            widget.findChildren(QLabel, "writebackActualPath")[0].text()
            == facts.actual_path
        )
        assert widget.findChildren(QLabel, "writebackCurrent")[0].text() == "map[2]"
        assert widget.findChildren(QLabel, "writebackProposed")[0].text() == "map[2]"
        widget.findChildren(QDialog)[0].reject()
        assert widget.findChildren(QCheckBox)[0].isChecked()
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
@pytest.mark.parametrize("repopulate", [False, True])
def test_cfg_edit_and_retarget_refresh_live_facts_and_applied_state(
    qapp,
    waveform_writeback,
    pane: Literal["analysis", "post_analysis"],
    *,
    repopulate: bool,
) -> None:
    control, service, draft = waveform_writeback
    item = draft.items[0]
    model = service.get_item_draft(draft, item.session_id)
    ctrl = MagicMock()
    ctrl.get_writeback_item_draft_for_pane.return_value = model
    ctrl.get_writeback_applied_for_pane.side_effect = lambda *_: (
        service.get_all_applied(draft)
    )
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = (
        lambda *_, **kwargs: service.preview_values(draft, **kwargs)
    )
    ctrl.set_writeback_item_for_pane.side_effect = lambda _tab, _pane, sid, **kwargs: (
        draft.edit(sid, **kwargs)
    )
    widget = WritebackWidget(ctrl, tab_id="captured", pane=pane)
    try:
        widget.populate(service.preview_draft(draft), service.preview_values(draft))
        widget.show()
        checkbox = widget.findChildren(QCheckBox)[0]
        assert not checkbox.text().endswith("*")
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
            and candidate.windowTitle() == "Edit Config: Q1.waveforms.base"
        )
        if repopulate:
            widget.populate(service.preview_draft(draft), service.preview_values(draft))
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            checkbox = widget.findChildren(QCheckBox)[0]
        model.set_target("length", 0.2)
        assert checkbox.text().endswith("*")
        name_edit = dialog.findChildren(QLineEdit)[0]
        name_edit.setText("Q1.waveforms.other")
        name_edit.editingFinished.emit()
        assert checkbox.text() == "Q1.waveforms.other*"
        texts = [label.text() for label in widget.findChildren(QLabel)]
        assert "Q1.waveforms.other" in texts
        assert "—" in texts
        control.set_waveform(
            control.read_context().target,
            "Q1.waveforms.other",
            ConstWaveformCfg(length=0.4),
        )
        model.set_target("length", 0.3)
        assert "—" not in [label.text() for label in widget.findChildren(QLabel)]
        assert service.preview_values(draft)[item.session_id].proposed["length"] == 0.3
        next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "Close"
        ).click()
        qapp.processEvents()
        assert checkbox.isChecked()
        assert checkbox.text().endswith("*")
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("pane", ["analysis", "post_analysis"])
def test_retarget_edit_refreshes_captured_path_current_and_proposal(
    qapp, pane: Literal["analysis", "post_analysis"]
) -> None:
    item = MetaDictWriteback("driver.frequency", "Value", proposed_value=20.09)
    item.session_id = "md-1"
    original = WritebackValues(
        actual_path="Q1.frequency", current=19.8, proposed=20.09, unit="MHz"
    )
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {}
    ctrl.get_writeback_summaries_for_pane.return_value = {"md-1": ("19.8", "20.09")}

    def commit_edit(
        tab_id: str,
        requested_pane: str,
        session_id: str,
        *,
        target_name: str,
        proposed_value: object,
    ) -> None:
        assert (tab_id, requested_pane, session_id) == ("captured", pane, "md-1")
        assert target_name == "driver.ext.other"
        ctrl.writeback_control.get_writeback_values_for_pane.return_value = {
            "md-1": WritebackValues(
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
        widget.populate([item], {"md-1": original})
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
