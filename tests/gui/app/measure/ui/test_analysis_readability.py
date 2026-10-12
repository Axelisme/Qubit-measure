"""Observable display-only contracts for analysis parameters and writeback."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
)
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, ParameterWriteback
from zcu_tools.gui.app.measure.services.writeback import WritebackValues
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.writeback_widget import WritebackWidget


@pytest.fixture
def scalar_ledger(qapp: QApplication):
    item = ParameterWriteback(
        "Q1.t1", "Lifetime", proposed_value=33.8724123470065, stderr=66.9146381234567
    )
    item.session_id = "lifetime"
    facts = {
        item.session_id: WritebackValues(
            actual_path="Q1.t1",
            current=20.1234567890123,
            proposed=item.proposed_value,
            unit="us",
        )
    }
    ctrl = MagicMock()
    ctrl.get_writeback_applied_for_pane.return_value = {}
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = lambda *a, **k: {
        item.session_id: WritebackValues(
            actual_path="Q1.t1",
            current=20.1234567890123,
            proposed=item.proposed_value,
            unit="us",
        )
    }
    widget = WritebackWidget(ctrl, tab_id="t")
    widget.populate([item], facts)
    widget.resize(800, 300)
    widget.show()
    qapp.processEvents()
    yield widget, item
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def value_labels(widget: WritebackWidget) -> tuple[QLabel, QLabel]:
    current = widget.findChild(QLabel, "writebackCurrent")
    proposed = widget.findChild(QLabel, "writebackProposed")
    assert current is not None and proposed is not None
    return current, proposed


def test_scalar_values_and_stderr_are_compact_with_full_precision_tooltips(
    scalar_ledger,
):
    widget, item = scalar_ledger
    current, proposed = value_labels(widget)
    assert current.text() == "20.1235 us"
    assert proposed.text() == "33.8724 ± 66.9146 us"
    assert current.toolTip() == "20.1234567890123 us"
    assert proposed.toolTip() == "33.8724123470065 ± 66.9146381234567 us"
    assert item.proposed_value == 33.8724123470065
    assert item.stderr == 66.9146381234567


def test_value_labels_elide_in_the_middle_and_restore_on_resize(scalar_ledger, qapp):
    widget, _ = scalar_ledger
    _, proposed = value_labels(widget)
    proposed.setFixedWidth(100)
    qapp.processEvents()
    assert "…" in proposed.text()
    assert proposed.text().startswith("33")
    assert proposed.text().endswith("us")
    assert (
        proposed.fontMetrics().horizontalAdvance(proposed.text())
        <= proposed.contentsRect().width()
    )
    assert proposed.toolTip() == "33.8724123470065 ± 66.9146381234567 us"
    proposed.setFixedWidth(300)
    qapp.processEvents()
    assert proposed.text() == "33.8724 ± 66.9146 us"


def test_edit_retains_full_precision_and_refreshes_display_and_tooltip(
    scalar_ledger, qapp
):
    widget, item = scalar_ledger
    next(b for b in widget.findChildren(QPushButton) if b.text() == "Edit").click()
    dialog = next(d for d in widget.findChildren(QDialog) if d.isVisible())
    try:
        current = dialog.findChild(QLabel, "writebackCurrentReadonly")
        assert current is not None
        assert current.text() == "20.1234567890123 us"
        edit = dialog.findChildren(QLineEdit)[1]
        assert edit.text() == "33.8724123470065"
        edit.setText("5999.30228396247")
        next(b for b in dialog.findChildren(QPushButton) if b.text() == "Save").click()
        _, proposed = value_labels(widget)
        assert proposed.text() == "5999.3 ± 66.9146 us"
        assert proposed.toolTip() == "5999.30228396247 ± 66.9146381234567 us"
        assert item.proposed_value == 5999.30228396247
    finally:
        dialog.close()


def test_unavailable_reason_wraps_as_warning_without_rewriting_owner_message(qapp):
    widget = WritebackWidget(MagicMock(), tab_id="t")
    reason = "Q1: Sample writeback requires $flux_dev: flux_dev: Alias is not bound"
    try:
        widget.populate([], {}, unavailable_reasons=(reason,))
        warning = widget.findChild(QLabel, "writebackUnavailableReason")
        assert warning is not None
        assert warning.text() == f"⚠ {reason}"
        assert warning.wordWrap()
        assert "#b7791f" in warning.styleSheet()
    finally:
        widget.deleteLater()
        qapp.processEvents()


@dataclass
class NoParameters:
    pass


@dataclass
class WithParameters:
    threshold: float = 0.5


def test_analysis_parameter_hint_tracks_empty_and_populated_forms(qapp):
    ctrl = MagicMock()
    ctrl.get_left_panel_width.return_value = 500
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.get_adapter_guide.return_value = {}
    tab = ExpTabWidget("t", ctrl, AdapterCapabilities())
    tab.show()
    try:
        tabs = next(
            t
            for t in tab.findChildren(QTabWidget)
            if any(t.tabText(i) == "Analysis" for i in range(t.count()))
        )
        tabs.setCurrentIndex(
            next(i for i in range(tabs.count()) if tabs.tabText(i) == "Analysis")
        )
        tab.populate_analyze_params(NoParameters())
        qapp.processEvents()
        hint = tab.analyze_form.findChild(QLabel, "analysisNoParameters")
        assert hint is not None
        assert hint.isVisible()
        assert hint.text() == "This analysis has no parameters."
        assert "gray" in hint.styleSheet()
        assert tab.read_analyze_params() == NoParameters()
        tab.populate_analyze_params(WithParameters())
        qapp.processEvents()
        assert not hint.isVisible()
        tab.populate_analyze_params(NoParameters())
        qapp.processEvents()
        assert hint.isVisible()
    finally:
        tab.close()
        tab.deleteLater()
        qapp.processEvents()
