from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import pytest
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLineEdit
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.gui.app.measure.adapter import ParamMeta
from zcu_tools.gui.app.measure.ui.analyze_form import AnalyzeFormWidget
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

from tests.gui._user_input import ComponentDriver


@dataclass
class _TestParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)]
    model: Annotated[Literal["a", "b"], ParamMeta(label="Model")]


@dataclass
class _OptionalParams:
    t0: Annotated[float | None, ParamMeta(label="T0")] = None


@dataclass
class _OptionalIntParams:
    t0: Annotated[int | None, ParamMeta(label="T0")] = None


@dataclass
class _CoreOptions:
    model: Annotated[Literal["lor", "sinc"], FieldMeta(label="Model type")] = "lor"
    cutoff: Annotated[float | None, FieldMeta(label="Gain cutoff")] = None
    skip: Annotated[int, ParamMeta(label="Skip points")] = 2


def test_core_options_and_param_meta_share_visible_form(qapp, tmp_path: Path):
    form = AnalyzeFormWidget()
    form.populate(_CoreOptions())
    form.resize(420, 160)
    form.show()
    qapp.processEvents()
    driver = ComponentDriver(form)
    try:
        driver.screenshot(tmp_path / "01-core-defaults.png")
        labels = [
            widget.text
            for widget in driver.visible().widgets
            if widget.kind == "QLabel"
        ]
        assert labels == ["Model type:", "Gain cutoff:", "Skip points:"]
        controls = [
            widget.text
            for widget in driver.visible().widgets
            if widget.kind in {"QComboBox", "QLineEdit"}
        ]
        assert controls == ["lor", "", "2"]
        driver.choose_option("Model type:", "sinc")
        driver.enter_text("Gain cutoff:", "0.25")
        driver.screenshot(tmp_path / "02-core-edited.png")
        assert form.read_params() == _CoreOptions(model="sinc", cutoff=0.25)
    finally:
        form.close()
        form.deleteLater()


def test_analyze_form_round_trips_values(qapp):
    form = AnalyzeFormWidget()
    form.populate(_TestParams(threshold=0.5, model="a"))

    result = form.read_params()

    assert result == _TestParams(threshold=0.5, model="a")
    assert form.is_valid() is True


def test_analyze_form_has_params_after_populate(qapp):
    form = AnalyzeFormWidget()
    assert form.has_params() is False

    form.populate(_TestParams(threshold=0.5, model="a"))

    assert form.has_params() is True


def test_analyze_form_populate_values_restores_state(qapp):
    form = AnalyzeFormWidget()
    form.populate(_TestParams(threshold=0.5, model="a"))
    form.populate_values(_TestParams(threshold=0.9, model="b"))
    assert form.read_params() == _TestParams(threshold=0.9, model="b")


def test_analyze_form_hydration_does_not_emit_params_changed(qapp):
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.params_changed.connect(lambda values: emitted.append(values))

    form.populate(_TestParams(threshold=0.5, model="a"))
    form.populate_values(_TestParams(threshold=0.9, model="b"))

    assert emitted == []


def _form_layout(form: AnalyzeFormWidget) -> QtWidgets.QFormLayout:
    """Find the form's unique layout through the public Qt object tree."""
    layout = form.findChild(QtWidgets.QFormLayout)
    assert layout is not None
    return layout


def _widgets_by_label(form: AnalyzeFormWidget) -> dict[str, QtWidgets.QWidget]:
    """Observe field widgets by the fixtures' distinct visible row labels."""
    layout = _form_layout(form)
    widgets: dict[str, QtWidgets.QWidget] = {}
    for row in range(layout.rowCount()):
        label_item = layout.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
        assert label_item is not None
        label = label_item.widget()
        assert isinstance(label, QtWidgets.QLabel)
        field_item = layout.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole)
        assert field_item is not None
        widget = field_item.widget()
        assert widget is not None
        assert label.text() not in widgets
        widgets[label.text()] = widget
    return widgets


def test_sync_same_class_preserves_widgets_and_only_hydrates(qapp):
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.params_changed.connect(emitted.append)
    form.sync(_TestParams(threshold=0.5, model="a"))
    widgets = _widgets_by_label(form)
    row_count = _form_layout(form).rowCount()

    form.sync(_TestParams(threshold=0.9, model="b"))

    current_widgets = _widgets_by_label(form)
    assert current_widgets == widgets
    assert all(current_widgets[name] is widget for name, widget in widgets.items())
    assert _form_layout(form).rowCount() == row_count
    assert form.read_params() == _TestParams(threshold=0.9, model="b")
    assert emitted == []


def test_sync_class_change_rebuilds_and_clear_fast_fails(qapp):
    form = AnalyzeFormWidget()
    form.sync(_TestParams(threshold=0.5, model="a"))
    old_widgets = tuple(_widgets_by_label(form).values())

    form.sync(_OptionalParams(t0=1.0))

    current_widgets = _widgets_by_label(form)
    assert all(widget not in current_widgets.values() for widget in old_widgets)
    assert form.read_params() == _OptionalParams(t0=1.0)

    form.clear()
    assert form.has_params() is False
    assert _form_layout(form).rowCount() == 0
    with pytest.raises(RuntimeError, match="not been populated"):
        form.read_params()


def test_analyze_form_user_edit_emits_params_changed(qapp):
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.populate(_TestParams(threshold=0.5, model="a"))
    form.params_changed.connect(lambda values: emitted.append(values))

    spin = form.findChild(TrimDoubleSpinBox)
    assert spin is not None
    spin.setValue(0.9)

    assert emitted[-1] == _TestParams(threshold=0.9, model="a")


def test_analyze_form_text_edit_emits_only_when_committed(qapp):
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.populate(_OptionalParams(t0=None))
    form.params_changed.connect(emitted.append)
    edit = form.findChild(QLineEdit)
    assert edit is not None

    edit.setText("1.5")
    assert emitted == []

    edit.editingFinished.emit()
    assert emitted == [_OptionalParams(t0=1.5)]


def _finish_optional_edit(driver: ComponentDriver, commit: str) -> None:
    """Leave by ordinary Tab and return for the next edit, without hiding fields."""
    if commit == "focus_loss":
        driver.press_key(Qt.Key.Key_Tab, "T0:")
        driver.press_key(Qt.Key.Key_Tab, "Commit")


@pytest.mark.parametrize("commit", ["enter", "focus_loss"])
@pytest.mark.parametrize(
    ("params_type", "valid_text", "valid_value"),
    [(_OptionalParams, "1.5", 1.5), (_OptionalIntParams, "3", 3)],
)
def test_optional_invalid_commit_blocks_params_and_recovers(
    qapp, tmp_path: Path, commit, params_type, valid_text, valid_value
):
    """Public params/validity evidence, with keyboard input through the UI driver."""
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)
    form = AnalyzeFormWidget()
    form.populate(params_type())
    layout.addWidget(form)
    layout.addWidget(QtWidgets.QPushButton("Commit"))
    emitted: list[object] = []
    form.params_changed.connect(emitted.append)
    window.resize(420, 180)
    window.show()
    qapp.processEvents()
    driver = ComponentDriver(window)
    try:
        driver.screenshot(tmp_path / "01-blank.png")
        assert form.read_params() == params_type()
        driver.enter_text("T0:", "-", commit=commit == "enter")
        _finish_optional_edit(driver, commit)
        driver.screenshot(tmp_path / "02-invalid.png")
        with pytest.raises(ValueError, match="T0"):
            form.read_params()
        assert form.is_valid() is False
        assert emitted == []

        driver.enter_text("T0:", valid_text, commit=commit == "enter")
        _finish_optional_edit(driver, commit)
        driver.screenshot(tmp_path / "03-valid.png")
        assert form.read_params() == params_type(valid_value)
        assert form.is_valid() is True
        assert emitted[-1] == params_type(valid_value)
        assert all(item == params_type(valid_value) for item in emitted)
        emitted.clear()

        driver.enter_text("T0:", "-", commit=commit == "enter")
        _finish_optional_edit(driver, commit)
        driver.screenshot(tmp_path / "04-invalid-after-valid.png")
        assert form.is_valid() is False
        assert emitted == []

        driver.enter_text("T0:", "", commit=commit == "enter")
        _finish_optional_edit(driver, commit)
        driver.screenshot(tmp_path / "05-blank-again.png")
        assert form.read_params() == params_type()
        assert form.is_valid() is True
        assert emitted and all(item == params_type() for item in emitted)
    finally:
        window.close()
        window.deleteLater()


# --- optional analyze fields (blank = None) --------------------------------


def test_analyze_form_optional_blank_reads_none(qapp):
    form = AnalyzeFormWidget()
    form.populate(_OptionalParams(t0=None))
    # an optional float renders the "(none)" QLineEdit, starting empty
    edit = form.findChild(QLineEdit)
    assert edit is not None and edit.text() == ""
    assert form.read_params() == _OptionalParams(t0=None)


def test_analyze_form_optional_typed_value_reads_float(qapp):
    form = AnalyzeFormWidget()
    form.populate(_OptionalParams(t0=None))
    edit = form.findChild(QLineEdit)
    assert edit is not None
    edit.setText("1.5")
    assert form.read_params() == _OptionalParams(t0=1.5)


def test_analyze_form_optional_round_trips_none_and_value(qapp):
    form = AnalyzeFormWidget()
    form.populate(_OptionalParams(t0=2.0))  # starts set
    assert form.read_params() == _OptionalParams(t0=2.0)
    form.populate_values(_OptionalParams(t0=None))  # back to None -> empty field
    assert form.read_params() == _OptionalParams(t0=None)
