from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from qtpy import QtWidgets
from qtpy.QtWidgets import QLineEdit
from zcu_tools.gui.app.measure.adapter import ParamMeta
from zcu_tools.gui.app.measure.ui.analyze_form import AnalyzeFormWidget
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox


@dataclass
class _TestParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)]
    model: Annotated[Literal["a", "b"], ParamMeta(label="Model")]


@dataclass
class _OptionalParams:
    t0: Annotated[float | None, ParamMeta(label="T0")] = None


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


def test_field_metadata_is_resolved_once_per_class_build(qapp, monkeypatch):
    from zcu_tools.gui.app.measure.ui import analyze_form as module

    real_hints = module.get_type_hints
    real_resolve = module._resolve_field_info
    hint_calls = 0
    resolve_calls = 0

    def counted_hints(*args, **kwargs):
        nonlocal hint_calls
        hint_calls += 1
        return real_hints(*args, **kwargs)

    def counted_resolve(*args, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(module, "get_type_hints", counted_hints)
    monkeypatch.setattr(module, "_resolve_field_info", counted_resolve)
    form = AnalyzeFormWidget()

    form.sync(_TestParams(threshold=0.5, model="a"))
    form.read_params()
    form.sync(_TestParams(threshold=0.9, model="b"))
    form.read_params()

    assert hint_calls == 1
    assert resolve_calls == 2


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
