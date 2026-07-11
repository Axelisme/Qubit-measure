from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from zcu_tools.gui.app.main.adapter import ParamMeta
from zcu_tools.gui.app.main.ui.analyze_form import AnalyzeFormWidget
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox


@dataclass
class _TestParams:
    threshold: Annotated[float, ParamMeta(label="Threshold", decimals=2)]
    model: Annotated[Literal["a", "b"], ParamMeta(label="Model")]


@dataclass
class _OptionalParams:
    t0: Annotated[float | None, ParamMeta(label="T0")] = None


def test_analyze_form_round_trips_values(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.populate(_TestParams(threshold=0.5, model="a"))

    result = form.read_params()

    assert result == _TestParams(threshold=0.5, model="a")
    assert form.is_valid() is True


def test_analyze_form_has_params_after_populate(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    assert form.has_params() is False

    form.populate(_TestParams(threshold=0.5, model="a"))

    assert form.has_params() is True


def test_analyze_form_populate_values_restores_state(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.populate(_TestParams(threshold=0.5, model="a"))
    form.populate_values(_TestParams(threshold=0.9, model="b"))
    assert form.read_params() == _TestParams(threshold=0.9, model="b")


def test_analyze_form_hydration_does_not_emit_params_changed(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.params_changed.connect(lambda values: emitted.append(values))

    form.populate(_TestParams(threshold=0.5, model="a"))
    form.populate_values(_TestParams(threshold=0.9, model="b"))

    assert emitted == []


def test_sync_same_class_preserves_widgets_and_only_hydrates(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.params_changed.connect(emitted.append)
    form.sync(_TestParams(threshold=0.5, model="a"))
    widgets = dict(form._widgets)
    row_count = form._form.rowCount()

    form.sync(_TestParams(threshold=0.9, model="b"))

    assert form._widgets == widgets
    assert all(form._widgets[name] is widget for name, widget in widgets.items())
    assert form._form.rowCount() == row_count
    assert form.read_params() == _TestParams(threshold=0.9, model="b")
    assert emitted == []


def test_sync_class_change_rebuilds_and_clear_fast_fails(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.sync(_TestParams(threshold=0.5, model="a"))
    old_widgets = tuple(form._widgets.values())

    form.sync(_OptionalParams(t0=1.0))

    assert all(widget not in form._widgets.values() for widget in old_widgets)
    assert form.read_params() == _OptionalParams(t0=1.0)

    form.clear()
    assert form.has_params() is False
    assert form._form.rowCount() == 0
    with pytest.raises(RuntimeError, match="not been populated"):
        form.read_params()


def test_field_metadata_is_resolved_once_per_class_build(qapp, monkeypatch):  # noqa: ARG001
    from zcu_tools.gui.app.main.ui import analyze_form as module

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


def test_analyze_form_user_edit_emits_params_changed(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    emitted: list[object] = []
    form.populate(_TestParams(threshold=0.5, model="a"))
    form.params_changed.connect(lambda values: emitted.append(values))

    spin = form.findChild(TrimDoubleSpinBox)
    assert spin is not None
    spin.setValue(0.9)

    assert emitted[-1] == _TestParams(threshold=0.9, model="a")


# --- optional analyze fields (blank = None) --------------------------------


def test_analyze_form_optional_blank_reads_none(qapp):  # noqa: ARG001
    from qtpy.QtWidgets import QLineEdit

    form = AnalyzeFormWidget()
    form.populate(_OptionalParams(t0=None))
    # an optional float renders the "(none)" QLineEdit, starting empty
    edit = form.findChild(QLineEdit)
    assert edit is not None and edit.text() == ""
    assert form.read_params() == _OptionalParams(t0=None)


def test_analyze_form_optional_typed_value_reads_float(qapp):  # noqa: ARG001
    from qtpy.QtWidgets import QLineEdit

    form = AnalyzeFormWidget()
    form.populate(_OptionalParams(t0=None))
    edit = form.findChild(QLineEdit)
    assert edit is not None
    edit.setText("1.5")
    assert form.read_params() == _OptionalParams(t0=1.5)


def test_analyze_form_optional_round_trips_none_and_value(qapp):  # noqa: ARG001
    form = AnalyzeFormWidget()
    form.populate(_OptionalParams(t0=2.0))  # starts set
    assert form.read_params() == _OptionalParams(t0=2.0)
    form.populate_values(_OptionalParams(t0=None))  # back to None -> empty field
    assert form.read_params() == _OptionalParams(t0=None)
