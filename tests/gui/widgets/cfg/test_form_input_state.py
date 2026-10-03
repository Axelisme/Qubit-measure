"""Cfg form input state is owned by the model and survives widget recreation.

Partial, invalid, or non-canonical text typed into scalar and sweep widgets is
kept in the model, published in form snapshots, and restored when the form is
rebuilt, while canonical values stay visible beside the raw input. Expression
resolution, mode switches, and value-source tokens update the same binding state.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QLabel, QLineEdit
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.widgets.cfg import CfgFormWidget

from tests.gui.widgets.cfg._form_support import (
    attach_draft,
    scalar_field,
    section_schema,
)


def test_sweep_widget_publishes_incomplete_edge_in_form_snapshot(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"axis": SweepSpec()},
        {"axis": SweepValue(0.0, 1.0, 5)},
    )
    widget = CfgFormWidget()
    draft = attach_draft(widget, schema, ctrl)
    try:
        entry = widget.findChild(QLineEdit)
        assert entry is not None
        entry.setText("1e")
        value = widget.read_values().fields["axis"]
        assert isinstance(value, SweepValue)
        assert isinstance(value.start, DirectValue)
        assert value.start.raw == "1e"
        assert value.start.value is None
        assert value.start.error is not None
        assert not draft.is_valid()

        entry.setText("0.20")
        recovered = widget.read_values().fields["axis"]
        assert isinstance(recovered, SweepValue)
        assert recovered.start == DirectValue(0.2, raw="0.20")
        assert recovered.step == pytest.approx(0.2)
        assert draft.is_valid()
    finally:
        widget.detach()
        widget.close()
        widget.deleteLater()
        draft.teardown()


def test_complex_direct_widget_keeps_partial_input_in_model(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields.common import ScalarWidget

    field = scalar_field(ctrl, ScalarSpec("Center", complex), DirectValue(1 + 2j))
    widget = ScalarWidget(field)
    try:
        entry = widget.findChild(QLineEdit)
        assert entry is not None
        entry.setText("2-")
        assert not field.is_valid()
        value = field.get_value()
        assert isinstance(value, DirectValue)
        assert value.raw == "2-"
        assert value.value is None
        entry.setText("2-3j")
        assert field.is_valid()
        assert field.get_value() == DirectValue(2 - 3j, raw="2-3j")
    finally:
        widget.teardown()
        widget.close()
        widget.deleteLater()
        field.teardown()


@pytest.mark.parametrize("optional", [False, True])
def test_numeric_edit_is_owned_by_model_and_survives_widget_recreation(
    qapp, ctrl, optional
):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields.common import ScalarWidget

    field = scalar_field(
        ctrl, ScalarSpec("Mixer", float, optional=optional), DirectValue(5.0)
    )
    widget = ScalarWidget(field)
    try:
        entry = widget.findChild(QLineEdit)
        assert entry is not None
        entry.setText("1e")
        value = field.get_value()
        assert isinstance(value, DirectValue)
        assert value.raw == "1e"
        assert value.value is None
        assert value.error
        assert not field.is_valid()
    finally:
        widget.teardown()
        widget.close()
        widget.deleteLater()

    replacement = ScalarWidget(field)
    try:
        entry = replacement.findChild(QLineEdit)
        assert entry is not None
        assert entry.text() == "1e"
        entry.setText("1e2")
        assert field.get_value() == DirectValue(100.0, raw="1e2")
        assert field.is_valid()
    finally:
        replacement.teardown()
        replacement.close()
        replacement.deleteLater()
        field.teardown()


@pytest.mark.parametrize(
    ("centered", "part"),
    [
        (False, "expts"),
        (False, "step"),
        (True, "span"),
        (True, "expts"),
        (True, "step"),
    ],
)
@pytest.mark.parametrize("text", ["", "1e"])
def test_sweep_text_survives_form_recreation(qapp, ctrl, centered, part, text):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    spec = CenteredSweepSpec() if centered else SweepSpec()
    value = CenteredSweepValue(0.0, 1.0, 5) if centered else SweepValue(0.0, 1.0, 5)
    form = CfgFormWidget()
    root = attach_draft(form, section_schema({"axis": spec}, {"axis": value}), ctrl)
    entry = form.findChild(QLineEdit, part)
    assert entry is not None
    entry.setText(text)
    snapshot = form.read_schema()
    saved = getattr(snapshot.value.fields["axis"], part)
    assert isinstance(saved, DirectValue)
    assert saved.raw == text
    assert saved.value is None
    assert not root.is_valid()
    form.detach()

    restored = CfgFormWidget()
    restored_root = attach_draft(restored, snapshot, ctrl)
    restored_input = restored.findChild(QLineEdit, part)
    assert restored_input is not None
    assert restored_input.text() == text
    assert not restored_root.is_valid()
    restored_input.setText("5" if part == "expts" else "0.25")
    assert restored_root.is_valid()
    assert saved.raw == text and saved.value is None
    restored.detach()
    root.teardown()
    restored_root.teardown()


def test_sweep_step_shows_canonical_value_without_replacing_raw(qapp, ctrl):
    from qtpy.QtWidgets import QLabel, QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    form = CfgFormWidget()
    root = attach_draft(
        form,
        section_schema({"axis": SweepSpec()}, {"axis": SweepValue(0.0, 1.0, 5)}),
        ctrl,
    )
    entry = form.findChild(QLineEdit, "step")
    assert entry is not None
    entry.setText("0.3")
    value = form.read_values().fields["axis"]
    assert isinstance(value, SweepValue)
    assert value.expts == 4
    assert isinstance(value.step, DirectValue)
    assert value.step.raw == entry.text() == "0.3"
    assert value.step.value == pytest.approx(1 / 3)
    labels = [label.text() for label in form.findChildren(QLabel)]
    assert f"step = {value.step.value}" in labels
    form.detach()
    root.teardown()


def test_scalar_widget_eval_mode_shows_resolved_ghost(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("r_f"),
    )

    widget = ScalarWidget(field)
    try:
        ghost = widget.findChild(QLabel)
        editor = widget.findChild(QLineEdit)
        assert ghost is not None and ghost.text() == "= 6000.0"
        assert editor is not None and editor.text() == "r_f"

        editor.setText("r_f + 1")

        value = field.get_value()
        assert isinstance(value, EvalValue)
        assert value.expr == "r_f + 1" and value.resolved == 6001.0
        assert field.is_valid()
        assert ghost.text() == "= 6001.0"
    finally:
        widget.teardown()
        widget.close()
        field.teardown()


def test_scalar_widget_eval_mode_recovers_from_unresolved_expression(qapp, ctrl):
    from qtpy.QtGui import QColor, QPalette
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("missing"),
    )

    widget = ScalarWidget(field)
    try:
        widget.show()
        qapp.processEvents()
        ghost = widget.findChild(QLabel)
        editor = widget.findChild(QLineEdit)
        assert ghost is not None and ghost.text() == "= ?"
        assert "missing" in ghost.toolTip()
        assert ghost.palette().color(QPalette.ColorRole.WindowText) == QColor("red")
        assert not field.is_valid()
        assert editor is not None

        editor.setText("2 + 3")

        value = field.get_value()
        assert isinstance(value, EvalValue)
        assert value.expr == "2 + 3" and value.resolved == 5.0
        assert value.error is None and field.is_valid()
        assert ghost.text() == "= 5.0"
        assert ghost.toolTip() == ""
    finally:
        widget.teardown()
        widget.close()
        field.teardown()


def test_measure_cfg_form_value_source_resolves_on_space_in_eval_input(qapp, ctrl):
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QKeyEvent
    from zcu_tools.gui.app.measure.ui.cfg_binding import (
        make_value_source_input_enhancer,
    )
    from zcu_tools.gui.session.value_lookup import ValueInfo
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        0.125,
    )
    schema = section_schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue("r_f")},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(text_input_enhancer=make_value_source_input_enhancer(ctrl))
    try:
        form.attach(draft)
        editor = form.findChild(QLineEdit)
        assert editor is not None
        editor.setText("@{device.flux.value}")
        editor.setCursorPosition(len(editor.text()))

        qapp.sendEvent(
            editor,
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
                " ",
            ),
        )

        value = form.read_values().fields["freq"]
        assert isinstance(value, EvalValue)
        assert value.expr == "0.125" and value.resolved == 0.125
        assert editor.text() == "0.125"
        ctrl.read_value_source.assert_called_once_with("device.flux.value")
    finally:
        form.detach()
        form.close()
        draft.close()


def test_scalar_widget_context_menu_uses_resolved_direct_value(qapp, ctrl, monkeypatch):
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QMenu
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("r_f"),
    )

    def choose(menu: QMenu, _position: QPoint):
        return next(
            action for action in menu.actions() if action.text() == "Use direct value"
        )

    monkeypatch.setattr(QMenu, "exec_", choose)
    widget = ScalarWidget(field)
    try:
        editor = widget.findChild(QLineEdit)
        assert editor is not None and editor.text() == "r_f"

        editor.customContextMenuRequested.emit(QPoint())

        value = field.get_value()
        assert isinstance(value, DirectValue) and value.value == 6000.0
        direct_editor = widget.findChild(QLineEdit)
        assert direct_editor is not None
        assert float(direct_editor.text()) == 6000.0
    finally:
        widget.teardown()
        widget.close()
        field.teardown()


def test_sweep_widget_start_supports_eval_mode(qapp, ctrl):
    from qtpy.QtWidgets import QWidget
    from zcu_tools.resources.context import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=EvalValue("r_f - 1"), stop=6005.0, expts=11)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        start = form.findChild(QWidget, "start")
        assert start is not None
        editor = start.findChild(QLineEdit)
        assert editor is not None and editor.text() == "r_f - 1"

        editor.setText("r_f + 1")

        value = form.read_values().fields["f"]
        assert isinstance(value, SweepValue)
        assert isinstance(value.start, EvalValue)
        assert value.start.expr == "r_f + 1" and value.start.resolved == 6001.0
        assert value.stop == 6005.0
        assert form.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()
