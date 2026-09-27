"""Cfg form input state is owned by the model and survives widget recreation.

Partial, invalid, or non-canonical text typed into scalar and sweep widgets is
kept in the model, published in form snapshots, and restored when the form is
rebuilt, while canonical values stay visible beside the raw input.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    DirectValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

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
