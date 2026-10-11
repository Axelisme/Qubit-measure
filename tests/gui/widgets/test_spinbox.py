"""Finite numeric editor contracts through public Qt input and value seams."""

from sys import float_info

import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent, QValidator
from qtpy.QtWidgets import QApplication, QLineEdit
from zcu_tools.gui.widgets.spinbox import FiniteFloatSpinBox


@pytest.mark.parametrize(
    "value",
    [5e-324, -5e-324, 1.2345678901234567, float_info.max, -float_info.max, -0.0],
)
def test_finite_float_roundtrips_without_quantization(qapp, value):
    editor = FiniteFloatSpinBox()
    try:
        editor.setValue(value)
        assert editor.value() == value
        assert float(editor.text()) == value
        assert editor.text() == repr(value)
    finally:
        editor.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_programmatic_input_is_rejected(qapp, value):
    editor = FiniteFloatSpinBox()
    try:
        editor.setValue(0.25)
        with pytest.raises(ValueError, match="finite"):
            editor.setValue(value)
        assert editor.value() == 0.25
    finally:
        editor.close()


@pytest.mark.parametrize("text", ["nan", "inf", "1e309", "abc"])
def test_nonfinite_or_nonnumeric_text_is_rejected(qapp, text):
    editor = FiniteFloatSpinBox()
    try:
        assert editor.validate(text, len(text))[0] == QValidator.State.Invalid
    finally:
        editor.close()


def test_scientific_text_edit_emits_exact_value_and_incomplete_text_is_uncommitted(
    qapp,
):
    editor = FiniteFloatSpinBox()
    values: list[float] = []
    editor.valueChanged.connect(values.append)
    line = editor.findChild(QLineEdit)
    assert line is not None
    line.selectAll()
    _type_text(line, "5e-324")
    assert editor.value() == 5e-324
    assert values[-1] == 5e-324
    line.selectAll()
    _type_text(line, "-")
    assert editor.value() == 5e-324
    assert editor.fixup("-") == "5e-324"
    editor.close()


def _type_text(line: QLineEdit, text: str) -> None:
    for char in text:
        for event_type in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            QApplication.sendEvent(
                line, QKeyEvent(event_type, 0, Qt.KeyboardModifier.NoModifier, char)
            )


@pytest.mark.parametrize(
    "initial,steps,expected",
    [
        (0.25, 2, 2.25),
        (0.25, -2, -1.75),
        (float_info.max, 1, float_info.max),
        (-float_info.max, -1, -float_info.max),
        (0.0, 10**400, float_info.max),
        (0.0, -(10**400), -float_info.max),
    ],
)
def test_stepping_retains_unit_step_and_finite_extremes(qapp, initial, steps, expected):
    editor = FiniteFloatSpinBox()
    try:
        editor.setValue(initial)
        editor.stepBy(steps)
        assert editor.value() == expected
    finally:
        editor.close()
