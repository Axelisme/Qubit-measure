"""Published scalar presentation and input submission through public Qt seams."""

from collections.abc import Iterator

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QMenu
from zcu_tools.gui.cfg import DirectValue, EvalValue, ScalarSpec
from zcu_tools.gui.widgets.cfg.fields.common import ScalarInputWidget


@pytest.fixture
def widgets(qapp: QApplication) -> Iterator[list[ScalarInputWidget]]:
    owned: list[ScalarInputWidget] = []
    yield owned
    for widget in owned:
        widget.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize("type_,text", [(float, "-"), (complex, "1+1j")])
def test_numeric_text_is_submitted_unparsed(widgets, type_, text):
    submitted: list[DirectValue | EvalValue] = []
    widget = ScalarInputWidget(
        ScalarSpec("Value", type_),
        DirectValue(2),
        options=None,
        submit=submitted.append,
    )
    widgets.append(widget)
    line = widget.findChild(QLineEdit)
    assert line is not None
    line.setText(text)
    assert submitted == [DirectValue(None, raw=text)]
    assert line.text() == text

    widget.display(DirectValue(None, raw=text, error="Invalid input"), options=None)
    assert line.text() == text
    assert len(submitted) == 1
    widget.display(DirectValue(3), options=None)
    assert line.text() == "3"
    assert len(submitted) == 1


def test_expression_displays_only_published_resolution(widgets):
    submitted: list[DirectValue | EvalValue] = []
    widget = ScalarInputWidget(
        ScalarSpec("Value", float),
        EvalValue("2", resolved=2),
        options=None,
        submit=submitted.append,
    )
    widgets.append(widget)
    line = widget.findChild(QLineEdit)
    ghost = widget.findChild(QLabel)
    assert line is not None and ghost is not None
    line.setText("sin(pi / 2)")
    assert submitted == [EvalValue("sin(pi / 2)")]
    widget.display(EvalValue("sin(pi / 2)", resolved=1), options=None)
    assert ghost.text() == "= 1.0"
    widget.display(EvalValue("missing", error="Unknown name"), options=None)
    assert line.text() == "missing"
    assert ghost.text() == "= ?"
    assert ghost.toolTip() == "Unknown name"
    assert len(submitted) == 1


@pytest.mark.parametrize(
    "type_,options", [(int, (1, 2)), (str, ("a", "b")), (bool, (True, False))]
)
def test_choice_submits_typed_value_and_refresh_does_not_submit(
    widgets, type_, options
):
    submitted: list[DirectValue | EvalValue] = []
    widget = ScalarInputWidget(
        ScalarSpec("Value", type_),
        DirectValue(options[0]),
        options=options,
        submit=submitted.append,
    )
    widgets.append(widget)
    combo = widget.findChild(QComboBox)
    assert combo is not None
    combo.setCurrentIndex(1)
    assert submitted == [DirectValue(options[1])]
    widget.display(DirectValue(options[0]), options=options[:1])
    replacement = widget.findChild(QComboBox)
    assert replacement is not None
    assert replacement.count() == 1
    assert replacement.currentText() == str(options[0])
    assert len(submitted) == 1


def test_context_menu_switches_modes_using_published_value(widgets, monkeypatch):
    submitted: list[DirectValue | EvalValue] = []
    widget = ScalarInputWidget(
        ScalarSpec("Value", float),
        DirectValue(2),
        options=None,
        submit=submitted.append,
    )
    widgets.append(widget)
    selected = "Use expression"

    def choose(menu, _position):
        return next(action for action in menu.actions() if action.text() == selected)

    monkeypatch.setattr(QMenu, "exec_", choose)
    line = widget.findChild(QLineEdit)
    assert line is not None
    line.customContextMenuRequested.emit(QPoint())
    assert submitted == [EvalValue("2")]
    widget.display(EvalValue("sin(pi / 2)", resolved=1), options=None)
    selected = "Use direct value"
    line = widget.findChild(QLineEdit)
    assert line is not None
    line.customContextMenuRequested.emit(QPoint())
    assert submitted[-1] == DirectValue(1)


def test_read_only_scalar_remains_disabled_after_mode_publication(widgets):
    submitted: list[DirectValue | EvalValue] = []
    widget = ScalarInputWidget(
        ScalarSpec("Value", float, editable=False),
        DirectValue(2),
        options=None,
        submit=submitted.append,
    )
    widgets.append(widget)
    line = widget.findChild(QLineEdit)
    assert line is not None and not line.isEnabled()
    widget.display(EvalValue("2", resolved=2), options=None)
    line = widget.findChild(QLineEdit)
    assert line is not None and not line.isEnabled()
    assert submitted == []


@pytest.mark.parametrize(
    "value,preview,tone",
    [
        (EvalValue("qubit.freq", resolved=600.0), "= 600.0", "gray"),
        (EvalValue("missing", error="Unknown name"), "= ?", "red"),
    ],
)
def test_expression_preview_wraps_below_full_width_input(widgets, value, preview, tone):
    from qtpy.QtWidgets import QBoxLayout, QSizePolicy

    widget = ScalarInputWidget(
        ScalarSpec("Value", float), value, options=None, submit=lambda value: None
    )
    widgets.append(widget)
    line = widget.findChild(QLineEdit)
    ghost = widget.findChild(QLabel)
    assert line is not None and ghost is not None
    layout = widget.layout()
    assert isinstance(layout, QBoxLayout)
    assert layout.direction() == QBoxLayout.Direction.TopToBottom
    assert layout.indexOf(ghost) > layout.indexOf(line)
    assert ghost.wordWrap()
    assert ghost.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    assert ghost.font().pointSizeF() < line.font().pointSizeF()
    assert ghost.text() == preview
    assert tone in ghost.styleSheet()
    assert ghost.toolTip() == (value.error or "")

    widget.display(DirectValue(600), options=None)
    direct_layout = widget.layout()
    assert isinstance(direct_layout, QBoxLayout)
    assert direct_layout.direction() == QBoxLayout.Direction.TopToBottom
    direct_line = widget.findChild(QLineEdit)
    assert direct_line is not None
    assert direct_layout.indexOf(direct_line) == 0
    direct_item, expression_item = direct_layout.itemAt(0), layout.itemAt(0)
    assert direct_item is not None and expression_item is not None
    assert direct_item.alignment() == expression_item.alignment()


def test_expression_blur_returns_cursor_to_start_without_losing_tooltip(widgets, qapp):
    expression = "$qubit.resonator.freq - 4 * $qubit.resonator.kappa"
    widget = ScalarInputWidget(
        ScalarSpec("Value", float),
        EvalValue(expression, resolved=5996),
        options=None,
        submit=lambda value: None,
    )
    widgets.append(widget)
    widget.show()
    widget.activateWindow()
    line = widget.findChild(QLineEdit)
    assert line is not None
    line.setFocus()
    qapp.processEvents()
    assert line.hasFocus()
    line.setCursorPosition(len(expression))
    widget.display(EvalValue(expression, resolved=5996), options=None)
    assert line.cursorPosition() == len(expression)
    line.clearFocus()
    qapp.processEvents()
    assert not line.hasFocus()
    assert line.cursorPosition() == 0
    assert line.text() == expression
    assert line.toolTip() == expression


def test_unfocused_expression_publication_starts_at_beginning(widgets):
    widget = ScalarInputWidget(
        ScalarSpec("Value", float),
        EvalValue("$qubit.freq", resolved=600),
        options=None,
        submit=lambda value: None,
    )
    widgets.append(widget)
    line = widget.findChild(QLineEdit)
    assert line is not None
    assert not line.hasFocus()
    assert line.cursorPosition() == 0
    widget.display(EvalValue("$qubit.freq * 2", resolved=1200), options=None)
    assert line.cursorPosition() == 0
    assert line.toolTip() == "$qubit.freq * 2"
