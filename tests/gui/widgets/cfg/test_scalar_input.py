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
