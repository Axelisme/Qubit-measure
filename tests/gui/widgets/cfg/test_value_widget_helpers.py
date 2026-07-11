from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)
from zcu_tools.gui.widgets.cfg.fields import (
    connect_committed_value_widget,
    connect_value_widget,
    read_value_widget,
    write_value_widget,
)
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox


@pytest.mark.parametrize(
    ("factory", "value", "type_", "expected"),
    [
        (lambda: QCheckBox(), True, bool, True),
        (lambda: QSpinBox(), 7, int, 7),
        (lambda: TrimDoubleSpinBox(), 2.5, float, 2.5),
        (lambda: QLineEdit(), "text", str, "text"),
    ],
)
def test_write_and_connect_value_widget(
    qapp,
    factory,
    value,
    type_,
    expected,  # noqa: ARG001
) -> None:
    widget = factory()
    callback = MagicMock()
    connect_value_widget(widget, callback)

    write_value_widget(widget, value)

    assert read_value_widget(widget, type_) == expected
    callback.assert_called_once()


def test_write_and_connect_combo_widget(qapp) -> None:  # noqa: ARG001
    widget = QComboBox()
    widget.addItems(["a", "b"])
    widget.setCurrentIndex(-1)
    callback = MagicMock()
    connect_value_widget(widget, callback)

    write_value_widget(widget, "b")

    assert read_value_widget(widget, str) == "b"
    callback.assert_called_once()


def test_committed_value_widget_defers_line_edit_callback(qapp) -> None:  # noqa: ARG001
    widget = QLineEdit()
    callback = MagicMock()
    connect_committed_value_widget(widget, callback)

    widget.setText("partial")
    callback.assert_not_called()

    widget.editingFinished.emit()
    callback.assert_called_once()


@pytest.mark.parametrize(
    ("factory", "change"),
    [
        (lambda: QCheckBox(), lambda widget: widget.setChecked(True)),
        (lambda: QSpinBox(), lambda widget: widget.setValue(1)),
        (
            lambda: QComboBox(),
            lambda widget: (widget.addItems(["a", "b"]), widget.setCurrentIndex(1)),
        ),
    ],
)
def test_committed_value_widget_keeps_non_text_changes_immediate(
    qapp, factory, change
) -> None:  # noqa: ARG001
    widget = factory()
    callback = MagicMock()
    connect_committed_value_widget(widget, callback)

    change(widget)

    callback.assert_called()


def test_value_widget_helpers_reject_unsupported_widget(qapp) -> None:  # noqa: ARG001
    widget = QWidget()
    with pytest.raises(TypeError, match="Unsupported value widget"):
        write_value_widget(widget, 1)
    with pytest.raises(TypeError, match="Unsupported value widget"):
        connect_value_widget(widget, lambda: None)
