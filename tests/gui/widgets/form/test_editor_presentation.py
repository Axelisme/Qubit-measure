"""Direct rows show no echo, choice rows list only choices, and rows stay dense."""

from collections.abc import Iterator
from typing import Literal

import pytest
from pydantic import Field
from PyQt6.QtGui import QAction
from qtpy.QtCore import QPoint, QRect
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)
from zcu_tools.gui.form import FormEnv, FormModel, FormState
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class Inner(FormModel):
    """Nested group with two numeric rows."""

    gain: float = 1.0
    phase: float = 0.0


class PresentationForm(FormModel):
    """One row of every scalar kind, a whole sequence and a nested group."""

    amount: float = 2.0
    count: int = 3
    unit: Literal["native", "mA"] = "native"
    nqz: Literal[1, 2] = 2
    enabled: bool = True
    samples: tuple[float, ...] = (1.0,)
    inner: Inner = Field(default_factory=Inner)
    after: float = 5.0


ROWS = (
    "amount",
    "count",
    "unit",
    "nqz",
    "enabled",
    "samples",
    "inner",
    "inner.gain",
    "inner.phase",
    "after",
)
LEAVES = tuple(path for path in ROWS if path != "inner")
LABELS = {"amount": "Amount", "unit": "Unit", "inner.gain": "Gain", "after": "After"}
# Leaf labels of each model level; the top level also holds the inner group row.
TOP_LEAVES = {
    "amount": "Amount",
    "count": "Count",
    "unit": "Unit",
    "nqz": "Nqz",
    "enabled": "Enabled",
    "samples": "Samples",
    "after": "After",
}
INNER_LEAVES = {"inner.gain": "Gain", "inner.phase": "Phase"}


def alias_env() -> FormEnv:
    """Capture general.ext.seed=4, memory only."""
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-13T00:00:00Z", ext={"seed": 4.0}),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=SavedPoint(entry_id="entry", point="synthetic"),
        roles=None,
        storage=None,
    )
    return FormEnv(parameters=parameters)


@pytest.fixture
def editor(
    qapp: QApplication,
) -> Iterator[tuple[FormEditor[PresentationForm], FormState[PresentationForm]]]:
    """Own one editor over default input; close and delete it afterwards."""
    env = alias_env()
    state = FormState(PresentationForm.default_input(env))
    view = FormEditor(state, env)
    yield view, state
    view.close()
    view.deleteLater()
    qapp.processEvents()


def child[W: QWidget](view: QWidget, kind: type[W], name: str) -> W:
    """Find a documented public child by its complete object name."""
    widget = view.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


def mode(view: QWidget, path: str) -> QAction:
    """Find the documented Expression action of one row."""
    action = view.findChild(QAction, f"form.mode.{path}")
    assert isinstance(action, QAction), path
    return action


def visible_combos(view: QWidget, path: str) -> list[str]:
    """Object names of the not-hidden combo boxes that belong to one row."""
    return sorted(
        combo.objectName()
        for combo in view.findChildren(QComboBox)
        if combo.objectName().endswith(f".{path}") and not combo.isHidden()
    )


def rect(view: QWidget, widget: QWidget) -> QRect:
    """Widget geometry in view coordinates."""
    return QRect(widget.mapTo(view, QPoint(0, 0)), widget.size())


def label_rect(view: QWidget, text: str) -> QRect:
    """Geometry of the unique row label showing text."""
    (label,) = [w for w in view.findChildren(QLabel) if w.text() == text]
    return rect(view, label)


def beside_label(view: QWidget, path: str) -> QRect:
    """Geometry of a leaf row's first control, its input."""
    return rect(view, child(view, QWidget, f"form.input.{path}"))


def lay_out(view: QWidget, qapp: QApplication, extra_height: int = 0) -> None:
    """Show view at 600 px wide and its own sizeHint height plus extra_height."""
    view.resize(600, view.sizeHint().height() + extra_height)
    view.show()
    qapp.processEvents()


def test_direct_rows_show_no_value_echo(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
) -> None:
    view, _state = editor
    for path in ROWS:
        echo = child(view, QLabel, f"form.value.{path}")
        assert echo.isHidden(), path
        assert echo.text() == "", path


def test_expression_rows_show_the_resolved_value_or_the_error(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
) -> None:
    view, state = editor
    mode(view, "amount").trigger()
    text = child(view, QLineEdit, "form.input.amount")
    text.setText("general.ext.seed / 2")
    echo = child(view, QLabel, "form.value.amount")
    error = child(view, QLabel, "form.error.amount")
    assert not echo.isHidden()
    assert float(echo.text()) == 2.0
    assert error.isHidden()

    text.setText("absent_name")
    assert echo.isHidden()
    assert echo.text() == ""
    assert not error.isHidden()
    assert error.text()

    mode(view, "amount").trigger()
    child(view, QLineEdit, "form.input.amount").setText("3")
    assert state.input.data["amount"] == 3.0
    assert echo.isHidden()
    assert error.isHidden()


def test_choice_rows_show_one_dropdown_of_only_the_choices(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
) -> None:
    view, _state = editor
    for path, options in (("unit", ["native", "mA"]), ("nqz", [1, 2])):
        assert visible_combos(view, path) == [f"form.input.{path}"]
        direct = child(view, QComboBox, f"form.input.{path}")
        values = [direct.itemData(index) for index in range(direct.count())]
        assert values == options


def test_rows_and_groups_are_dense_and_do_not_overlap(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
) -> None:
    view, _state = editor
    lay_out(view, qapp)
    inputs = {
        path: rect(view, child(view, QWidget, f"form.input.{path}")) for path in LEAVES
    }
    line = inputs["amount"].height()
    gaps = {
        (above, below): inputs[below].top()
        - (inputs[above].top() + inputs[above].height())
        for above, below in zip(LEAVES, LEAVES[1:], strict=False)
    }
    for pair, gap in gaps.items():
        assert gap > 0, pair
    group_pairs = {("samples", "inner.gain"), ("inner.phase", "after")}
    for pair, gap in gaps.items():
        if pair not in group_pairs:
            assert gap < line / 2, (pair, gap, line)
    # Entering a group also passes its header row; leaving it passes its frame.
    assert gaps["samples", "inner.gain"] < 2 * line, gaps
    assert gaps["inner.phase", "after"] < line, gaps
    for path, text in LABELS.items():
        label = label_rect(view, text)
        beside = beside_label(view, path)
        assert not label.intersects(beside), path
        assert not label.intersects(inputs[path]), path
        assert beside.left() - (label.left() + label.width()) < line, path
    for path in LEAVES:
        control = child(view, QWidget, f"form.input.{path}")
        assert control.height() >= control.minimumSizeHint().height(), path
    # Empty error labels take no space.
    assert child(view, QLabel, "form.error").isHidden()
    for path in ROWS:
        assert child(view, QLabel, f"form.error.{path}").isHidden(), path

    # Extra height goes below the last row, not between rows or into groups.
    lay_out(view, qapp, extra_height=300)
    for path in LEAVES:
        assert rect(view, child(view, QWidget, f"form.input.{path}")) == inputs[path]


def test_each_model_level_aligns_its_controls_on_one_label_column(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
) -> None:
    view, _state = editor
    lay_out(view, qapp)
    line = child(view, QLineEdit, "form.input.amount").height()
    (fold,) = [b for b in view.findChildren(QPushButton) if b.text() == "Collapse"]
    for leaves, group_row in ((TOP_LEAVES, rect(view, fold)), (INNER_LEAVES, None)):
        lefts = {path: beside_label(view, path).left() for path in leaves}
        if group_row is not None:
            lefts["inner"] = group_row.left()
        assert len(set(lefts.values())) == 1, lefts
        labels = [label_rect(view, text) for text in leaves.values()]
        # Labels stay left-aligned in the column.
        assert len({label.left() for label in labels}) == 1, labels
        widest = max(
            w.sizeHint().width()
            for w in view.findChildren(QLabel)
            if w.text() in leaves.values()
        )
        column = next(iter(lefts.values())) - labels[0].left()
        assert widest <= column <= widest + line / 2, (column, widest)
    # A nested level has its own column, indented inside its group.
    assert (
        beside_label(view, "inner.gain").left() != beside_label(view, "amount").left()
    )


def test_field_error_appears_below_its_input_without_overlapping_the_next_row(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
) -> None:
    view, state = editor
    lay_out(view, qapp)
    error = child(view, QLabel, "form.error.count")
    assert error.isHidden()
    text = child(view, QLineEdit, "form.input.count")
    text.setText("-")
    lay_out(view, qapp)
    assert not error.isHidden()
    assert error.text()
    shown = rect(view, error)
    field = rect(view, text)
    following = rect(view, child(view, QComboBox, "form.input.unit"))
    assert shown.top() >= field.top() + field.height()
    assert following.top() >= shown.top() + shown.height()
    text.setText("4")
    assert state.input.data["count"] == 4
    assert error.isHidden()
