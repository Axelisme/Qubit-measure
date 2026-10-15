"""Direct rows show no echo, choice rows list only choices, and rows stay dense."""

import re
from collections.abc import Iterator
from typing import Literal, Self

import pytest
from pydantic import Field, model_validator
from PyQt6.QtGui import QAction
from qtpy.QtCore import QPoint, QRect
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QWidget,
)
from zcu_tools.gui.form import FormEnv, FormModel, FormState
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from tests.gui.widgets.form._field_errors import error_widget, field_error, row_tooltip
from tests.gui.widgets.form._text_input import commit_text


class Inner(FormModel):
    """Nested group with two numeric rows."""

    gain: float = 1.0
    phase: float = 0.0

    @model_validator(mode="after")
    def validate_gain(self) -> Self:
        """Leave a model-section error for a negative gain."""
        if self.gain < 0:
            raise ValueError("Gain must not be negative")
        return self


class PresentationForm(FormModel):
    """One row of every scalar kind, a whole sequence and a nested group."""

    amount: float = Field(default=2.0, description="Requested amount")
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


def _palette(dark: bool) -> QPalette:
    """Synthetic light or dark colours shared by presentation tests."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Base, QColor("#242424" if dark else "#fafafa"))
    palette.setColor(
        QPalette.ColorRole.Window, QColor("#353535" if dark else "#fafafa")
    )
    palette.setColor(QPalette.ColorRole.Text, QColor("#eeeeee" if dark else "#151515"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4488cc"))
    return palette


def alias_env() -> FormEnv:
    """Capture general.ext.seed=4, memory only."""
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-13T00:00:00Z", ext={"seed": 4.0}),
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
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
    commit_text(text, "general.ext.seed / 2")
    echo = child(view, QLabel, "form.value.amount")
    assert not echo.isHidden()
    assert echo.text() == "= 2.0"
    commit_text(text, "5000 + 0.125")
    assert echo.text() == "= 5000.125"
    assert not field_error(view, "amount")

    commit_text(text, "absent_name")
    assert echo.isHidden()
    assert echo.text() == ""
    assert field_error(view, "amount")

    mode(view, "amount").trigger()
    commit_text(child(view, QLineEdit, "form.input.amount"), "3")
    assert state.input.data["amount"] == 3.0
    assert echo.isHidden()
    assert not field_error(view, "amount")


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
    # The form-wide error stays hidden when valid.
    assert child(view, QLabel, "form.error").isHidden()
    for path in ROWS:
        assert not field_error(view, path), path

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
    # The inner section's fold title is its row's name label in the top column.
    title = child(view, QLabel, "form.fold.inner")
    for leaves, titles in ((TOP_LEAVES, [title]), (INNER_LEAVES, [])):
        lefts = {path: beside_label(view, path).left() for path in leaves}
        assert len(set(lefts.values())) == 1, lefts
        labels = [label_rect(view, text) for text in leaves.values()]
        labels += [rect(view, shown) for shown in titles]
        # Labels stay left-aligned in one column of one width.
        assert len({label.left() for label in labels}) == 1, labels
        assert len({label.width() for label in labels}) == 1, labels
        widest = max(
            w.sizeHint().width()
            for w in [*view.findChildren(QLabel), *titles]
            if w.text() in leaves.values() or w in titles
        )
        column = next(iter(lefts.values())) - labels[0].left()
        assert widest <= column <= widest + line / 2, (column, widest)
    # A nested level has its own column, indented inside its group.
    assert (
        beside_label(view, "inner.gain").left() != beside_label(view, "amount").left()
    )


def test_field_error_keeps_row_height_and_following_input_position(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
) -> None:
    view, state = editor
    lay_out(view, qapp)
    text = child(view, QLineEdit, "form.input.count")
    following = child(view, QComboBox, "form.input.unit")
    before = rect(view, text), rect(view, following), view.sizeHint()
    commit_text(text, "-")
    lay_out(view, qapp)
    assert field_error(view, "count")
    assert (rect(view, text), rect(view, following), view.sizeHint()) == before
    commit_text(text, "4")
    assert state.input.data["count"] == 4
    assert not field_error(view, "count")


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_error_tints_and_tooltips_restore_in_both_palettes(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
    dark: bool,
) -> None:
    view, state = editor
    palette = _palette(dark)
    view.setPalette(palette)
    lay_out(view, qapp)
    paths = ("amount", "unit", "enabled", "inner")
    original = {
        path: (
            error_widget(view, path).palette(),
            row_tooltip(view, path),
        )
        for path in paths
    }
    state.write(
        {
            ("amount",): "invalid",
            ("unit",): "missing",
            ("enabled",): "invalid",
            ("inner", "gain"): -1.0,
        },
        FormEnv(),
    )
    qapp.processEvents()
    errors = view.errors()
    assert errors is not None
    for path in paths:
        target = error_widget(view, path)
        message = errors.messages[(path,)]
        assert target.property("formError") is True
        assert row_tooltip(view, path).startswith(message)
        assert field_error(view, path) == message
        role = (
            QPalette.ColorRole.Base
            if path in ("amount", "unit")
            else QPalette.ColorRole.WindowText
        )
        colour = target.palette().color(role)
        before = original[path][0].color(role)
        assert colour.red() - colour.green() > before.red() - before.green()
        if path in ("enabled", "inner"):
            assert isinstance(target, QLabel)
            assert colour == QColor("#ff8a80" if dark else "#b3261e")
    assert row_tooltip(view, "amount").endswith(original["amount"][1])
    assert "Requested amount" in row_tooltip(view, "amount")
    assert "Right-click" in row_tooltip(view, "amount")

    state.write(
        {
            ("amount",): 2.0,
            ("unit",): "native",
            ("enabled",): True,
            ("inner", "gain"): 1.0,
        },
        FormEnv(),
    )
    qapp.processEvents()
    for path in paths:
        target = error_widget(view, path)
        assert target.property("formError") is False
        assert row_tooltip(view, path) == original[path][1]
        for role in (QPalette.ColorRole.Base, QPalette.ColorRole.WindowText):
            assert target.palette().color(role) == original[path][0].color(role)


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_expression_errors_keep_the_mode_tint_and_add_a_red_border(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
    dark: bool,
) -> None:
    view, state = editor
    palette = _palette(dark)
    view.setPalette(palette)
    mode(view, "amount").trigger()
    text = child(view, QLineEdit, "form.input.amount")
    commit_text(text, "2 + 3")
    qapp.processEvents()
    expression_base = text.palette().color(QPalette.ColorRole.Base)
    normal_style = text.styleSheet()
    assert expression_base != palette.color(QPalette.ColorRole.Base)

    commit_text(text, "absent_name")
    qapp.processEvents()
    assert field_error(view, "amount")
    assert text.palette().color(QPalette.ColorRole.Base) == expression_base
    border = re.search(r"border:\s*1px solid (#[0-9a-fA-F]+)", text.styleSheet())
    assert border is not None
    red = QColor(border.group(1))
    assert red == QColor("#ff8a80" if dark else "#b3261e")

    mode(view, "amount").trigger()
    qapp.processEvents()
    direct_error = text.palette().color(QPalette.ColorRole.Base)
    assert direct_error != expression_base
    assert direct_error.red() > direct_error.green()
    assert text.styleSheet() == normal_style

    mode(view, "amount").trigger()
    qapp.processEvents()
    assert text.palette().color(QPalette.ColorRole.Base) == expression_base
    assert "border" in text.styleSheet()
    commit_text(text, "3 + 4")
    qapp.processEvents()
    assert state.input.data["amount"] == {"expr": "3 + 4"}
    assert not field_error(view, "amount")
    assert text.palette().color(QPalette.ColorRole.Base) == expression_base
    assert text.styleSheet() == normal_style


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
@pytest.mark.parametrize("path", ["unit", "amount"], ids=["dropdown", "expression"])
def test_error_background_is_painted_without_changing_row_height(
    editor: tuple[FormEditor[PresentationForm], FormState[PresentationForm]],
    qapp: QApplication,
    dark: bool,
    path: str,
) -> None:
    view, state = editor
    palette = _palette(dark)
    palette.setColor(
        QPalette.ColorRole.Button, QColor("#303030" if dark else "#efefef")
    )
    view.setPalette(palette)
    if path == "amount":
        mode(view, path).trigger()
        commit_text(child(view, QLineEdit, f"form.input.{path}"), "5000 + 0.125")
    control = child(view, QWidget, f"form.input.{path}")
    lay_out(view, qapp)

    def interior() -> QColor:
        # Blank patch left of a combo's arrow and away from borders and text.
        pixels = control.grab().toImage()
        x, y = pixels.width() - 40, pixels.height() // 2
        patch = [
            pixels.pixelColor(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        ]
        return QColor(
            round(sum(c.red() for c in patch) / len(patch)),
            round(sum(c.green() for c in patch) / len(patch)),
            round(sum(c.blue() for c in patch) / len(patch)),
        )

    normal = interior()
    geometry = (
        control.height(),
        child(view, QLineEdit, "form.input.after").y(),
        view.sizeHint().height(),
    )
    if path == "amount":
        commit_text(child(view, QLineEdit, f"form.input.{path}"), "absent_name")
    else:
        state.write({(path,): "missing"}, FormEnv())
    lay_out(view, qapp)
    invalid = interior()
    assert field_error(view, path)
    if path == "amount":
        # Compare rendered colours, not just palette roles: a border-only Qt
        # stylesheet can otherwise paint a white interior over the mode tint.
        assert invalid == normal
    else:
        assert invalid.red() - invalid.green() > normal.red() - normal.green()
        assert invalid.red() - invalid.blue() > normal.red() - normal.blue()
    assert (
        control.height(),
        child(view, QLineEdit, "form.input.after").y(),
        view.sizeHint().height(),
    ) == geometry
