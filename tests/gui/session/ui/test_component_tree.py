"""The component frontend submits owner commands and exposes failures."""

from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ConfigDict
from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import (
    Profiled,
    register_fakes,
    registry_state,
    restore_registry,
)

FIELD, VALUE, UNIT, SOURCE = range(4)


class RequiredComponent(ComponentSchema):
    reference: str


class OpenComponent(ComponentSchema):
    model_config = ConfigDict(extra="allow")
    rate: float | None = None


def _service() -> ContextService:
    return ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )


@pytest.fixture
def owner(qapp) -> Generator[ContextService]:
    """Real owner with fake kinds; the entry is installed by each widget fixture."""
    before = registry_state()
    try:
        register_fakes()
        component_registry.register("test/required", RequiredComponent)
        component_registry.register("test/profiled", Profiled)
        component_registry.register("test/open", OpenComponent)
        installed = registry_state()
        yield _service()
        assert registry_state() == installed
    finally:
        restore_registry(before)


def _close(qapp, widget: QWidget) -> None:
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def component_widget(
    qapp, tmp_path: Path, owner: ContextService
) -> Generator[tuple[ComponentTreeWidget, ContextControlFacet]]:
    owner.install_entry(
        ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
    )
    control = ContextControlFacet(context=owner)
    widget = ComponentTreeWidget(control)
    widget.resize(900, 700)
    widget.show()
    qapp.processEvents()
    try:
        yield widget, control
    finally:
        _close(qapp, widget)


def _tree(widget: ComponentTreeWidget) -> QTreeWidget:
    tree = widget.findChild(QTreeWidget, "component_tree")
    assert tree is not None
    return tree


def _row(widget: ComponentTreeWidget, *path: str) -> QTreeWidgetItem:
    """Find a row by component name and field segments as shown in the Field column."""
    tree = _tree(widget)
    rows = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    found: QTreeWidgetItem | None = None
    for segment in path:
        matches = [
            row for row in rows if row is not None and row.text(FIELD) == segment
        ]
        assert len(matches) == 1, f"no single tree row for {'.'.join(path)!r}"
        found = matches[0]
        rows = [found.child(i) for i in range(found.childCount())]
    assert found is not None
    return found


def _reveal(qapp, widget: ComponentTreeWidget, item: QTreeWidgetItem) -> None:
    parent = item.parent()
    while parent is not None:
        parent.setExpanded(True)
        parent = parent.parent()
    _tree(widget).scrollToItem(item)
    qapp.processEvents()


def _double_click_value(qapp, widget: ComponentTreeWidget, item: QTreeWidgetItem):
    _reveal(qapp, widget, item)
    tree = _tree(widget)
    viewport = tree.viewport()
    assert viewport is not None
    rect = tree.visualItemRect(item)
    assert rect.isValid()
    x = tree.columnViewportPosition(VALUE) + tree.columnWidth(VALUE) // 2
    position = QPoint(x, rect.center().y())
    for event_type, buttons in (
        (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
        (QEvent.Type.MouseButtonDblClick, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                event_type,
                QPointF(position),
                QPointF(viewport.mapToGlobal(position)),
                Qt.MouseButton.LeftButton,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    qapp.processEvents()


def _editor[T: QWidget](widget: ComponentTreeWidget, editor_type: type[T]) -> T:
    editors = [
        editor
        for editor in _tree(widget).findChildren(editor_type)
        if editor.isVisible()
    ]
    assert len(editors) == 1, f"expected one open {editor_type.__name__} editor"
    return editors[0]


def _press(qapp, target: QWidget, key: Qt.Key) -> None:
    QApplication.sendEvent(
        target, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )
    qapp.processEvents()


def _edit_text(qapp, widget, item: QTreeWidgetItem, text: str, key: Qt.Key) -> None:
    _double_click_value(qapp, widget, item)
    editor = _editor(widget, QLineEdit)
    editor.setText(text)
    _press(qapp, editor, key)


def _error(widget: ComponentTreeWidget) -> QLabel:
    label = widget.findChild(QLabel, "component_error")
    assert label is not None
    return label


def _is_red(label: QLabel) -> bool:
    color = label.palette().color(label.foregroundRole())
    return color.red() > color.green() and color.red() > color.blue()


def _add_button(widget: ComponentTreeWidget) -> QPushButton:
    button = widget.findChild(QPushButton, "component_add")
    assert button is not None
    return button


def test_add_component_renders_tree(component_widget):
    widget, _control = component_widget
    assert widget.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    tree = _tree(widget)
    assert any(
        tree.topLevelItem(i).text(0) == "Q1" for i in range(tree.topLevelItemCount())
    )


def test_missing_required_field_displays_error_without_mutation(component_widget):
    widget, control = component_widget
    assert not widget.add_component("Ref", "test/required", {})
    error = _error(widget)
    assert "reference" in error.text()
    assert _is_red(error)
    assert "Ref" not in control.inspect_document().components
    assert widget.add_component(
        "Ref", "test/required", {"reference": "unresolved-name"}
    )
    assert widget.edit_parameter("Ref.reference", "ordinary-string")
    assert control.read_context().value("Ref.reference") == "ordinary-string"


def test_tree_lists_every_schema_field_with_unit_and_description(component_widget):
    widget, _control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})
    tree = _tree(widget)
    assert tree.columnCount() == 4
    root = _row(widget, "P1")
    # Kind appears once, on the component row, never as its own child row.
    assert root.text(VALUE) == "test/profiled"
    assert all(root.child(i).text(FIELD) != "kind" for i in range(root.childCount()))

    label, rate = _row(widget, "P1", "label"), _row(widget, "P1", "rate")
    assert label.text(VALUE) == "probe"
    assert not label.font(VALUE).italic()
    # Unset optional fields stay visible, styled apart from real values.
    assert rate.text(VALUE) != ""
    assert rate.font(VALUE).italic()
    assert rate.text(UNIT) == "MHz"
    assert "Sample rate." in rate.toolTip(FIELD)
    assert "P1.rate" in rate.toolTip(FIELD)
    assert label.text(SOURCE) != ""
    assert rate.text(SOURCE) == ""

    nested = _row(widget, "P1", "routing", "main", "ch")
    assert nested.font(VALUE).italic()
    assert "Channel index." in nested.toolTip(FIELD)
    assert _row(widget, "P1", "tags").text(VALUE) == "{}"


def test_in_place_edit_fills_an_unset_field(qapp, component_widget):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})

    _edit_text(qapp, widget, _row(widget, "P1", "rate"), "4300.5", Qt.Key.Key_Return)

    assert control.read_context().value("P1.rate") == 4300.5
    rate = _row(widget, "P1", "rate")
    assert rate.text(VALUE) == "4300.5"
    assert not rate.font(VALUE).italic()
    assert _error(widget).text() == ""


def test_child_of_unset_group_writes_the_shallowest_unset_ancestor(
    qapp, component_widget
):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})

    _edit_text(
        qapp,
        widget,
        _row(widget, "P1", "routing", "main", "ch"),
        "2",
        Qt.Key.Key_Return,
    )

    captured = control.read_context()
    assert captured.value("P1.routing.main") == {"ch": 2}
    assert _row(widget, "P1", "routing", "main", "ch").text(VALUE) == "2"


def test_escape_cancels_the_edit_without_writing(qapp, component_widget):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})
    before = control.inspect_document()

    _edit_text(qapp, widget, _row(widget, "P1", "label"), "changed", Qt.Key.Key_Escape)

    assert control.inspect_document() == before
    assert _row(widget, "P1", "label").text(VALUE) == "probe"
    assert not any(
        editor.isVisible() for editor in _tree(widget).findChildren(QLineEdit)
    )


def test_choice_field_edits_through_a_drop_down(qapp, component_widget):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})

    _double_click_value(qapp, widget, _row(widget, "P1", "mode"))
    combo = _editor(widget, QComboBox)
    assert {"A", "V"} <= {combo.itemText(i) for i in range(combo.count())}
    combo.setCurrentText("V")
    _press(qapp, combo, Qt.Key.Key_Return)

    assert control.read_context().value("P1.mode") == "V"


def test_rejected_edit_shows_error_and_keeps_document_value(qapp, component_widget):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe", "count": 3})

    _edit_text(qapp, widget, _row(widget, "P1", "count"), "-x: [", Qt.Key.Key_Return)
    assert _error(widget).text() != ""
    assert _is_red(_error(widget))

    _edit_text(qapp, widget, _row(widget, "P1", "count"), "2.5", Qt.Key.Key_Return)
    assert _error(widget).text() != ""
    assert control.read_context().value("P1.count") == 3
    assert _row(widget, "P1", "count").text(VALUE) == "3"


def test_successful_edit_keeps_expansion_and_selection(qapp, component_widget):
    widget, _control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})
    assert widget.add_component("P2", "test/profiled", {"label": "other"})
    tree = _tree(widget)
    _row(widget, "P2").setExpanded(False)
    routing = _row(widget, "P1", "routing")
    routing.setExpanded(True)
    _row(widget, "P1", "routing", "bias").setExpanded(True)
    tree.setCurrentItem(_row(widget, "P1", "routing", "bias", "ch"))

    assert widget.edit_parameter("P1.label", "renamed")

    assert not _row(widget, "P2").isExpanded()
    assert _row(widget, "P1", "routing", "bias").isExpanded()
    current = tree.currentItem()
    assert current is not None
    path: list[str] = []
    while current is not None:
        path.insert(0, current.text(FIELD))
        current = current.parent()
    assert path == ["P1", "routing", "bias", "ch"]


def test_extra_document_data_outside_the_schema_stays_visible_and_editable(
    qapp, component_widget
):
    widget, control = component_widget
    assert widget.add_component("O1", "test/open", {"custom": 5})
    custom = _row(widget, "O1", "custom")
    assert custom.text(VALUE) == "5"
    assert _row(widget, "O1", "rate").font(VALUE).italic()

    _edit_text(qapp, widget, custom, "6", Qt.Key.Key_Return)

    assert control.read_context().value("O1.custom") == 6


def test_without_parameter_context_the_tree_clears_and_add_is_disabled(
    qapp, tmp_path: Path, owner: ContextService
):
    control = ContextControlFacet(context=owner)
    widget = ComponentTreeWidget(control)
    try:
        assert _tree(widget).topLevelItemCount() == 0
        assert not _add_button(widget).isEnabled()
        hint = widget.findChild(QLabel, "component_hint")
        assert hint is not None and hint.text() != ""
        assert not _is_red(hint)
        assert _error(widget).text() == ""

        owner.install_entry(
            ResultEntry.create(
                "entry",
                result_root=tmp_path / "results",
                database_root=tmp_path / "Database",
            )
        )
        destination = next(
            combo
            for combo in widget.findChildren(QComboBox)
            if combo.findText("Setup template") >= 0
        )
        destination.setCurrentText("Setup template")
        assert _add_button(widget).isEnabled()
        assert hint.text() == ""
        assert widget.add_component(
            "T1", "test/profiled", {"label": "template"}, setup=True
        )
        assert _row(widget, "T1", "label").text(VALUE) == "template"
    finally:
        _close(qapp, widget)
