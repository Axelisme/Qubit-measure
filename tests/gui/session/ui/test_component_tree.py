"""The component frontend submits owner commands and exposes failures."""

from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from qtpy.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.component_tree import ComponentTreeWidget
from zcu_tools.gui.widgets import error_colour
from zcu_tools.resources.entry import (
    ComponentSchema,
    ResultEntry,
    component_registry,
    document_value,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._user_input import ComponentDriver
from tests.resources.entry.fakes import (
    Profiled,
    ProfiledChannel,
    register_fakes,
    registry_state,
    restore_registry,
)

FIELD, VALUE, UNIT, METADATA = range(4)


class RequiredComponent(ComponentSchema):
    reference: str


class OpenComponent(ComponentSchema):
    model_config = ConfigDict(extra="allow")
    rate: float | None = None


class RequiredRouting(BaseModel):
    main: ProfiledChannel
    bias: ProfiledChannel | None = None


class RequiredForm(ComponentSchema):
    label: str = Field(description="Required label.")
    routing: RequiredRouting
    optional_routing: RequiredRouting | None = None
    rate: float = Field(json_schema_extra={"unit": "MHz"}, description="Sample rate.")
    mode: Literal["A", "V"]
    enabled: bool
    count: int = Field(strict=True)
    notes: str | None = None


class RequiredDrive(ComponentSchema):
    sense: str
    source: str


# Forty required fields: more rows than any test screen fits without scrolling.
# create_model takes field definitions as untyped keyword arguments.
_TALL_FIELDS: dict[str, Any] = {
    f"value_{index:02d}": (float, ...) for index in range(40)
}
TallComponent = create_model("TallComponent", __base__=ComponentSchema, **_TALL_FIELDS)


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
        component_registry.register("test/tall", TallComponent)
        component_registry.register("test/form", RequiredForm)
        component_registry.register("test/drive", RequiredDrive)
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
    entry = ResultEntry.create(
        "entry",
        result_root=tmp_path / "results",
        database_root=tmp_path / "Database",
    )
    # Parameter writes need a selected saved point; start on one.
    entry.new_point("working")
    owner.install_entry(entry)
    owner.use_point(PointUseRequest("working"))
    control = ContextControlFacet(context=owner)
    widget = ComponentTreeWidget(control)
    widget.resize(900, 700)
    widget.show()
    qapp.processEvents()
    try:
        yield widget, control
    finally:
        # An open Add dialog stays activeModalWidget() after its parent closes.
        for dialog in _visible_dialogs(widget):
            dialog.reject()
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
    return color == error_colour(label)


def _add_button(widget: ComponentTreeWidget) -> QPushButton:
    button = widget.findChild(QPushButton, "component_add")
    assert button is not None
    return button


def test_add_component_renders_tree(component_widget):
    widget, _control = component_widget
    assert widget.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    tree = _tree(widget)
    roots = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    assert any(root is not None and root.text(0) == "Q1" for root in roots)


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
    children = [root.child(i) for i in range(root.childCount())]
    assert all(child is not None and child.text(FIELD) != "kind" for child in children)

    label, rate = _row(widget, "P1", "label"), _row(widget, "P1", "rate")
    assert label.text(VALUE) == "probe"
    assert not label.font(VALUE).italic()
    # Unset optional fields stay visible, styled apart from real values.
    assert rate.text(VALUE) != ""
    assert rate.font(VALUE).italic()
    assert rate.text(UNIT) == "MHz"
    assert "Sample rate." in rate.toolTip(FIELD)
    assert "P1.rate" in rate.toolTip(FIELD)

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


def test_metadata_column_shows_caller_mapping_and_survives_value_edits(
    qapp, component_widget
):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})
    assert widget.add_component("O1", "test/open", {"custom": 5})
    header = _tree(widget).headerItem()
    assert header is not None
    assert header.text(METADATA) == "Metadata"
    target = control.read_context().target
    control.set_parameter(
        target, "P1.label", "probe", metadata={"stderr": 0.8, "comment": "ok"}
    )
    control.set_parameter(target, "O1.custom", 5, metadata={"note": [1, None]})
    widget.refresh()
    label = _row(widget, "P1", "label")
    assert label.text(METADATA) == "{stderr: 0.8, comment: ok}"
    assert label.toolTip(METADATA) == "stderr: 0.8\ncomment: ok"
    custom = _row(widget, "O1", "custom")
    assert custom.text(METADATA) == "{note: [1, null]}"
    assert custom.toolTip(METADATA) == "note:\n- 1\n- null"

    _edit_text(qapp, widget, label, "renamed", Qt.Key.Key_Return)

    label = _row(widget, "P1", "label")
    assert label.text(VALUE) == "renamed"
    assert label.text(METADATA) == "{stderr: 0.8, comment: ok}"
    control.set_parameter(target, "P1.label", "renamed", metadata={})
    widget.refresh()
    label = _row(widget, "P1", "label")
    assert label.text(METADATA) == ""
    assert label.toolTip(METADATA) == ""


def test_group_and_unset_rows_show_their_path_metadata(component_widget):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})
    target = control.read_context().target
    control.set_parameter(target, "P1.routing.main", {"ch": 1})
    control.set_parameter(target, "P1.routing.main.ch", 1, metadata={"stderr": 0.1})
    # A null group leaves its channel unset while that path keeps its metadata.
    control.set_parameter(
        target, "P1.routing.main", None, metadata={"comment": "parked"}
    )
    widget.refresh()

    group = _row(widget, "P1", "routing", "main")
    unset = _row(widget, "P1", "routing", "main", "ch")
    assert unset.font(VALUE).italic()
    shown = [(row.text(METADATA), row.toolTip(METADATA)) for row in (group, unset)]
    assert shown == [
        ("{comment: parked}", "comment: parked"),
        ("{stderr: 0.1}", "stderr: 0.1"),
    ]

    control.set_parameter(target, "P1.routing.main", None, metadata={})
    widget.refresh()
    group = _row(widget, "P1", "routing", "main")
    assert group.text(METADATA) == ""
    assert group.toolTip(METADATA) == ""


def test_multiline_metadata_summary_is_one_line_and_tooltip_keeps_content(
    component_widget,
):
    widget, control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe"})
    metadata: YamlMap = {
        "run\nnote": "first line\n\n  indented second\n",
        "history": ["a\nb", {"why": "x\ny"}],
        "stderr": 0.25,
    }
    target = control.read_context().target
    control.set_parameter(target, "P1.label", "probe", metadata=metadata)
    widget.refresh()

    label = _row(widget, "P1", "label")
    summary = label.text(METADATA)
    assert "\n" not in summary
    assert yaml.safe_load(summary) == metadata
    assert yaml.safe_load(label.toolTip(METADATA)) == metadata


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


def _count_rejection(count: object) -> ValidationError:
    """The pydantic error the owner raises for a non-integer Profiled count."""
    with pytest.raises(ValidationError) as caught:
        Profiled.model_validate({"kind": "test/profiled", "label": "x", "count": count})
    return caught.value


def test_rejected_edit_error_names_the_field_with_full_text_in_tooltip(
    qapp, component_widget
):
    widget, _control = component_widget
    assert widget.add_component("P1", "test/profiled", {"label": "probe", "count": 3})

    _edit_text(qapp, widget, _row(widget, "P1", "count"), "2.5", Qt.Key.Key_Return)

    message = _count_rejection(2.5).errors()[0]["msg"]
    error = _error(widget)
    assert error.text() == f"count: {message}"
    assert message in error.toolTip()
    assert len(error.toolTip()) > len(error.text())


# ----------------------------------------------------------------------
# Add component dialog
# ----------------------------------------------------------------------


def _visible_dialogs(widget: ComponentTreeWidget) -> list[QDialog]:
    return [dialog for dialog in widget.findChildren(QDialog) if dialog.isVisible()]


def _open_add_dialog(qapp, widget: ComponentTreeWidget, kind: str) -> QDialog:
    _add_button(widget).click()
    qapp.processEvents()
    dialogs = _visible_dialogs(widget)
    assert len(dialogs) == 1, "Add component… opens exactly one visible dialog"
    dialog = dialogs[0]
    _child(dialog, QComboBox, "add_kind").setCurrentText(kind)
    qapp.processEvents()
    return dialog


def _child[T: QWidget](parent: QWidget, widget_type: type[T], name: str) -> T:
    child = parent.findChild(widget_type, name)
    assert child is not None, f"no {widget_type.__name__} named {name!r}"
    return child


def _fill(dialog: QDialog, path: str, text: str) -> None:
    field = dialog.findChild(QWidget, f"field:{path}")
    if isinstance(field, QComboBox):
        field.setCurrentText(text)
    elif isinstance(field, QLineEdit):
        field.setText(text)
    else:
        raise AssertionError(f"no input widget for field {path!r}")


def _submit(qapp, dialog: QDialog) -> None:
    _child(dialog, QPushButton, "add_submit").click()
    qapp.processEvents()


def _current_path(widget: ComponentTreeWidget) -> list[str]:
    current = _tree(widget).currentItem()
    path: list[str] = []
    while current is not None:
        path.insert(0, current.text(FIELD))
        current = current.parent()
    return path


def test_add_button_opens_a_modal_dialog_naming_the_destination(qapp, component_widget):
    widget, control = component_widget
    control.new_point(PointCreateRequest("P1", "", CommandOrigin("gui")))
    widget.refresh()

    dialog = _open_add_dialog(qapp, widget, "test/profiled")
    assert dialog.isModal()
    current_text = _child(dialog, QLabel, "add_destination").text()
    assert "P1" in current_text
    _child(dialog, QPushButton, "add_cancel").click()
    qapp.processEvents()
    assert _visible_dialogs(widget) == []

    destination = next(
        combo
        for combo in widget.findChildren(QComboBox)
        if combo.findText("Setup template") >= 0
    )
    destination.setCurrentText("Setup template")
    dialog = _open_add_dialog(qapp, widget, "test/profiled")
    setup_text = _child(dialog, QLabel, "add_destination").text()
    assert setup_text != current_text
    assert "P1" not in setup_text
    _child(dialog, QLineEdit, "add_name").setText("T1")
    _fill(dialog, "label", "template")
    _submit(qapp, dialog)

    assert "T1" in control.inspect_document(setup=True).components
    assert "T1" not in control.inspect_document().components


def test_dialog_fields_follow_the_kind_schema_and_rebuild_on_kind_change(
    qapp, component_widget, tmp_path
):
    widget, _control = component_widget
    driver = ComponentDriver(widget)
    driver.click("Add component…")
    driver.select_combo("test/form", object_name="add_kind")
    driver.screenshot(tmp_path / "required-form.png")
    dialog = _visible_dialogs(widget)[0]

    fields = {
        field.objectName()
        for field in dialog.findChildren(QWidget)
        if field.objectName().startswith("field:")
    }
    assert fields == {
        "field:label",
        "field:routing.main.ch",
        "field:rate",
        "field:mode",
        "field:enabled",
        "field:count",
    }
    submit = driver.observe(object_name="add_submit")
    assert not submit.enabled
    assert (
        submit.tooltip
        == "Missing: name, label, routing.main.ch, rate, mode, enabled, count"
    )
    nested = _child(dialog, QLabel, "label:routing.main.ch")
    assert nested.text() == "routing.main.ch *"
    assert nested.indent() == 0
    assert driver.observe(object_name="field:routing.main.ch").visible
    assert (
        "Channel index." in driver.observe(object_name="field:routing.main.ch").tooltip
    )
    labels = {label.text() for label in dialog.findChildren(QLabel)}
    assert "routing" not in labels and "main" not in labels
    mode = _child(dialog, QComboBox, "field:mode")
    assert _combo_items(mode) == ["", "A", "V"]
    enabled = _child(dialog, QComboBox, "field:enabled")
    assert set(_combo_items(enabled)) == {"", "true", "false"}
    assert driver.observe(object_name="unit:rate").text == "MHz"

    driver.select_combo("test/required", object_name="add_kind")
    driver.screenshot(tmp_path / "changed-kind.png")
    assert driver.observe(object_name="field:reference").visible
    assert (
        driver.observe(object_name="add_submit").tooltip == "Missing: name, reference"
    )
    assert dialog.findChild(QWidget, "field:label") is None


def test_dialog_without_required_fields_shows_hint_and_adds_name_and_kind(
    qapp, component_widget, tmp_path
):
    widget, control = component_widget
    driver = ComponentDriver(widget)
    driver.click("Add component…")
    driver.select_combo("test/open", object_name="add_kind")
    driver.screenshot(tmp_path / "no-required-fields.png")
    assert "No required fields. Edit the others in Inspect after adding." in {
        item.text for item in driver.visible().widgets
    }
    submit = driver.observe(object_name="add_submit")
    assert not submit.enabled
    assert submit.tooltip == "Missing: name"
    driver.enter_text(object_name="add_name", value="O1")
    driver.screenshot(tmp_path / "added.png")
    assert driver.dialogs() == ()
    assert document_value(control.inspect_document(), "O1") == {"kind": "test/open"}


def _fill_required_form(dialog: QDialog, *, count: str = "3") -> None:
    _fill(dialog, "label", "probe")
    _fill(dialog, "routing.main.ch", "2")
    _fill(dialog, "rate", "4300.5")
    _fill(dialog, "mode", "V")
    _fill(dialog, "enabled", "true")
    _fill(dialog, "count", count)


def test_dialog_submits_filled_fields_as_nested_mapping_and_selects_the_component(
    qapp, component_widget
):
    widget, control = component_widget
    dialog = _open_add_dialog(qapp, widget, "test/form")
    _child(dialog, QLineEdit, "add_name").setText("P1")
    _fill_required_form(dialog)
    assert _child(dialog, QLabel, "label:routing.main.ch").text() == "routing.main.ch *"
    _submit(qapp, dialog)

    assert _visible_dialogs(widget) == []
    component = document_value(control.inspect_document(), "P1")
    assert component == {
        "kind": "test/form",
        "label": "probe",
        "routing": {"main": {"ch": 2}},
        "rate": 4300.5,
        "mode": "V",
        "enabled": True,
        "count": 3,
    }
    assert _current_path(widget) == ["P1"]
    assert _row(widget, "P1", "routing", "main", "ch").text(VALUE) == "2"


def test_dialog_marks_missing_required_fields_and_does_not_submit(
    qapp, component_widget
):
    widget, control = component_widget
    dialog = _open_add_dialog(qapp, widget, "test/profiled")
    name_label = _child(dialog, QLabel, "label:add_name")
    required_label = _child(dialog, QLabel, "label:label")
    submit = _child(dialog, QPushButton, "add_submit")
    assert _is_red(name_label) and _is_red(required_label)
    assert not submit.isEnabled()
    assert submit.toolTip() == "Missing: name, label"

    _submit(qapp, dialog)

    assert _visible_dialogs(widget) == [dialog]
    assert control.inspect_document().components == {}
    _child(dialog, QLineEdit, "add_name").setText("P1")
    assert not submit.isEnabled()
    assert submit.toolTip() == "Missing: label"
    _fill(dialog, "label", "probe")
    assert not _is_red(name_label) and not _is_red(required_label)
    assert submit.isEnabled() and submit.toolTip() == ""
    _fill(dialog, "label", "")
    assert _is_red(required_label) and not submit.isEnabled()
    assert submit.toolTip() == "Missing: label"
    _fill(dialog, "label", "probe")
    _child(dialog, QLineEdit, "add_name").setText("")
    assert _is_red(name_label) and not submit.isEnabled()
    assert submit.toolTip() == "Missing: name"
    _child(dialog, QLineEdit, "add_name").setText("P1")
    assert submit.isEnabled() and submit.toolTip() == ""
    _submit(qapp, dialog)
    assert "P1" in control.inspect_document().components


def test_owner_rejection_stays_in_the_dialog_with_input_kept(qapp, component_widget):
    widget, control = component_widget
    dialog = _open_add_dialog(qapp, widget, "test/form")
    _child(dialog, QLineEdit, "add_name").setText("P1")
    _fill_required_form(dialog, count="2.5")

    _submit(qapp, dialog)

    assert _visible_dialogs(widget) == [dialog]
    assert "P1" not in control.inspect_document().components
    message = _count_rejection(2.5).errors()[0]["msg"]
    error = _child(dialog, QLabel, "add_error")
    assert error.text() == f"count: {message}"
    assert _is_red(error)
    assert message in error.toolTip()
    assert len(error.toolTip()) > len(error.text())
    assert _child(dialog, QLineEdit, "add_name").text() == "P1"
    assert _child(dialog, QLineEdit, "field:label").text() == "probe"
    assert _child(dialog, QLineEdit, "field:count").text() == "2.5"
    # The main tree's own error line stays clear: the failure belongs to Add.
    assert _error(widget).text() == ""


def test_dialog_height_stays_within_the_screen_and_fields_scroll(
    qapp, component_widget, tmp_path
):
    widget, _control = component_widget
    driver = ComponentDriver(widget)
    driver.click("Add component…")
    driver.select_combo("test/required", object_name="add_kind")
    dialog = _visible_dialogs(widget)[0]
    screen = dialog.screen()
    assert screen is not None
    available = screen.availableGeometry()
    # Set up a short window near the bottom, as a user may have placed it.
    dialog.move(
        available.left() + 10, available.bottom() - dialog.frameGeometry().height() - 10
    )
    qapp.processEvents()
    driver.screenshot(tmp_path / "short-kind.png")
    assert available.contains(dialog.frameGeometry())
    driver.select_combo("test/tall", object_name="add_kind")
    driver.screenshot(tmp_path / "tall-kind.png")
    assert available.contains(dialog.frameGeometry())
    assert dialog.height() <= 0.9 * available.height()
    scroll = _child(dialog, QScrollArea, "add_fields")
    scrollbar = scroll.verticalScrollBar()
    assert scrollbar is not None and scrollbar.maximum() > 0
    for name in ("add_submit", "add_cancel"):
        assert driver.observe(object_name=name).visible
        button = _child(dialog, QPushButton, name)
        placed = QRect(button.mapTo(dialog, QPoint(0, 0)), button.size())
        assert dialog.rect().contains(placed)
    driver.click("Cancel")
    driver.screenshot(tmp_path / "cancelled.png")
    assert driver.dialogs() == ()


def _combo_items(combo: QComboBox) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def _completions(editor: QLineEdit) -> set[str]:
    completer = editor.completer()
    assert completer is not None, "expected a component-name completer"
    model = completer.model()
    assert model is not None
    return {str(model.index(row, 0).data()) for row in range(model.rowCount())}


def test_role_reference_field_picks_a_matching_component(qapp, component_widget):
    widget, control = component_widget
    assert widget.add_component("S1", "fake/sensor", {})
    assert widget.add_component("S2", "fake/sensor", {})
    assert widget.add_component("D1", "fake/drive/a", {})

    _double_click_value(qapp, widget, _row(widget, "D1", "sense"))
    combo = _editor(widget, QComboBox)
    assert _combo_items(combo) == ["", "S1", "S2"]
    combo.setCurrentText("S2")
    _press(qapp, combo, Qt.Key.Key_Return)

    assert control.read_context().value("D1.sense") == "S2"


def test_other_text_fields_complete_component_names(qapp, component_widget):
    widget, _control = component_widget
    assert widget.add_component("S1", "fake/sensor", {})
    assert widget.add_component("D1", "fake/drive/a", {})

    _double_click_value(qapp, widget, _row(widget, "D1", "source"))
    editor = _editor(widget, QLineEdit)
    assert _completions(editor) == {"S1", "D1"}
    _press(qapp, editor, Qt.Key.Key_Escape)


def test_dialog_role_reference_field_is_a_drop_down_of_matching_components(
    qapp, component_widget
):
    widget, control = component_widget
    control.new_point(PointCreateRequest("P1", "", CommandOrigin("gui")))
    widget.refresh()
    assert widget.add_component("S1", "fake/sensor", {})

    dialog = _open_add_dialog(qapp, widget, "test/drive")
    sense = _child(dialog, QComboBox, "field:sense")
    assert _combo_items(sense) == ["", "S1"]
    assert _completions(_child(dialog, QLineEdit, "field:source")) == {"S1"}
    _child(dialog, QLineEdit, "add_name").setText("D1")
    _fill(dialog, "sense", "S1")
    _fill(dialog, "source", "S1")
    _submit(qapp, dialog)

    assert control.read_context().value("D1.sense") == "S1"


def test_component_error_uses_theme_foreground(component_widget, window_palette):
    widget, _ = component_widget
    widget.setPalette(window_palette)
    assert not widget.add_component("Ref", "test/required", {})
    error = _error(widget)
    assert error.palette().color(error.foregroundRole()) == error_colour(error)
