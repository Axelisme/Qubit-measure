"""Smoke tests for InspectDialog."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEvent, Qt  # type: ignore[attr-defined]
from qtpy.QtGui import QKeyEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (
    QApplication,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTreeWidget,
)
from zcu_tools.cfg_model import module_spec_from_raw, waveform_spec_from_raw
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.app.measure.ui import inspect_dialog
from zcu_tools.gui.app.measure.ui.inspect_dialog import (
    InspectDialog,
    _MlCreateDialog,
)
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.binding import CfgDraft
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, component_registry

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.resources.entry.fakes import registry_state, restore_registry
from zcu_lab.components import QubitSchema, ResonatorSchema


@pytest.fixture(autouse=True)
def registered_lab_kinds() -> Generator[None]:
    """Register the kinds _make_ml documents use; the component tree projects them."""
    before = registry_state()
    try:
        component_registry.register("resonator", ResonatorSchema)
        component_registry.register("qubit/transmon", QubitSchema)
        yield
    finally:
        restore_registry(before)


def _make_ml() -> ParameterContext:
    components = {
        "R1": ResonatorSchema(kind="resonator"),
        "Q1": QubitSchema(kind="qubit/transmon"),
    }
    parameters = parameter_capture(components)
    library = parameters.library.replaced_module(
        "R1.readout.readout_rf",
        module_spec_from_raw(
            {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_freq": 6000.0,
                "ro_length": 1.0,
                "trig_offset": 0.0,
            }
        ),
    ).replaced_waveform(
        "Q1.waveforms.drive_wav",
        waveform_spec_from_raw({"style": "const", "length": 1.0}),
    )
    return parameter_capture(components, library)


def _make_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.context_control = ctrl
    ctrl.inspect_document.side_effect = lambda **_kwargs: ctrl.read_context().document()
    return ctrl


def _make_ctrl_with_ml(parameters: ParameterContext) -> MagicMock:
    ctrl = _make_ctrl()
    ctrl.read_context.return_value = parameters
    ctrl.get_bus.return_value = MagicMock()
    ctrl.list_device_names.return_value = []
    ctrl.arb_waveforms.list_data_keys.return_value = []
    _wire_cfg_editor(ctrl)
    return ctrl


def _wire_cfg_editor(ctrl: MagicMock) -> None:
    """Simulate the CfgEditorService open/commit/get_draft contract.

    The embedded editor opens a *committable* session from the live ml
    (open_cfg_editor with from_name; ADR-0067) and commits via replace_cfg_editor;
    seeded sessions still exist for tab/writeback. Build a real CfgDraft
    per open so attach() works, keyed by a fake editor_id, with owner→id discovery,
    commit (records last commit), and teardown.
    """

    drafts: dict[str, CfgDraft] = {}
    owner_to_id: dict[str, str] = {}
    counter = {"n": 0}

    def _register(spec, value, owner_key) -> str:
        if owner_key is not None and owner_key in owner_to_id:
            drafts.pop(owner_to_id.pop(owner_key), None)
        counter["n"] += 1
        eid = f"editor-{counter['n']}"
        drafts[eid] = MeasureCfgBindings(ctrl).new_draft(
            CfgSchema(spec, value), parameters=ctrl.read_context()
        )
        if owner_key is not None:
            owner_to_id[owner_key] = eid
        return eid

    def _open_seeded(seed, *, gc=False, owner_key=None):
        return _register(seed.spec, seed.value, owner_key), []

    def _open(item_kind, *, from_name, gc=True, owner_key=None):
        # Load the existing entry's shape from the live ml (from_name path).
        library = ctrl.read_context().library
        stored = (
            library.get_module(from_name)
            if item_kind == "module"
            else library.get_waveform(from_name)
        )
        to_value = (
            module_cfg_to_value if item_kind == "module" else waveform_cfg_to_value
        )
        spec, value = to_value(stored)
        return _register(spec, value, owner_key), []

    def _remove(editor_id):
        drafts.pop(editor_id, None)
        for owner, eid in list(owner_to_id.items()):
            if eid == editor_id:
                owner_to_id.pop(owner)

    def _commit(editor_id, name):
        ctrl.committed = (name, drafts[editor_id])  # record for assertions
        _remove(editor_id)

    def _replace(editor_id, old_name, new_name):
        draft = drafts[editor_id]
        schema = draft.snapshot()
        from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict

        parameters = ctrl.read_context()
        raw = schema_to_raw_dict(schema, parameters)
        library = parameters.library
        is_module = old_name in library.module_keys()
        kind = "module" if is_module else "waveform"
        if old_name != new_name:
            library = library.renamed(old_name, new_name, kind=kind)
        library = (
            library.replaced_module(new_name, module_spec_from_raw(raw))
            if is_module
            else library.replaced_waveform(new_name, waveform_spec_from_raw(raw))
        )
        ctrl.read_context.return_value = ParameterContext(
            point=parameters.document(),
            library=library,
            target=parameters.target,
            roles=None,
        )
        ctrl.replaced = (old_name, new_name, schema)
        _remove(editor_id)

    ctrl._drafts = drafts
    ctrl.open_seeded_cfg_editor.side_effect = _open_seeded
    ctrl.open_cfg_editor.side_effect = _open
    ctrl.commit_cfg_editor.side_effect = _commit
    ctrl.replace_cfg_editor.side_effect = _replace
    ctrl.get_cfg_editor_draft.side_effect = lambda eid: drafts[eid]
    ctrl.editor_id_for_owner.side_effect = lambda owner: owner_to_id.get(owner)
    ctrl.teardown_cfg_editor.side_effect = _remove


def test_inspect_dialog_init_and_refresh(qapp):
    ctrl = _make_ctrl()
    bus = MagicMock()

    # Setup mock returns
    mock_md = MagicMock()
    mock_md.items.return_value = {
        "scalar_int": 42,
        "scalar_float": 3.14,
        "nested": {"key": "value"},
    }.items()
    ctrl.get_current_md.return_value = mock_md

    ctrl.read_context.return_value = _make_ml()

    dialog = InspectDialog(ctrl, bus)

    # Check subscriptions
    assert bus.subscribe.call_count >= 2

    # Check if ML was populated (tree widget)
    # The top level items should be "modules" and "waveforms"
    assert dialog._ml_tree.topLevelItemCount() == 2


def test_inspect_dialog_toolbar_has_arb_waveforms_before_refresh(qapp):
    from qtpy.QtWidgets import QWidget
    from zcu_tools.gui.app.measure.remote.dialogs import DialogName

    class Parent(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.opened: list[DialogName] = []

        def open_dialog(self, name: DialogName) -> None:
            self.opened.append(name)

    parent = Parent()
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock(), parent=parent)

    root_layout = dialog.layout()
    assert root_layout is not None
    toolbar_item = root_layout.itemAt(0)
    assert toolbar_item is not None
    toolbar = toolbar_item.layout()
    assert toolbar is not None
    assert toolbar.indexOf(dialog._arb_waveform_btn) < toolbar.indexOf(
        dialog._refresh_btn
    )

    dialog._arb_waveform_btn.click()

    assert parent.opened == [DialogName.ARB_WAVEFORM]


def test_C9_inspect_dialog_raises_when_parent_has_no_open_dialog(qapp):
    """_on_arb_waveform_clicked must raise RuntimeError when parent lacks open_dialog."""
    from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]

    # Parent is a plain QWidget with no open_dialog attribute.
    bare_parent = QWidget()
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock(), parent=bare_parent)

    with pytest.raises(RuntimeError, match="open_dialog"):
        dialog._on_arb_waveform_clicked()


def _select_ml_child(dialog: InspectDialog, group_index: int, child_index: int = 0):
    group = dialog._ml_tree.topLevelItem(group_index)
    assert group is not None
    child = group.child(child_index)
    assert child is not None
    dialog._ml_tree.setCurrentItem(child)
    return child


def test_inspect_dialog_ml_embeds_service_editor_without_raw_or_modify_controls(qapp):
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())

    assert isinstance(dialog._ml_form_widget, inspect_dialog.CfgFormWidget)
    assert not hasattr(dialog, "_ml_text")
    assert not hasattr(dialog, "_modify_ml_btn")
    assert not hasattr(dialog, "_rename_ml_btn")
    assert dialog._create_btn.text() == "New"
    assert dialog._del_ml_btn.text() == "Delete"

    _select_ml_child(dialog, 0)
    assert dialog._ml_name_edit.text() == "R1.readout.readout_rf"
    assert dialog._ml_status_label.text() == "Saved"
    assert dialog._ml_revert_btn.text() == "Revert"
    assert dialog._ml_apply_btn.text() == "Apply"
    assert dialog._ml_form_widget._draft is ctrl._drafts[dialog._ml_editor_id]


def test_inspect_dialog_ml_apply_replaces_name_and_opens_fresh_clean_draft(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    old_editor_id = dialog._ml_editor_id
    assert old_editor_id is not None

    draft = ctrl._drafts[old_editor_id]
    draft.root.fields["ro_freq"].set_value(6123.0)
    dialog._ml_name_edit.setText("R1.readout.readout_v2")
    dialog._ml_name_edit.textEdited.emit("R1.readout.readout_v2")
    qapp.processEvents()

    assert ml.library.get_module("R1.readout.readout_rf").to_dict()["ro_freq"] == 6000.0
    assert dialog._ml_status_label.text() == "Unsaved"
    assert dialog._ml_apply_btn.isEnabled()

    dialog._ml_apply_btn.click()
    qapp.processEvents()

    assert "R1.readout.readout_rf" not in ctrl.read_context().library.module_keys()
    assert (
        ctrl.read_context()
        .library.get_module("R1.readout.readout_v2")
        .to_dict()["ro_freq"]
        == 6123.0
    )
    ctrl.replace_cfg_editor.assert_called_once_with(
        old_editor_id, "R1.readout.readout_rf", "R1.readout.readout_v2"
    )
    assert dialog._ml_selected_data == ("modules", "R1.readout.readout_v2")
    assert dialog._ml_status_label.text() == "Saved"
    assert dialog._ml_editor_id != old_editor_id
    assert (
        getattr(
            dialog._ml_form_widget.read_schema().value.fields["ro_freq"], "value", None
        )
        == 6123.0
    )


def test_inspect_dialog_ml_waveform_selection_uses_same_embedded_session(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = InspectDialog(ctrl, MagicMock())

    _select_ml_child(dialog, 1)
    editor_id = dialog._ml_editor_id
    assert editor_id is not None
    ctrl._drafts[editor_id].root.fields["length"].set_value(1.5)
    qapp.processEvents()
    dialog._ml_apply_btn.click()

    assert (
        ctrl.read_context()
        .library.get_waveform("Q1.waveforms.drive_wav")
        .to_dict()["length"]
        == 1.5
    )
    assert dialog._ml_selected_data == ("waveforms", "Q1.waveforms.drive_wav")
    assert dialog._ml_status_label.text() == "Saved"


def test_inspect_dialog_ml_context_switch_apply_failure_keeps_draft(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorError

    ml_a = _make_ml()
    ml_b = _make_ml()
    ctrl = _make_ctrl_with_ml(ml_a)
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    editor_id = dialog._ml_editor_id
    assert editor_id is not None
    ctrl._drafts[editor_id].root.fields["ro_freq"].set_value(6123.0)
    dialog._ml_name_edit.setText("R1.readout.readout_v2")
    dialog._ml_name_edit.textEdited.emit("R1.readout.readout_v2")

    ctrl.read_context.return_value = ml_b
    dialog.refresh()
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    ctrl.replace_cfg_editor.side_effect = CfgEditorError("different context")
    dialog._ml_apply_btn.click()

    assert (
        ml_b.library.get_module("R1.readout.readout_rf").to_dict()["ro_freq"] == 6000.0
    )
    assert dialog._ml_editor_id == editor_id
    assert dialog._ml_name_edit.text() == "R1.readout.readout_v2"
    assert dialog._ml_status_label.text() == "Unsaved"


def test_inspect_dialog_ml_apply_failure_keeps_live_and_draft(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    editor_id = dialog._ml_editor_id
    assert editor_id is not None
    draft = ctrl._drafts[editor_id]
    draft.root.fields["ro_freq"].set_value(6123.0)
    dialog._ml_name_edit.setText("Q1.waveforms.drive_wav")
    dialog._ml_name_edit.textEdited.emit("Q1.waveforms.drive_wav")
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    ctrl.replace_cfg_editor.side_effect = ValueError("collision")

    dialog._ml_apply_btn.click()

    assert set(ml.library.module_keys()) == {"R1.readout.readout_rf"}
    assert set(ml.library.waveform_keys()) == {"Q1.waveforms.drive_wav"}
    assert dialog._ml_editor_id == editor_id
    assert (
        getattr(
            dialog._ml_form_widget.read_schema().value.fields["ro_freq"], "value", None
        )
        == 6123.0
    )
    assert dialog._ml_status_label.text() == "Unsaved"


def test_inspect_dialog_ml_revert_reloads_live_entry(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    editor_id = dialog._ml_editor_id
    assert editor_id is not None
    ctrl._drafts[editor_id].root.fields["ro_freq"].set_value(6123.0)
    dialog._ml_name_edit.setText("R1.readout.renamed")
    dialog._ml_name_edit.textEdited.emit("R1.readout.renamed")

    dialog._ml_revert_btn.click()

    assert dialog._ml_selected_data == ("modules", "R1.readout.readout_rf")
    assert dialog._ml_name_edit.text() == "R1.readout.readout_rf"
    assert dialog._ml_status_label.text() == "Saved"
    assert dialog._ml_editor_id != editor_id
    assert (
        getattr(
            dialog._ml_form_widget.read_schema().value.fields["ro_freq"], "value", None
        )
        == 6000.0
    )
    assert (
        "R1.readout.readout_rf" in ml.library.module_keys()
        and "R1.readout.renamed" not in ml.library.module_keys()
    )


def test_inspect_dialog_ml_dirty_selection_cannot_silently_discard(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    dialog._ml_name_edit.setText("R1.readout.draft_name")
    dialog._ml_name_edit.textEdited.emit("R1.readout.draft_name")
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )

    _select_ml_child(dialog, 1)

    assert dialog._ml_selected_data == ("modules", "R1.readout.readout_rf")
    assert dialog._current_ml_item_data() == ("modules", "R1.readout.readout_rf")
    assert dialog._ml_name_edit.text() == "R1.readout.draft_name"


def test_inspect_dialog_ml_dirty_selection_save_refreshes_tree_after_apply(
    qapp, monkeypatch
):
    from qtpy.QtWidgets import QMessageBox

    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    dialog._ml_name_edit.setText("R1.readout.readout_v2")
    dialog._ml_name_edit.textEdited.emit("R1.readout.readout_v2")

    replace = ctrl.replace_cfg_editor.side_effect

    def _replace_and_refresh(*args: Any) -> None:
        replace(*args)
        dialog.refresh()

    ctrl.replace_cfg_editor.side_effect = _replace_and_refresh
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Save,
    )
    _select_ml_child(dialog, 1)
    qapp.processEvents()

    assert dialog._ml_selected_data == ("waveforms", "Q1.waveforms.drive_wav")
    assert dialog._ml_name_edit.text() == "Q1.waveforms.drive_wav"
    assert dialog._find_ml_item(("modules", "R1.readout.readout_rf")) is None
    assert dialog._find_ml_item(("modules", "R1.readout.readout_v2")) is not None
    assert not dialog._ml_refresh_pending


def test_inspect_dialog_ml_dirty_close_requires_explicit_decision(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    dialog._ml_name_edit.setText("R1.readout.draft_name")
    dialog._ml_name_edit.textEdited.emit("R1.readout.draft_name")
    dialog.show()
    dialog.raise_()

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )
    dialog.close()
    assert dialog.isVisible()
    assert dialog._ml_editor_id is not None
    assert dialog._ml_status_label.text() == "Unsaved"

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: QMessageBox.StandardButton.Discard,
    )
    dialog.close()
    assert not dialog.isVisible()
    assert dialog._ml_editor_id is None


@pytest.fixture
def native_delete_inspector(
    qapp, tmp_path: Path, monkeypatch
) -> Generator[tuple[InspectDialog, ResultEntry]]:
    background = BackgroundRunner()
    monkeypatch.setattr(
        "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
        lambda: background,
    )
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        progress_transport=DirectProgressTransport(),
        project_root=str(tmp_path),
    )
    ctrl.setup_control.new_entry(EntryRequest("entry"))
    entry = ResultEntry.open(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    entry.new_point("point")
    with entry.module_library("point").edit() as draft:
        draft.set_module(
            "R1.readout.readout_rf",
            module_spec_from_raw(
                {
                    "type": "readout/direct",
                    "ro_ch": 0,
                    "ro_freq": 6000.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.0,
                }
            ),
        )
        draft.set_waveform(
            "Q1.waveforms.drive_wav",
            waveform_spec_from_raw({"style": "const", "length": 1.0}),
        )
    ctrl.context_control.use_point(PointUseRequest("point"))
    dialog = InspectDialog(ctrl, ctrl.get_bus())
    try:
        yield dialog, entry
    finally:
        dialog.close()
        background.quiesce()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("trigger", ["button", "key"])
def test_inspect_dialog_ml_delete_button_confirms_and_delete_key_is_direct(
    qapp, monkeypatch, native_delete_inspector, kind, trigger
):
    dialog, entry = native_delete_inspector
    before = entry.module_library("point").snapshot()
    source = "R1.readout.readout_rf" if kind == "module" else "Q1.waveforms.drive_wav"
    group = "modules" if kind == "module" else "waveforms"
    tree = next(
        tree
        for tree in dialog.findChildren(QTreeWidget)
        if tree.findItems(group, Qt.MatchFlag.MatchExactly)
    )
    errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda _parent, _title, message: errors.append(message)
    )
    tree.setCurrentItem(
        tree.findItems(source, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive)[
            0
        ]
    )
    button = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Delete"
    )
    assert button.isEnabled()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    button.click()
    assert entry.module_library("point").snapshot().document() == before.document()
    tabs = dialog.findChild(QTabWidget)
    assert tabs is not None
    tabs.setCurrentIndex(1)
    dialog.show()
    dialog.activateWindow()
    dialog.raise_()
    tree.setFocus()
    qapp.processEvents()
    exceptions: list[BaseException] = []
    monkeypatch.setattr(
        sys, "excepthook", lambda _type, error, _trace: exceptions.append(error)
    )
    if trigger == "button":
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        button.click()
    else:
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *a, **k: pytest.fail("tree Delete must not confirm"),
        )
        event = QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
        )
        QApplication.sendEvent(tree, event)
    qapp.processEvents()
    assert errors == [], f"Inspect reported a delete failure: {errors}"
    assert exceptions == [], f"Inspect delete raised: {exceptions}"
    persisted = entry.module_library("point").snapshot()
    keys = persisted.module_keys() if kind == "module" else persisted.waveform_keys()
    assert keys == ()
    if kind == "module":
        assert persisted.entry_state(
            "Q1.waveforms.drive_wav", kind="waveform"
        ) == before.entry_state("Q1.waveforms.drive_wav", kind="waveform")
    else:
        assert persisted.entry_state(
            "R1.readout.readout_rf", kind="module"
        ) == before.entry_state("R1.readout.readout_rf", kind="module")


def test_inspect_dialog_ml_delete_key_does_not_intercept_name_edit(qapp):
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    dialog._ml_name_edit.setFocus()
    dialog._ml_name_edit.setCursorPosition(0)
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Delete,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(dialog._ml_name_edit, event)
    assert ctrl.del_ml_module.call_count == 0
    assert dialog._ml_name_edit.text() == "1.readout.readout_rf"


def _catalog():
    from zcu_lab.templates import register_all_templates

    cat = TemplateCatalog()
    register_all_templates(cat)
    return cat


def test_create_dialog_populates_roles_and_creates(qapp):
    ctrl = _make_ctrl()
    ctrl.get_template_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: False

    dlg = _MlCreateDialog(ctrl)
    combo = dlg._template_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert any("Resonator probe" in t for t in labels)  # md-aware
    assert any("Blank: reset/bath" in t for t in labels)  # blank role
    assert any("Blank: drag" in t for t in labels)  # waveform-only blank shape

    # Pick the first role, give a name, create.
    dlg._template_combo.setCurrentIndex(0)
    dlg._name_edit.setText("my_entry")
    dlg._on_create()

    entry = dlg._template_combo.itemData(0)
    ctrl.create_from_template.assert_called_once_with(
        entry.item_kind, entry.template_id, "my_entry"
    )


def _ctrl_with_catalog(existing: set[tuple[str, str]] | None = None) -> MagicMock:
    """Mock controller exposing the real role catalog + a controllable ml."""
    have = existing or set()
    ctrl = _make_ctrl()
    ctrl.get_template_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: (kind, name) in have
    return ctrl


def _suggested_for(dlg: _MlCreateDialog, role_id: str) -> str:
    """Drive the combo to ``role_id`` and read back the suggested name."""
    for i in range(dlg._template_combo.count()):
        entry = dlg._template_combo.itemData(i)
        if entry is not None and entry.template_id == role_id:
            dlg._template_combo.setCurrentIndex(i)
            return dlg._name_edit.text()
    raise AssertionError(f"role {role_id!r} not in combo")


def test_create_dialog_prefills_default_name(qapp):
    ctrl = _ctrl_with_catalog()
    dlg = _MlCreateDialog(ctrl)
    # The first dropdown entry is res_probe -> readout_rf (see registry).
    first = dlg._template_combo.itemData(0)
    assert first.template_id == "res_probe"
    assert dlg._name_edit.text() == "readout_rf"


def test_create_dialog_blank_role_suggests_empty(qapp):
    ctrl = _ctrl_with_catalog()
    dlg = _MlCreateDialog(ctrl)
    # Blank roles carry no default_name -> the field stays empty.
    assert _suggested_for(dlg, "pulse:blank") == ""


def test_create_dialog_dedups_existing_name(qapp):
    # readout_rf already taken -> suggestion bumps to readout_rf_2.
    ctrl = _ctrl_with_catalog(existing={("module", "readout_rf")})
    dlg = _MlCreateDialog(ctrl)
    assert dlg._name_edit.text() == "readout_rf_2"


def test_create_dialog_role_switch_updates_suggestion(qapp):
    ctrl = _ctrl_with_catalog()
    dlg = _MlCreateDialog(ctrl)
    assert _suggested_for(dlg, "bath_reset") == "reset_bath"
    assert _suggested_for(dlg, "pi_pulse") == "pi_amp"


def test_create_dialog_role_switch_keeps_user_typed_name(qapp):
    ctrl = _ctrl_with_catalog()
    dlg = _MlCreateDialog(ctrl)
    # Simulate a user keystroke: textEdited carries the edited-by-hand semantics.
    dlg._name_edit.setText("my_custom")
    dlg._name_edit.textEdited.emit("my_custom")
    # Switching role must not clobber the hand-typed name.
    assert _suggested_for(dlg, "bath_reset") == "my_custom"


def test_create_dialog_records_created_on_success(qapp):
    ctrl = _ctrl_with_catalog()
    dlg = _MlCreateDialog(ctrl)
    dlg._template_combo.setCurrentIndex(0)
    dlg._name_edit.setText("my_entry")
    dlg._on_create()
    assert dlg.created == ("module", "my_entry")


def test_inspect_create_selects_embedded_editor_without_blocking(qapp, monkeypatch):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    ctrl.get_template_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: False
    dialog = InspectDialog(ctrl, MagicMock())

    class FakeCreateDialog:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            class DummySignal:
                def connect(self, cb: Any) -> None:
                    self.cb = cb

                def emit(self) -> None:
                    if hasattr(self, "cb"):
                        self.cb(None)

            self.finished = DummySignal()
            self.created: tuple[str, str] | None = ("module", "R1.readout.readout_rf")

        def setAttribute(self, *args: Any) -> None:
            pass

        def open(self) -> None:
            self.finished.emit()

    monkeypatch.setattr(inspect_dialog, "_MlCreateDialog", FakeCreateDialog)
    dialog._on_create_clicked()

    assert dialog._new_ml_dialog is None
    assert dialog._ml_selected_data == ("modules", "R1.readout.readout_rf")
    assert dialog._ml_name_edit.text() == "R1.readout.readout_rf"
    assert dialog._ml_status_label.text() == "Saved"


def test_inspect_create_closes_previous_editor_before_selecting_new_entry(
    qapp, monkeypatch
):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = InspectDialog(ctrl, MagicMock())
    _select_ml_child(dialog, 0)
    old_editor_id = dialog._ml_editor_id
    assert old_editor_id is not None

    class FakeCreateDialog:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            class DummySignal:
                def connect(self, cb: Any) -> None:
                    self.cb = cb

                def emit(self) -> None:
                    if hasattr(self, "cb"):
                        self.cb(None)

            self.finished = DummySignal()
            self.created: tuple[str, str] | None = ("module", "R1.readout.new_mod")

        def setAttribute(self, *args: Any) -> None:
            pass

        def open(self) -> None:
            current = ctrl.read_context()
            ctrl.read_context.return_value = ParameterContext(
                point=current.document(),
                library=current.library.replaced_module(
                    "R1.readout.new_mod",
                    current.library.module_spec("R1.readout.readout_rf"),
                ),
                target=current.target,
                roles=None,
            )
            self.finished.emit()

    monkeypatch.setattr(inspect_dialog, "_MlCreateDialog", FakeCreateDialog)
    dialog._on_create_clicked()

    assert dialog._ml_selected_data == ("modules", "R1.readout.new_mod")
    assert dialog._ml_name_edit.text() == "R1.readout.new_mod"
    assert dialog._ml_status_label.text() == "Saved"
    assert dialog._ml_editor_id != old_editor_id


def test_inspect_create_cancelled_does_not_select_entry(qapp, monkeypatch):
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())

    class FakeCreateDialog:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            class DummySignal:
                def connect(self, cb: Any) -> None:
                    self.cb = cb

                def emit(self) -> None:
                    if hasattr(self, "cb"):
                        self.cb(None)

            self.finished = DummySignal()
            self.created: tuple[str, str] | None = None

        def setAttribute(self, *args: Any) -> None:
            pass

        def open(self) -> None:
            self.finished.emit()

    monkeypatch.setattr(inspect_dialog, "_MlCreateDialog", FakeCreateDialog)
    dialog._on_create_clicked()

    assert dialog._ml_selected_data is None
    assert dialog._ml_editor_id is None


def test_create_dialog_rejects_empty_name(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    ctrl = _make_ctrl()
    ctrl.get_template_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: False

    dlg = _MlCreateDialog(ctrl)
    dlg._name_edit.setText("   ")
    dlg._on_create()

    ctrl.create_from_template.assert_not_called()


def test_inspect_ml_toolbar_has_single_create_button(qapp):
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())
    # The three old buttons (Add Module / Add Waveform / From template) collapse
    # into one Create entry; no add-by-discriminator path remains.
    assert hasattr(dialog, "_create_btn")
    assert not hasattr(dialog, "_add_mod_btn")
    assert not hasattr(dialog, "_add_wav_btn")
