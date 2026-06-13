"""Smoke tests for InspectDialog."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.app.main.ui import inspect_dialog
from zcu_tools.gui.app.main.ui.inspect_dialog import (
    InspectDialog,
    _MlCreateDialog,
    _MlModifyDialog,
)
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory


def _make_ml() -> ModuleLibrary:
    ml = ModuleLibrary()
    ml.modules["readout_rf"] = ModuleCfgFactory.from_raw(
        {
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 6000.0,
            "ro_length": 1.0,
            "trig_offset": 0.0,
        }
    )
    ml.waveforms["drive_wav"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 1.0}
    )
    return ml


def _make_ctrl_with_ml(ml: ModuleLibrary) -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_current_ml.return_value = ml
    ctrl.get_current_md.return_value = None
    ctrl.get_bus.return_value = MagicMock()
    _wire_cfg_editor(ctrl)
    return ctrl


def _wire_cfg_editor(ctrl: MagicMock) -> None:
    """Simulate the CfgEditorService open/open_seeded/commit/get_root contract.

    The modify dialog opens a *committable* session from the live ml
    (open_cfg_editor with from_name; ADR-0006) and commits via commit_cfg_editor;
    seeded sessions still exist for tab/writeback. Build a real SectionLiveField
    per open so attach() works, keyed by a fake editor_id, with owner→id discovery,
    commit (records last commit), and teardown.
    """
    from zcu_tools.gui.app.main.cfg_schemas import (
        module_cfg_to_value,
        waveform_cfg_to_value,
    )
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField

    ml = ctrl.get_current_ml.return_value
    roots: dict[str, SectionLiveField] = {}
    owner_to_id: dict[str, str] = {}
    counter = {"n": 0}

    def _register(spec, value, owner_key) -> str:
        if owner_key is not None and owner_key in owner_to_id:
            roots.pop(owner_to_id.pop(owner_key), None)
        counter["n"] += 1
        eid = f"editor-{counter['n']}"
        roots[eid] = SectionLiveField(spec, LiveModelEnv(ctrl=ctrl), value)
        if owner_key is not None:
            owner_to_id[owner_key] = eid
        return eid

    def _open_seeded(seed, *, gc=False, owner_key=None):
        return _register(seed.spec, seed.value, owner_key), []

    def _open(
        item_kind, *, discriminator=None, from_name=None, gc=True, owner_key=None
    ):
        # Load the existing entry's shape from the live ml (from_name path).
        store = ml.modules if item_kind == "module" else ml.waveforms
        to_value = (
            module_cfg_to_value if item_kind == "module" else waveform_cfg_to_value
        )
        spec, value = to_value(store[from_name])
        return _register(spec, value, owner_key), []

    def _commit(editor_id, name):
        ctrl.committed = (name, roots[editor_id])  # record for assertions
        roots.pop(editor_id, None)
        for owner, eid in list(owner_to_id.items()):
            if eid == editor_id:
                owner_to_id.pop(owner)

    ctrl.open_seeded_cfg_editor.side_effect = _open_seeded
    ctrl.open_cfg_editor.side_effect = _open
    ctrl.commit_cfg_editor.side_effect = _commit
    ctrl.get_cfg_editor_root.side_effect = lambda eid: roots[eid]
    ctrl.editor_id_for_owner.side_effect = lambda owner: owner_to_id.get(owner)
    ctrl.teardown_cfg_editor.side_effect = lambda eid: roots.pop(eid, None)


def test_inspect_dialog_init_and_refresh(qapp):
    ctrl = MagicMock()
    bus = MagicMock()

    # Setup mock returns
    mock_md = MagicMock()
    mock_md.items.return_value = {
        "scalar_int": 42,
        "scalar_float": 3.14,
        "nested": {"key": "value"},
    }.items()
    ctrl.get_current_md.return_value = mock_md

    ctrl.get_current_ml.return_value = _make_ml()

    dialog = InspectDialog(ctrl, bus)

    # Check subscriptions
    assert bus.subscribe.call_count >= 2

    # Check if MD dict was populated
    # The layout is flat, so there should be 3 rows (scalar_int, scalar_float, nested)
    assert dialog._md_table.rowCount() == 3

    # Check if ML was populated (tree widget)
    # The top level items should be "modules" and "waveforms"
    assert dialog._ml_tree.topLevelItemCount() == 2


def test_inspect_dialog_md_edit(qapp):
    ctrl = MagicMock()
    bus = MagicMock()

    mock_md = MagicMock()
    mock_md.items.return_value = {"key1": 10}.items()
    ctrl.get_current_md.return_value = mock_md
    ctrl.coerce_md_value.return_value = 20
    dialog = InspectDialog(ctrl, bus)

    # Select row to populate edit box
    dialog._md_table.selectRow(0)
    dialog._on_md_row_clicked(0, 0)

    dialog._edit_value.setText("20")
    dialog._set_btn.click()

    ctrl.coerce_md_value.assert_called_with("key1", "20")
    ctrl.set_md_attr.assert_called_with("key1", 20)


def test_inspect_dialog_md_set_success_clears_value_keeps_key(qapp):
    """After a successful Set, the value field is cleared; the key field is kept."""
    ctrl = MagicMock()
    bus = MagicMock()

    mock_md = MagicMock()
    mock_md.items.return_value = {"key1": 10}.items()
    ctrl.get_current_md.return_value = mock_md
    ctrl.coerce_md_value.return_value = 42
    dialog = InspectDialog(ctrl, bus)

    dialog._on_md_row_clicked(0, 0)  # fills key="key1", value="10"
    dialog._edit_value.setText("42")

    dialog._set_btn.click()

    # Value field must be empty after successful set (visual "committed" signal).
    assert dialog._edit_value.text() == ""
    # Key field must be preserved so the user can chain edits or use Delete.
    assert dialog._edit_key.text() == "key1"
    # Set button remains enabled because the key is still present.
    assert dialog._set_btn.isEnabled()


def test_inspect_dialog_md_set_failure_preserves_value(qapp, monkeypatch):
    """When coerce_md_value raises MdValueError, the value field is NOT cleared."""
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.session.services.context import MdValueError

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

    ctrl = MagicMock()
    bus = MagicMock()

    mock_md = MagicMock()
    mock_md.items.return_value = {"key1": 10}.items()
    ctrl.get_current_md.return_value = mock_md
    ctrl.coerce_md_value.side_effect = MdValueError("bad value")
    dialog = InspectDialog(ctrl, bus)

    dialog._on_md_row_clicked(0, 0)  # fills key="key1", value="10"
    dialog._edit_value.setText("not_a_valid_value")

    dialog._set_btn.click()

    # coerce raised: set_md_attr must NOT have been called.
    ctrl.set_md_attr.assert_not_called()
    # Value field must still contain the user's input so they can fix it.
    assert dialog._edit_value.text() == "not_a_valid_value"
    # Key field must also be untouched.
    assert dialog._edit_key.text() == "key1"


def test_inspect_dialog_ml_delete(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    ctrl = MagicMock()
    bus = MagicMock()

    ctrl.get_current_ml.return_value = _make_ml()

    dialog = InspectDialog(ctrl, bus)

    # By default, delete button should be disabled
    assert not dialog._del_ml_btn.isEnabled()

    # Expand top-level item and select child
    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))

    # Delete button should be enabled now
    assert dialog._del_ml_btn.isEnabled()

    # Mock QMessageBox.question to return Yes
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )

    # Click delete
    dialog._del_ml_btn.click()

    ctrl.del_ml_module.assert_called_with("readout_rf")


def test_inspect_dialog_ml_delete_no(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    ctrl = MagicMock()
    bus = MagicMock()

    ml = ModuleLibrary()
    ml.modules["readout_rf"] = ModuleCfgFactory.from_raw(
        {
            "type": "readout/direct",
            "ro_ch": 0,
            "ro_freq": 6000.0,
            "ro_length": 1.0,
            "trig_offset": 0.0,
        }
    )
    ctrl.get_current_ml.return_value = ml

    dialog = InspectDialog(ctrl, bus)
    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))

    # Mock QMessageBox.question to return No
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No
    )

    dialog._del_ml_btn.click()

    ctrl.del_ml_module.assert_not_called()


def test_inspect_dialog_ml_modify_enabled_only_for_children(qapp):
    ctrl = _make_ctrl_with_ml(_make_ml())
    bus = MagicMock()
    dialog = InspectDialog(ctrl, bus)

    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None

    dialog._ml_tree.setCurrentItem(modules_item)
    assert not dialog._modify_ml_btn.isEnabled()

    dialog._ml_tree.setCurrentItem(modules_item.child(0))
    assert dialog._modify_ml_btn.isEnabled()


def test_modify_dialog_module_fixed_shape_saves_same_name_and_type(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = _MlModifyDialog(
        ctrl, "module", name="readout_rf", cfg=ml.modules["readout_rf"]
    )

    # No type combo: modify never changes shape (the type is read-only).
    assert not hasattr(dialog, "_type_combo")
    assert dialog._save_btn.isEnabled()

    dialog._save_btn.click()

    # ADR-0006: save commits the session via the single write authority.
    name, _root = ctrl.committed
    assert name == "readout_rf"
    dialog.clear()


def test_modify_dialog_waveform_fixed_shape(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = _MlModifyDialog(
        ctrl, "waveform", name="drive_wav", cfg=ml.waveforms["drive_wav"]
    )

    assert not hasattr(dialog, "_type_combo")
    assert dialog._save_btn.isEnabled()

    dialog._save_btn.click()

    name, _root = ctrl.committed
    assert name == "drive_wav"
    dialog.clear()


def test_inspect_dialog_modify_clears_form_widget_after_exec(qapp, monkeypatch):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    bus = MagicMock()
    dialog = InspectDialog(ctrl, bus)
    clear_calls: list[str] = []

    class FakeDialog:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            class DummySignal:
                def connect(self, cb: Any) -> None:
                    self.cb = cb

                def emit(self) -> None:
                    if hasattr(self, "cb"):
                        self.cb(None)

            self.finished = DummySignal()

        def setAttribute(self, *args: Any) -> None:
            pass

        def open(self) -> None:
            self.finished.emit()

        def clear(self) -> None:
            clear_calls.append("clear")

    monkeypatch.setattr(inspect_dialog, "_MlModifyDialog", FakeDialog)
    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))

    dialog._modify_ml_btn.click()

    assert clear_calls == ["clear"]


def _catalog():
    from zcu_tools.experiment.v2_gui.registry import register_all_roles
    from zcu_tools.gui.app.main.role_catalog import RoleCatalog

    cat = RoleCatalog()
    register_all_roles(cat)
    return cat


def test_create_dialog_populates_roles_and_creates(qapp):
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: False

    dlg = _MlCreateDialog(ctrl)
    # Combo lists role labels (md-aware + blank), not raw type strings.
    labels = [dlg._role_combo.itemText(i) for i in range(dlg._role_combo.count())]
    assert any("Resonator probe" in t for t in labels)  # md-aware
    assert any("Blank: reset/bath" in t for t in labels)  # blank role
    assert any("Blank: drag" in t for t in labels)  # waveform-only blank shape

    # Pick the first role, give a name, create.
    dlg._role_combo.setCurrentIndex(0)
    dlg._name_edit.setText("my_entry")
    dlg._on_create()

    entry = dlg._role_combo.itemData(0)
    ctrl.create_from_role.assert_called_once_with(
        entry.item_kind, entry.role_id, "my_entry"
    )


def _ctrl_with_catalog(existing: set[tuple[str, str]] | None = None) -> MagicMock:
    """Mock controller exposing the real role catalog + a controllable ml."""
    have = existing or set()
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: (kind, name) in have
    return ctrl


def _suggested_for(dlg: _MlCreateDialog, role_id: str) -> str:
    """Drive the combo to ``role_id`` and read back the suggested name."""
    for i in range(dlg._role_combo.count()):
        entry = dlg._role_combo.itemData(i)
        if entry is not None and entry.role_id == role_id:
            dlg._role_combo.setCurrentIndex(i)
            return dlg._name_edit.text()
    raise AssertionError(f"role {role_id!r} not in combo")


def test_create_dialog_prefills_default_name(qapp):
    ctrl = _ctrl_with_catalog()
    dlg = _MlCreateDialog(ctrl)
    # The first dropdown entry is res_probe -> readout_rf (see registry).
    first = dlg._role_combo.itemData(0)
    assert first.role_id == "res_probe"
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
    dlg._role_combo.setCurrentIndex(0)
    dlg._name_edit.setText("my_entry")
    dlg._on_create()
    assert dlg.created == ("module", "my_entry")


def test_inspect_create_auto_opens_modify(qapp, monkeypatch):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    ctrl.get_role_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: False
    bus = MagicMock()
    dialog = InspectDialog(ctrl, bus)

    opened: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dialog, "_open_ml_modify", lambda group, name: opened.append((group, name))
    )

    class FakeCreateDialog:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            class DummySignal:
                def connect(self, cb: Any) -> None:
                    self.cb = cb

                def emit(self) -> None:
                    if hasattr(self, "cb"):
                        self.cb(None)

            self.finished = DummySignal()
            self.created: tuple[str, str] | None = ("module", "readout_rf")

        def setAttribute(self, *args: Any) -> None:
            pass

        def open(self) -> None:
            self.finished.emit()

    monkeypatch.setattr(inspect_dialog, "_MlCreateDialog", FakeCreateDialog)
    dialog._on_create_clicked()

    assert opened == [("modules", "readout_rf")]


def test_inspect_create_cancelled_does_not_open_modify(qapp, monkeypatch):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    bus = MagicMock()
    dialog = InspectDialog(ctrl, bus)

    opened: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dialog, "_open_ml_modify", lambda group, name: opened.append((group, name))
    )

    class FakeCreateDialog:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            class DummySignal:
                def connect(self, cb: Any) -> None:
                    self.cb = cb

                def emit(self) -> None:
                    if hasattr(self, "cb"):
                        self.cb(None)

            self.finished = DummySignal()
            self.created: tuple[str, str] | None = None  # cancelled

        def setAttribute(self, *args: Any) -> None:
            pass

        def open(self) -> None:
            self.finished.emit()

    monkeypatch.setattr(inspect_dialog, "_MlCreateDialog", FakeCreateDialog)
    dialog._on_create_clicked()

    assert opened == []


def test_create_dialog_rejects_empty_name(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value = _catalog()
    ctrl.has_ml_entry.side_effect = lambda kind, name: False

    dlg = _MlCreateDialog(ctrl)
    dlg._name_edit.setText("   ")
    dlg._on_create()

    ctrl.create_from_role.assert_not_called()


def test_inspect_ml_toolbar_has_single_create_button(qapp):
    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())
    # The three old buttons (Add Module / Add Waveform / From template) collapse
    # into one Create entry; no add-by-discriminator path remains.
    assert hasattr(dialog, "_create_btn")
    assert not hasattr(dialog, "_add_mod_btn")
    assert not hasattr(dialog, "_add_wav_btn")


def test_inspect_rename_button_calls_controller(qapp, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())

    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))
    assert dialog._rename_ml_btn.isEnabled()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("readout_v2", True))
    dialog._rename_ml_btn.click()
    ctrl.rename_ml_module.assert_called_once_with("readout_rf", "readout_v2")


def test_inspect_rename_cancelled_does_nothing(qapp, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    ctrl = _make_ctrl_with_ml(_make_ml())
    dialog = InspectDialog(ctrl, MagicMock())
    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    dialog._rename_ml_btn.click()
    ctrl.rename_ml_module.assert_not_called()
