"""Smoke tests for InspectDialog."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from zcu_tools.gui.ui import inspect_dialog
from zcu_tools.gui.ui.inspect_dialog import InspectDialog, _MlConfigDialog
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
    """Simulate the CfgEditorService open_seeded/get_root/teardown contract.

    The dialog now opens a service-owned model from its seed schema and attaches
    (ADR-0010). Build a real SectionLiveField per open so attach() works, keyed by
    a fake editor_id, with owner→id discovery and teardown.
    """
    from zcu_tools.gui.live_model import LiveModelEnv, SectionLiveField

    roots: dict[str, SectionLiveField] = {}
    owner_to_id: dict[str, str] = {}
    counter = {"n": 0}

    def _open_seeded(seed, *, gc=False, owner_key=None):
        if owner_key is not None and owner_key in owner_to_id:
            roots.pop(owner_to_id.pop(owner_key), None)
        counter["n"] += 1
        eid = f"editor-{counter['n']}"
        roots[eid] = SectionLiveField(seed.spec, LiveModelEnv(ctrl=ctrl), seed.value)
        if owner_key is not None:
            owner_to_id[owner_key] = eid
        return eid, []

    ctrl.open_seeded_cfg_editor.side_effect = _open_seeded
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


def test_ml_config_dialog_modify_module_keeps_name_and_changes_type(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = _MlConfigDialog(
        ctrl,
        "module",
        "modify",
        name="readout_rf",
        cfg=ml.modules["readout_rf"],
    )

    assert dialog._name_edit.isReadOnly()
    dialog._name_edit.setText("ignored_name")
    dialog._type_combo.setCurrentText("reset/none")
    assert dialog._save_btn.isEnabled()

    dialog._save_btn.click()

    name, raw = ctrl.set_ml_module_from_raw.call_args.args
    assert name == "readout_rf"
    assert raw["type"] == "reset/none"
    dialog.clear()


def test_ml_config_dialog_modify_waveform_keeps_name_and_changes_style(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = _MlConfigDialog(
        ctrl,
        "waveform",
        "modify",
        name="drive_wav",
        cfg=ml.waveforms["drive_wav"],
    )

    assert dialog._name_edit.isReadOnly()
    dialog._name_edit.setText("ignored_name")
    dialog._type_combo.setCurrentText("cosine")
    assert dialog._save_btn.isEnabled()

    dialog._save_btn.click()

    name, raw = ctrl.set_ml_waveform_from_raw.call_args.args
    assert name == "drive_wav"
    assert raw["style"] == "cosine"
    dialog.clear()


def test_ml_config_dialog_style_change_rebuilds_schema_and_validity(qapp):
    ml = _make_ml()
    ctrl = _make_ctrl_with_ml(ml)
    dialog = _MlConfigDialog(
        ctrl,
        "waveform",
        "modify",
        name="drive_wav",
        cfg=ml.waveforms["drive_wav"],
    )

    assert dialog._save_btn.isEnabled()
    dialog._type_combo.setCurrentText("arb")
    assert not dialog._save_btn.isEnabled()

    dialog._type_combo.setCurrentText("const")
    assert dialog._save_btn.isEnabled()
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

    monkeypatch.setattr(inspect_dialog, "_MlConfigDialog", FakeDialog)
    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))

    dialog._modify_ml_btn.click()

    assert clear_calls == ["clear"]


def _catalog():
    from zcu_tools.experiment.v2_gui.role_catalog_registry import register_all_roles
    from zcu_tools.gui.role_catalog import RoleCatalog

    cat = RoleCatalog()
    register_all_roles(cat)
    return cat


def test_template_dialog_populates_roles_and_creates(qapp):
    from zcu_tools.gui.ui.inspect_dialog import _MlTemplateDialog

    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value = _catalog()

    dlg = _MlTemplateDialog(ctrl)
    # Combo lists role labels (modules then waveforms), not type strings.
    labels = [dlg._role_combo.itemText(i) for i in range(dlg._role_combo.count())]
    assert any("Resonator probe" in t for t in labels)
    assert any("waveform" in t for t in labels)

    # Pick the first role, give a name, create.
    dlg._role_combo.setCurrentIndex(0)
    dlg._name_edit.setText("my_entry")
    dlg._on_create()

    entry = dlg._role_combo.itemData(0)
    ctrl.create_from_role.assert_called_once_with(
        entry.item_kind, entry.role_id, "my_entry"
    )


def test_template_dialog_rejects_empty_name(qapp, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    from zcu_tools.gui.ui.inspect_dialog import _MlTemplateDialog

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    ctrl = MagicMock()
    ctrl.get_role_catalog.return_value = _catalog()

    dlg = _MlTemplateDialog(ctrl)
    dlg._name_edit.setText("   ")
    dlg._on_create()

    ctrl.create_from_role.assert_not_called()
