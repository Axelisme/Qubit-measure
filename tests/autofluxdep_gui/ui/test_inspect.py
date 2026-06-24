"""Inspect-dialog integration for autofluxdep-gui (Phase 160c).

autofluxdep reuses the app-agnostic ``InspectDialogBase`` unchanged (the
measure-only ml create/modify path is excluded). These tests drive the *real*
controller + shared session ``ContextService`` + shared ``BaseEventBus`` so the
emit→auto-refresh path is exercised end to end, not mocked:

- the controller structurally satisfies ``SessionControllerPort`` (the surface
  the base dialog depends on),
- the dialog opens and shows the live md / ml,
- a md edit through the dialog writes back through the controller,
- an ml rename / delete routes through the controller,
- a md change fired on the bus auto-refreshes an already-open dialog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.session.events import MdChangedPayload
from zcu_tools.gui.session.ui.inspect_base import InspectDialogBase
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

if TYPE_CHECKING:
    from zcu_tools.gui.app.autofluxdep.controller import Controller
    from zcu_tools.gui.session.controller_port import SessionControllerPort


def _seed_context(ctrl: Controller) -> tuple[MetaDict, ModuleLibrary]:
    """Install a DRAFT context with one md key + one module + one waveform.

    Goes through the shared ``ContextService.set_startup_context`` so the context
    becomes non-EMPTY (``has_context()`` true) — md/ml writes then succeed and the
    real emit path fires, exactly as a configured GUI session.
    """
    md = MetaDict()
    md.r_f = 6000.0
    ml = ModuleLibrary()
    ml.register_module(
        readout_rf=ModuleCfgFactory.from_raw(
            {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_freq": 6000.0,
                "ro_length": 1.0,
                "trig_offset": 0.0,
            }
        )
    )
    ml.register_waveform(
        drive_wav=WaveformCfgFactory.from_raw({"style": "const", "length": 1.0})
    )
    ctrl._ctx_svc.set_startup_context(md, ml)
    return md, ml


@pytest.fixture
def ctrl(qapp):
    c = build_core()
    yield c
    # Quiesce the BackgroundRunner before GC (a pending queued delivery to a
    # GC'd runner segfaults a later test's event pump — see test_interaction).
    c._background_svc.quiesce()


@pytest.fixture
def dialog(qapp, ctrl):
    _seed_context(ctrl)
    dlg = InspectDialogBase(ctrl, ctrl.get_bus())
    yield dlg
    dlg.reject()  # fires ``finished`` → unsubscribes from the bus
    dlg.deleteLater()


# --- port conformance --------------------------------------------------------


def test_controller_conforms_to_session_port(ctrl):
    """The controller satisfies the inspect surface the base dialog depends on.

    A typed sink (``_accept``) binds the controller to ``SessionControllerPort``;
    pyright enforces structural conformance statically, and constructing the base
    dialog against it is the runtime proof the surface is callable.
    """

    def _accept(_port: SessionControllerPort) -> None:
        return None

    _accept(ctrl)  # static: must type-check as a SessionControllerPort

    for method in (
        "get_bus",
        "get_current_md",
        "get_current_ml",
        "coerce_md_value",
        "set_md_attr",
        "del_md_attr",
        "rename_ml_module",
        "rename_ml_waveform",
        "del_ml_module",
        "del_ml_waveform",
    ):
        assert callable(getattr(ctrl, method)), method


# --- open + display ----------------------------------------------------------


def test_dialog_shows_live_md_and_ml(dialog):
    # one md key (r_f) → one row
    assert dialog._md_table.rowCount() == 1
    # modules + waveforms groups → two top-level tree items
    assert dialog._ml_tree.topLevelItemCount() == 2


# --- md edit writeback -------------------------------------------------------


def test_md_edit_writes_back_through_controller(dialog, ctrl):
    dialog._edit_key.setText("r_f")
    dialog._edit_value.setText("6100.0")
    dialog._set_btn.click()

    # The real ContextService coerced + wrote the value into the live MetaDict.
    assert ctrl.get_current_md().r_f == 6100.0


def test_md_delete_removes_key_through_controller(dialog, ctrl):
    dialog._edit_key.setText("r_f")
    dialog._delete_btn.click()

    assert "r_f" not in dict(ctrl.get_current_md().items())


# --- ml rename / delete via controller ---------------------------------------


def test_ml_rename_routes_through_controller(dialog, ctrl, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))
    assert dialog._rename_ml_btn.isEnabled()

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("readout_v2", True))
    dialog._rename_ml_btn.click()

    ml = ctrl.get_current_ml()
    assert "readout_v2" in ml.modules
    assert "readout_rf" not in ml.modules


def test_ml_delete_routes_through_controller(dialog, ctrl, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    modules_item = dialog._ml_tree.topLevelItem(0)
    assert modules_item is not None
    dialog._ml_tree.setCurrentItem(modules_item.child(0))

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    dialog._del_ml_btn.click()

    assert "readout_rf" not in ctrl.get_current_ml().modules


# --- event-driven auto-refresh ----------------------------------------------


def test_bus_md_change_refreshes_open_dialog(dialog, ctrl):
    # Start: one md key.
    assert dialog._md_table.rowCount() == 1

    # A md mutation outside the dialog (e.g. another path) emits MD_CHANGED on the
    # shared bus; the open dialog is subscribed and must rebuild its table.
    md = ctrl.get_current_md()
    md.new_key = 1.23
    ctrl.get_bus().emit(MdChangedPayload(md=md))

    assert dialog._md_table.rowCount() == 2


def test_bus_subscriptions_cleaned_on_close(qapp, ctrl):
    _seed_context(ctrl)
    bus = ctrl.get_bus()
    dlg = InspectDialogBase(ctrl, bus)
    assert dlg._bus_subs_active

    # ``finished`` (the base's cleanup trigger) fires on reject/accept — the same
    # signal the modeless app dialog emits on close.
    dlg.reject()
    qapp.processEvents()

    assert not dlg._bus_subs_active
    # A later emit must not reach the closed dialog (no subscribers left for it).
    assert bus._subs.get(MdChangedPayload, []) == []
