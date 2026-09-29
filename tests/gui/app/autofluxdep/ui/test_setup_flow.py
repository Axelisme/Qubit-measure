"""autofluxdep-gui Setup across restore, explicit Apply/Connect, run guards and re-save."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QCheckBox, QLineEdit, QPushButton, QSpinBox
from zcu_tools.gui.app.autofluxdep.app import AutoFluxDepGuiBehavior, build_core
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.services import create_persistence_caretaker
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow
from zcu_tools.gui.runtime import GuiAssembly
from zcu_tools.gui.session.services.connection import ConnectMockRequest
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    ProjectRequest,
)
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.resources.qubit_params import FluxDepFit, ParamsProject, QubitParams


@dataclass(frozen=True)
class _Relaunched:
    ctrl: Controller
    win: MainWindow
    root: Path
    cache_dir: Path


def _write_fluxdep_params(root: Path) -> None:
    params = QubitParams(root / "result" / "ChipA" / "Q1" / "params.json")
    params.ensure_project(ParamsProject("ChipA", "Q1", "R1"))
    params.set_fluxdep_fit(
        FluxDepFit(
            EJ=4.0,
            EC=1.0,
            EL=0.5,
            flux_half=0.25,
            flux_int=0.75,
            flux_period=0.8,
        )
    )


def _seed_saved_settings(root: Path, cache_dir: Path) -> None:
    ctrl = build_core(project_root=str(root))
    ctrl.attach_caretaker(create_persistence_caretaker(ctrl, cache_dir=cache_dir))
    ctrl.setup_control.remember_connection(
        ConnectionPreferences(ip="10.9.8.7", port=4321)
    )
    ctrl.setup_control.apply_project(ProjectRequest("ChipA", "Q1", "R1"))
    # The saved predictor model would come back on restore; drop it so the Apply
    # in the relaunched app is the only thing that can install one.
    ctrl.predictor_control.clear_predictor()
    ctrl.persist_all()
    ctrl.quiesce_background()


def _restore(ctrl: Controller, cache_dir: Path) -> None:
    ctrl.attach_caretaker(create_persistence_caretaker(ctrl, cache_dir=cache_dir))
    outcome = ctrl.restore_all()
    assert outcome is not None and outcome.load_error is None


@pytest.fixture
def relaunched(qapp: QApplication, tmp_path: Path) -> Iterator[_Relaunched]:
    """Launch the app the way the runtime does, over settings saved by an earlier run."""
    cache_dir = tmp_path / "cache"
    _write_fluxdep_params(tmp_path)
    _seed_saved_settings(tmp_path, cache_dir)

    ctrl = build_core(project_root=str(tmp_path))
    win = MainWindow(ctrl)
    _restore(ctrl, cache_dir)
    win.restore_workflow_view()
    behavior = AutoFluxDepGuiBehavior(project_root=str(tmp_path))
    behavior.after_show(GuiAssembly(controller=ctrl, window=win, control_adapter=None))
    qapp.processEvents()
    yield _Relaunched(ctrl, win, tmp_path, cache_dir)
    ctrl.quiesce_background()
    win.close()
    win.deleteLater()


def _setup_dialog(win: MainWindow) -> SetupDialog:
    (dialog,) = [d for d in win.findChildren(SetupDialog) if d.isVisible()]
    return dialog


def _field_texts(dialog: SetupDialog) -> set[str]:
    return {edit.text() for edit in dialog.findChildren(QLineEdit)}


def _button(dialog: SetupDialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton) if b.text() == text)


@pytest.mark.uses_wall_clock
def test_restored_settings_prefill_setup_until_the_user_applies_and_connects(
    relaunched: _Relaunched, qapp: QApplication
) -> None:
    ctrl, win = relaunched.ctrl, relaunched.win
    dialog = _setup_dialog(win)

    # Restore only prefilled the dialog: no project, no predictor, no connection.
    assert {"ChipA", "Q1", "R1", "10.9.8.7"} <= _field_texts(dialog)
    assert 4321 in [spin.value() for spin in dialog.findChildren(QSpinBox)]
    assert ctrl.state.project is None
    assert ctrl.predictor_control.get_predictor_info() is None
    assert ctrl.setup_control.get_soccfg() is None

    # Explicit Apply syncs ProjectInfo and installs the predictor from params.json.
    _button(dialog, "Apply project").click()
    qapp.processEvents()
    project = ctrl.state.project
    assert project is not None
    assert (project.chip_name, project.qub_name) == ("ChipA", "Q1")
    assert ctrl.predictor_control.get_predictor_info() is not None

    # Explicit Connect (offline MockSoc) is what connects.
    mock = next(
        box
        for box in dialog.findChildren(QCheckBox)
        if box.text().startswith("Use MockSoc")
    )
    mock.setChecked(True)
    _button(dialog, "Connect").click()
    deadline = time.monotonic() + 5.0
    while ctrl.setup_control.get_soccfg() is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert ctrl.setup_control.get_soccfg() is not None
    ctrl.quiesce_background()

    # The next launch restores the applied settings but neither applies nor connects.
    applied = ctrl.setup_control.get_setup_preferences()
    ctrl.persist_all()
    again = build_core(project_root=str(relaunched.root))
    _restore(again, relaunched.cache_dir)
    restored = again.setup_control.get_setup_preferences()
    assert (restored.chip_name, restored.qub_name, restored.res_name) == (
        "ChipA",
        "Q1",
        "R1",
    )
    assert restored.scope_id == applied.scope_id
    assert again.state.project is None
    assert again.setup_control.get_soccfg() is None
    again.quiesce_background()


@pytest.mark.parametrize("state", ["is_running", "is_paused"])
def test_setup_changes_are_refused_while_a_run_owns_the_session(
    relaunched: _Relaunched, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    ctrl = relaunched.ctrl
    monkeypatch.setattr(Controller, state, property(lambda _self: True))
    setup = ctrl.setup_control
    locked = "locked while a run is"

    with pytest.raises(RuntimeError, match=locked):
        setup.apply_project(ProjectRequest("ChipB", "Q1", "R1"))
    with pytest.raises(RuntimeError, match=locked):
        setup.remember_connection(ConnectionPreferences(ip="1.2.3.4", port=5))
    with pytest.raises(RuntimeError, match=locked):
        setup.start_connect(ConnectMockRequest())
    with pytest.raises(RuntimeError, match=locked):
        setup.new_context()

    # Reads stay available and nothing changed.
    prefs = setup.get_setup_preferences()
    assert (prefs.chip_name, prefs.ip, prefs.port) == ("ChipA", "10.9.8.7", 4321)
    assert ctrl.state.project is None
    assert setup.get_soccfg() is None
