"""measure-gui Setup across launch, MCP screenshot, toolbar reopen and project tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLineEdit, QPushButton, QSpinBox
from zcu_tools.gui.app.measure.services.persistence_types import (
    AppPersistedState,
    PersistedStartup,
)
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui.app.measure._app_launch import (
    LaunchedMeasureApp,
    click_toolbar_setup,
    launched_measure_app,
    visible_setup_dialogs,
)

_SAVED = AppPersistedState(
    startup=PersistedStartup(
        chip_name="chipR",
        qub_name="qubR",
        res_name="resR",
        ip="10.1.2.3",
        port=4321,
    )
)


def _field_texts(dialog: SetupDialog) -> set[str]:
    return {edit.text() for edit in dialog.findChildren(QLineEdit)}


def _port_values(dialog: SetupDialog) -> list[int]:
    return [spin.value() for spin in dialog.findChildren(QSpinBox)]


def _button(dialog: SetupDialog, text: str) -> QPushButton:
    return next(
        button for button in dialog.findChildren(QPushButton) if button.text() == text
    )


def _reopen_setup(app: LaunchedMeasureApp, qapp: QApplication) -> SetupDialog:
    """Close the open Setup and open it again from the toolbar."""
    (current,) = visible_setup_dialogs(app.window)
    current.reject()
    qapp.processEvents()
    assert visible_setup_dialogs(app.window) == []
    click_toolbar_setup(app.window)
    qapp.processEvents()
    (reopened,) = visible_setup_dialogs(app.window)
    assert reopened is not current
    return reopened


@pytest.mark.uses_wall_clock
def test_restored_settings_prefill_one_setup_that_survives_screenshot_and_toolbar(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path, restore=_SAVED) as app:
        (dialog,) = visible_setup_dialogs(app.window)
        assert {"chipR", "qubR", "resR", "10.1.2.3"} <= _field_texts(dialog)
        assert 4321 in _port_values(dialog)
        # Restoring only prefills: nothing is applied and nothing connects.
        assert not app.controller.has_project()
        assert not app.controller.has_soc()

        connect = _button(dialog, "Connect")
        dialog.activateWindow()
        connect.setFocus()
        qapp.processEvents()
        before = (QApplication.activeWindow(), QApplication.focusWidget())
        assert before[1] is connect

        shot = Path(app.invoke("screenshot", {"target": "setup"})["path"])
        assert shot.read_bytes().startswith(b"\x89PNG")
        assert (QApplication.activeWindow(), QApplication.focusWidget()) == before
        assert visible_setup_dialogs(app.window) == [dialog]

        click_toolbar_setup(app.window)
        qapp.processEvents()
        assert visible_setup_dialogs(app.window) == [dialog]

        reopened = _reopen_setup(app, qapp)
        assert {"chipR", "qubR", "resR", "10.1.2.3"} <= _field_texts(reopened)
        assert not app.controller.has_project()
        assert not app.controller.has_soc()


@pytest.mark.uses_wall_clock
def test_project_tool_changes_show_in_reopened_setup_and_failures_keep_the_context(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        ctrl = app.controller

        app.invoke("project", {"chip": "chip-a", "qubit": "q1", "resonator": "res"})
        scope_a = ctrl.setup_control.get_setup_preferences().scope_id
        assert {"chip-a", "q1", "res"} <= _field_texts(_reopen_setup(app, qapp))

        label = app.invoke("context_create", {})["label"]
        assert ctrl.get_active_context_label() == label

        # An unchanged project is a no-op: the selected context stays selected.
        app.invoke("project", {"chip": "chip-a"})
        assert ctrl.get_active_context_label() == label
        assert {"chip-a", "q1", "res"} <= _field_texts(_reopen_setup(app, qapp))

        # A rejected update changes neither the project, the context, nor Setup.
        with pytest.raises(GuiRpcError) as failure:
            app.invoke("project", {"chip": "chip-b", "scope": scope_a})
        assert failure.value.reason == "scope_identity_mismatch"
        assert app.invoke("project", {})["chip"] == "chip-a"
        assert ctrl.get_active_context_label() == label
        assert "chip-b" not in _field_texts(_reopen_setup(app, qapp))

        # A real change deactivates the context and Setup shows the new project.
        changed = app.invoke("project", {"chip": "chip-b"})
        assert changed["chip"] == "chip-b"
        assert ctrl.get_active_context_label() is None
        assert {"chip-b", "q1", "res"} <= _field_texts(_reopen_setup(app, qapp))
