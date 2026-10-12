"""Measure Setup across launch, screenshots, toolbar reopen and entry/point tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton, QSpinBox
from zcu_tools.gui.app.measure.services.persistence_types import (
    AppPersistedState,
    PersistedStartup,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import SavedPoint
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
        entry_name="chipR",
        ip="10.1.2.3",
        port=4321,
    )
)


def _field_texts(dialog: SetupDialog) -> set[str]:
    return {edit.text() for edit in dialog.findChildren(QLineEdit)} | {
        combo.currentText() for combo in dialog.findChildren(QComboBox)
    }


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
        app.invoke("rpc_call", {"method": "control.release", "params": {}})
        (dialog,) = visible_setup_dialogs(app.window)
        assert {"chipR", "10.1.2.3"} <= _field_texts(dialog)
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

        shot = Path(
            app.invoke(
                "rpc_call",
                {
                    "method": "dialog.screenshot",
                    "params": {
                        "name": "setup",
                        "out_path": str(tmp_path / "setup.png"),
                    },
                },
            )["saved_to"]
        )
        assert shot.read_bytes().startswith(b"\x89PNG")
        assert (QApplication.activeWindow(), QApplication.focusWidget()) == before
        assert visible_setup_dialogs(app.window) == [dialog]

        click_toolbar_setup(app.window)
        qapp.processEvents()
        assert visible_setup_dialogs(app.window) == [dialog]

        reopened = _reopen_setup(app, qapp)
        assert {"chipR", "10.1.2.3"} <= _field_texts(reopened)
        assert not app.controller.has_project()
        assert not app.controller.has_soc()


@pytest.mark.uses_wall_clock
def test_entry_point_commands_show_in_reopened_setup_and_failures_keep_the_context(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        control = app.controller.context_control
        entry = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "chip-a"}}
        )
        assert {"chip-a"} <= _field_texts(_reopen_setup(app, qapp))

        point = app.invoke(
            "rpc_call", {"method": "point.new", "params": {"label": "q1"}}
        )
        target = SavedPoint(entry_id=entry["entry_id"], point=point["point"])
        assert control.read_context().target == target

        # Selecting the same point keeps the selected resource identity.
        app.invoke("rpc_call", {"method": "point.use", "params": {"label": "q1"}})
        assert control.read_context().target == target
        assert {"chip-a"} <= _field_texts(_reopen_setup(app, qapp))

        # A missing entry leaves the entry, selected point and Setup unchanged.
        binding = control.get_source_binding()
        with pytest.raises(GuiRpcError):
            app.invoke(
                "rpc_call", {"method": "entry.open", "params": {"name": "chip-b"}}
            )
        assert (
            app.invoke("rpc_call", {"method": "entry.info"})["entry"]["name"]
            == "chip-a"
        )
        assert control.read_context().target == target
        assert control.get_source_binding() == binding
        assert "chip-b" not in _field_texts(_reopen_setup(app, qapp))

        # A new entry publishes metadata without selecting a point; Setup shows its name.
        changed = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "chip-b"}}
        )
        assert changed["name"] == "chip-b"
        info = control.get_entry_info()
        assert info is not None and info.entry_id == changed["entry_id"]
        with pytest.raises(FailedPreconditionError) as caught:
            control.read_context()
        assert caught.value.reason_code == "no_context"
        assert {"chip-b"} <= _field_texts(_reopen_setup(app, qapp))
