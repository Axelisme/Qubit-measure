"""Measure Setup across launch, screenshots, toolbar reopen and entry/point tools."""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
)
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.services.persistence_types import (
    AppPersistedState,
    PersistedStartup,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog
from zcu_tools.mcp.core.rpc import GuiInvocationError
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._user_input import ComponentDriver
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
    """Close any open Setup and open a fresh instance from the toolbar."""
    previous = visible_setup_dialogs(app.window)
    for current in previous:
        current.reject()
    qapp.processEvents()
    assert visible_setup_dialogs(app.window) == []
    click_toolbar_setup(app.window)
    qapp.processEvents()
    (reopened,) = visible_setup_dialogs(app.window)
    assert reopened not in previous
    return reopened


@pytest.mark.uses_wall_clock
def test_restored_settings_prefill_one_setup_that_survives_screenshot_and_toolbar(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path, restore=_SAVED) as app:
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
def test_startup_setup_allows_owner_connect_and_tracks_control(qapp, tmp_path):
    with launched_measure_app(qapp, tmp_path) as app:
        (setup,) = visible_setup_dialogs(app.window)
        assert not setup.isModal(), "Startup Setup must not block owner acquisition"
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
        assert not _button(setup, "New entry…").isEnabled()
        app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "agent-entry"}}
        )
        assert setup.isVisible()
        assert "agent-entry" in _field_texts(setup)
        app.invoke("rpc_call", {"method": "control.release"})
        assert _button(setup, "New entry…").isEnabled()
        assert _button(setup, "Connect").isEnabled()


@pytest.mark.uses_wall_clock
def test_entry_point_commands_show_in_reopened_setup_and_failures_keep_the_context(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        control = app.controller.context_control
        (startup,) = visible_setup_dialogs(app.window)
        startup.reject()
        qapp.processEvents()
        # Identity is fixed per socket; acquire owner only after closing Local Setup.
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})

        def check_setup_entry(name: str) -> None:
            app.invoke("rpc_call", {"method": "control.release"})
            dialog = _reopen_setup(app, qapp)
            assert name in _field_texts(dialog)
            dialog.reject()
            qapp.processEvents()
            app.invoke("rpc_call", {"method": "control.acquire"})

        entry = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "chip-a"}}
        )
        check_setup_entry("chip-a")

        point = app.invoke(
            "rpc_call", {"method": "point.new", "params": {"label": "q1"}}
        )
        target = SavedPoint(entry_id=entry["entry_id"], point=point["point"])
        assert control.read_context().target == target

        # Selecting the same point keeps the selected resource identity.
        app.invoke("rpc_call", {"method": "point.use", "params": {"label": "q1"}})
        assert control.read_context().target == target
        check_setup_entry("chip-a")

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
        app.invoke("rpc_call", {"method": "control.release"})
        failed_setup = _reopen_setup(app, qapp)
        assert "chip-b" not in _field_texts(failed_setup)
        failed_setup.reject()
        qapp.processEvents()
        app.invoke("rpc_call", {"method": "control.acquire"})

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
        check_setup_entry("chip-b")


@pytest.mark.uses_wall_clock
def test_local_true_modal_still_refuses_owner_and_allows_retry(qapp, tmp_path):
    with launched_measure_app(qapp, tmp_path) as app:
        app.window.close_dialog(DialogName.SETUP)
        dialog = QDialog(app.window)
        try:
            dialog.open()
            with pytest.raises(GuiInvocationError, match="local_busy"):
                app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
            dialog.reject()
            app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
            created = app.invoke(
                "rpc_call", {"method": "entry.new", "params": {"name": "after-modal"}}
            )
            assert created["name"] == "after-modal"
        finally:
            dialog.reject()


@pytest.mark.uses_wall_clock
def test_error_notification_stays_visible_without_blocking_owner(qapp, tmp_path):
    with launched_measure_app(qapp, tmp_path) as app:
        app.window.show_error_dialog("Synthetic failure", "Agent operation failed")
        qapp.processEvents()
        box = app.window.findChild(QMessageBox)
        assert box is not None and box.isVisible()
        assert box.text() == "Agent operation failed"
        assert not box.isModal()
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
        created = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "after-error"}}
        )
        assert created["name"] == "after-error"
        assert box.isVisible()
        box.accept()


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize(
    "name", [DialogName.SETUP, DialogName.DEVICE, DialogName.INSPECT]
)
def test_observation_controls_recover_via_real_user_input(qapp, tmp_path, name):
    with launched_measure_app(qapp, tmp_path) as app:
        driver = ComponentDriver(app.window)
        captions = {
            DialogName.SETUP: ("Setup…", "New entry…"),
            DialogName.DEVICE: ("Devices…", "Add Device"),
            DialogName.INSPECT: ("Inspect…", "New"),
        }
        opener, action = captions[name]
        driver.screenshot(tmp_path / f"{name.value}-01-local.png")
        # Isolate toolbar reopening from the separately tested startup handoff.
        driver.click("Close")
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
        driver.click(opener)
        assert driver.dialogs(), "Remote must allow opening the observation window"
        if name is DialogName.INSPECT:
            driver.select_tab("Modules")
        assert not driver.observe(action).enabled
        driver.screenshot(tmp_path / f"{name.value}-02-remote.png")
        created = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "observed-entry"}}
        )
        assert created["name"] == "observed-entry"
        driver.screenshot(tmp_path / f"{name.value}-03-agent-command.png")
        driver.click(object_name="control_indicator")
        assert driver.observe(action).enabled
        assert driver.observe(object_name="control_indicator").text == "Local locked"
        driver.screenshot(tmp_path / f"{name.value}-04-local-reclaim.png")
        driver.click("Close")
        assert driver.dialogs() == ()
        driver.screenshot(tmp_path / f"{name.value}-05-closed.png")
        driver.click(opener)
        if name is DialogName.INSPECT:
            driver.select_tab("Modules")
        assert driver.observe(action).enabled
        driver.screenshot(tmp_path / f"{name.value}-06-reopened.png")
        driver.click("Close")
