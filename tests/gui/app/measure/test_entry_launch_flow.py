"""Measure Entry across launch, screenshots, toolbar reopen and entry/point tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
)
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.session.context_types import DocumentSetRef, EntryRequest
from zcu_tools.gui.session.ui.entry_dialog import EntryDialog
from zcu_tools.mcp.core.rpc import GuiInvocationError
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.gui._context_reads import capture_focus
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._app_launch import (
    LaunchedMeasureApp,
    click_toolbar_entry,
    launched_measure_app,
    visible_entry_dialogs,
)


def _field_texts(dialog: EntryDialog) -> set[str]:
    return {edit.text() for edit in dialog.findChildren(QLineEdit)} | {
        combo.currentText() for combo in dialog.findChildren(QComboBox)
    }


def _button(dialog: EntryDialog, text: str) -> QPushButton:
    return next(
        button for button in dialog.findChildren(QPushButton) if button.text() == text
    )


def _reopen_entry(app: LaunchedMeasureApp, qapp: QApplication) -> EntryDialog:
    """Close any open Entry and open a fresh instance from the toolbar."""
    previous = visible_entry_dialogs(app.window)
    for current in previous:
        current.reject()
    qapp.processEvents()
    assert visible_entry_dialogs(app.window) == []
    click_toolbar_entry(app.window)
    qapp.processEvents()
    (reopened,) = visible_entry_dialogs(app.window)
    assert reopened not in previous
    return reopened


@pytest.mark.uses_wall_clock
def test_entry_survives_toolbar_reopen(qapp: QApplication, tmp_path: Path) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        (dialog,) = visible_entry_dialogs(app.window)
        initial_fields = _field_texts(dialog)
        # Opening Entry alone does not apply entry or connection changes.
        assert not app.controller.has_project()
        assert not app.controller.has_soc()

        open_entry = _button(dialog, "Open")
        dialog.activateWindow()
        open_entry.setFocus()
        qapp.processEvents()
        before = (QApplication.activeWindow(), QApplication.focusWidget())
        assert before[1] is open_entry

        click_toolbar_entry(app.window)
        qapp.processEvents()
        assert visible_entry_dialogs(app.window) == [dialog]

        reopened = _reopen_entry(app, qapp)
        assert _field_texts(reopened) == initial_fields
        assert not app.controller.has_project()
        assert not app.controller.has_soc()


@pytest.mark.uses_wall_clock
def test_startup_entry_allows_owner_connect_and_tracks_control(qapp, tmp_path):
    with launched_measure_app(qapp, tmp_path) as app:
        (entry,) = visible_entry_dialogs(app.window)
        assert not entry.isModal(), "Startup Entry must not block owner acquisition"
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
        assert not _button(entry, "New entry…").isEnabled()
        app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "agent-entry"}}
        )
        assert entry.isVisible()
        assert "agent-entry" in _field_texts(entry)
        app.invoke("rpc_call", {"method": "control.release"})
        assert _button(entry, "New entry…").isEnabled()
        assert _button(entry, "Open").isEnabled()


@pytest.mark.uses_wall_clock
def test_entry_point_commands_show_in_reopened_entry_and_failures_keep_the_context(
    qapp: QApplication, tmp_path: Path
) -> None:
    with launched_measure_app(qapp, tmp_path) as app:
        control = app.controller.context_control
        (startup,) = visible_entry_dialogs(app.window)
        startup.reject()
        qapp.processEvents()
        # Identity is fixed per socket; acquire owner only after closing Local Entry.
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})

        def check_entry_entry(name: str) -> None:
            app.invoke("rpc_call", {"method": "control.release"})
            dialog = _reopen_entry(app, qapp)
            assert name in _field_texts(dialog)
            dialog.reject()
            qapp.processEvents()
            app.invoke("rpc_call", {"method": "control.acquire"})

        entry = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "chip-a"}}
        )
        check_entry_entry("chip-a")

        point = app.invoke(
            "rpc_call", {"method": "point.new", "params": {"label": "q1"}}
        )
        target = DocumentSetRef(entry_name=entry["name"], point=point["point"])
        assert capture_focus(control).target == target

        # Selecting the same point keeps the selected resource identity.
        app.invoke("rpc_call", {"method": "point.use", "params": {"label": "q1"}})
        assert capture_focus(control).target == target
        check_entry_entry("chip-a")

        # A missing entry leaves the entry, selected point and Entry unchanged.
        binding = control.get_source_binding()
        with pytest.raises(GuiRpcError):
            app.invoke(
                "rpc_call", {"method": "entry.open", "params": {"name": "chip-b"}}
            )
        assert (
            app.invoke("rpc_call", {"method": "entry.info"})["entry"]["name"]
            == "chip-a"
        )
        assert capture_focus(control).target == target
        assert control.get_source_binding() == binding
        app.invoke("rpc_call", {"method": "control.release"})
        failed_entry = _reopen_entry(app, qapp)
        assert "chip-b" not in _field_texts(failed_entry)
        failed_entry.reject()
        qapp.processEvents()
        app.invoke("rpc_call", {"method": "control.acquire"})

        # A new entry activates Global without selecting a point; Entry shows its name.
        changed = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "chip-b"}}
        )
        assert changed["name"] == "chip-b"
        info = control.get_entry_info()
        assert info is not None and info.name == changed["name"]
        assert capture_focus(control).target == DocumentSetRef(entry_name="chip-b")
        check_entry_entry("chip-b")


@pytest.mark.uses_wall_clock
def test_local_true_modal_still_refuses_owner_and_allows_retry(qapp, tmp_path):
    with launched_measure_app(qapp, tmp_path) as app:
        app.window.close_dialog(DialogName.ENTRY)
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
    "name", [DialogName.ENTRY, DialogName.DEVICE, DialogName.INSPECT]
)
def test_observation_controls_recover_via_real_user_input(qapp, tmp_path, name):
    with launched_measure_app(qapp, tmp_path) as app:
        driver = ComponentDriver(app.window)
        captions = {
            DialogName.ENTRY: ("Entry…", "New entry…"),
            DialogName.DEVICE: ("Devices…", "Add Device"),
            DialogName.INSPECT: ("Inspect…", "New"),
        }
        opener, action = captions[name]
        evidence = Path(os.environ.get("ENTRY_TITLE_EVIDENCE", str(tmp_path)))
        if name is DialogName.ENTRY:
            assert driver.dialogs()[0].title == "Entry"
        driver.screenshot(evidence / f"{name.value}-01-local.png")
        # Isolate toolbar reopening from the separately tested startup handoff.
        driver.click("Close")
        app.invoke("connect", {"port": app.remote.port, "mode": "owner"})
        driver.click(opener)
        assert driver.dialogs(), "Remote must allow opening the observation window"
        if name is DialogName.INSPECT:
            driver.select_tab("Modules")
        assert not driver.observe(action).enabled
        driver.screenshot(evidence / f"{name.value}-02-remote.png")
        created = app.invoke(
            "rpc_call", {"method": "entry.new", "params": {"name": "observed-entry"}}
        )
        assert created["name"] == "observed-entry"
        driver.screenshot(evidence / f"{name.value}-03-agent-command.png")
        driver.click(object_name="control_indicator")
        assert driver.observe(action).enabled
        assert driver.observe(object_name="control_indicator").text == "Local locked"
        driver.screenshot(evidence / f"{name.value}-04-local-reclaim.png")
        driver.click("Close")
        assert driver.dialogs() == ()
        driver.screenshot(evidence / f"{name.value}-05-closed.png")
        driver.click(opener)
        if name is DialogName.INSPECT:
            driver.select_tab("Modules")
        assert driver.observe(action).enabled
        driver.screenshot(evidence / f"{name.value}-06-reopened.png")
        driver.click("Close")


@pytest.fixture
def entry_title_ui(qapp, tmp_path):
    with launched_measure_app(qapp, tmp_path) as app:
        app.controller.entry_control.new_entry(EntryRequest("entry"))
        yield ComponentDriver(app.window)


def test_entry_toolbar_opens_the_entry_window_via_real_user_input(
    entry_title_ui, tmp_path
):
    driver = entry_title_ui
    evidence = Path(os.environ.get("ENTRY_TITLE_EVIDENCE", str(tmp_path)))
    assert driver.dialogs()[0].title == "Entry"
    driver.screenshot(evidence / "entry-title-01-startup.png")
    driver.click("Close")
    driver.screenshot(evidence / "entry-title-02-toolbar.png")
    driver.click("Entry…")
    assert driver.dialogs()[0].title == "Entry"
    driver.screenshot(evidence / "entry-title-03-reopened.png")
    driver.enter_text(object_name="point_label", value="title-point", commit=False)
    driver.screenshot(evidence / "entry-title-04-point-label.png")
    driver.click("Create")
    assert driver.dialogs()[0].title == "Entry"
    assert [row.text for row in driver.list_rows()] == ["Global", "● title-point"]
    driver.screenshot(evidence / "entry-title-05-point-created.png")
    driver.select_list_row("Global")
    driver.screenshot(evidence / "entry-title-06-global-selected.png")
    driver.click("Switch to selected")
    assert driver.dialogs()[0].title == "Entry"
    assert [row.text for row in driver.list_rows()] == ["● Global", "title-point"]
    driver.screenshot(evidence / "entry-title-07-global-focused.png")
    driver.click("Close")
