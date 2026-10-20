"""Source toolbar SoC/Setup entry points follow the workflow run lifetime."""

import os
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.experiment.simulation import white_noise

from tests.gui._user_input import ComponentDriver

from ._window_support import WindowSession, bind_point


@pytest.fixture
def soc_session(window_session: WindowSession, qapp: QApplication) -> WindowSession:
    bind_point(window_session.harness, window_session.root)
    env = white_noise()
    window_session.harness.services.soc_connection.install_soc(env.soc, env.soccfg)
    window_session.window.resize(1500, 900)
    window_session.window.show()
    qapp.processEvents()
    return window_session


def test_toolbar_soc_and_setup_reopen_and_follow_run_lock(
    soc_session: WindowSession, tmp_path: Path
) -> None:
    ui = ComponentDriver(soc_session.window)
    shots = Path(os.environ.get("SOC_UI_EVIDENCE", str(tmp_path))) / "workflows"
    ui.screenshot(shots / "01-toolbar.png")
    assert any(w.text == "SoC…" for w in ui.visible().widgets), (
        "SoC toolbar entry missing"
    )
    ui.click("SoC…")
    assert ui.dialogs()[0].title == "SoC"
    assert ui.observe(object_name="soc_ip").visible
    assert ui.observe("Connect").enabled
    ui.screenshot(shots / "02-soc.png")
    ui.click("Close")
    assert ui.dialogs() == ()
    ui.screenshot(shots / "03-closed.png")
    ui.click("SoC…")
    assert ui.observe(object_name="soc_ip").visible
    ui.screenshot(shots / "04-reopened.png")
    ui.click("Close")
    ui.click("Setup…")
    assert ui.dialogs()[0].title == "Setup"
    assert "IP address:" not in ui.dialogs()[0].texts
    assert ui.observe("New entry…").enabled
    ui.screenshot(shots / "05-setup.png")
    ui.click("Prepare forms")
    ui.click("Start")
    assert not ui.observe("Start").enabled
    assert all(
        not ui.observe(caption).enabled for caption in ("New entry…", "Open", "Create")
    )
    assert ui.observe(object_name="working_points").enabled
    ui.select_list_row(ui.list_rows()[0].text)
    assert ui.list_rows()[0].selected
    assert ui.observe("Close").enabled
    ui.screenshot(shots / "06-running-setup.png")
    ui.click("Close")
    ui.click("SoC…")
    assert not ui.observe("Connect").enabled
    assert not ui.observe(object_name="soc_ip").enabled
    assert ui.observe("Close").enabled
    assert any(
        w.kind == "QPlainTextEdit" and w.enabled and bool(w.text)
        for w in ui.visible().widgets
    )
    ui.screenshot(shots / "07-running-soc.png")
    ui.click("Close")
    ui.click("SoC…")
    assert not ui.observe("Connect").enabled
    ui.screenshot(shots / "08-running-soc-reopened.png")
    # The queued synthetic producer finishes when this external executor delivers it.
    soc_session.harness.complete_segment()
    assert ui.observe("Start").enabled
    assert ui.observe("Connect").enabled
    ui.screenshot(shots / "09-unlocked-soc.png")
    ui.click("Close")
    ui.click("Setup…")
    assert ui.observe("New entry…").enabled
    ui.screenshot(shots / "10-unlocked-setup.png")
    ui.click("Close")


@pytest.mark.parametrize("initially_connected", [False, True])
def test_soc_reopen_synchronizes_changes_while_closed(
    window_session: WindowSession,
    qapp: QApplication,
    tmp_path: Path,
    initially_connected: bool,
) -> None:
    connection = window_session.harness.services.soc_connection
    env = white_noise()
    if initially_connected:
        connection.install_soc(env.soc, env.soccfg)
    window_session.window.resize(1500, 900)
    window_session.window.show()
    qapp.processEvents()
    ui = ComponentDriver(window_session.window)
    shots = (
        Path(os.environ.get("SOC_UI_EVIDENCE", str(tmp_path)))
        / f"workflows-reopen-{initially_connected}"
    )
    ui.click("SoC…")
    ui.screenshot(shots / "01-before-close.png")
    ui.click("Close")
    ui.screenshot(shots / "02-closed.png")
    # The actual session owner changes while this retained view has no subscription.
    if initially_connected:
        connection.uninstall_soc()
    else:
        connection.install_soc(env.soc, env.soccfg)
    ui.click("SoC…")
    ui.screenshot(shots / "03-reopened.png")
    configs = [w.text for w in ui.visible().widgets if w.kind == "QPlainTextEdit"]
    if initially_connected:
        assert any(w.text == "Disconnected" for w in ui.visible().widgets)
        assert configs == []
    else:
        assert any("connected" in w.text.lower() for w in ui.visible().widgets)
        assert configs and all(configs)
    assert ui.observe("Connect").enabled
    assert ui.observe("Close").enabled

    # Reopen also restores the visible event subscription in both directions.
    if initially_connected:
        connection.install_soc(env.soc, env.soccfg)
    else:
        connection.uninstall_soc()
    ui.screenshot(shots / "04-visible-change.png")
    configs = [w.text for w in ui.visible().widgets if w.kind == "QPlainTextEdit"]
    if initially_connected:
        assert any("connected" in w.text.lower() for w in ui.visible().widgets)
        assert configs and all(configs)
    else:
        assert any(w.text == "Disconnected" for w in ui.visible().widgets)
        assert configs == []
    ui.click("Close")
    ui.click("SoC…")
    ui.screenshot(shots / "05-reopened-again.png")
    assert ui.observe("Connect").enabled
    ui.click("Close")


def test_window_shutdown_closes_visible_soc(
    soc_session: WindowSession, tmp_path: Path
) -> None:
    ui = ComponentDriver(soc_session.window)
    shots = Path(os.environ.get("SOC_UI_EVIDENCE", str(tmp_path))) / "workflows-close"
    assert any(w.text == "SoC…" for w in ui.visible().widgets), (
        "SoC toolbar entry missing"
    )
    ui.click("SoC…")
    ui.screenshot(shots / "01-open.png")
    assert ui.dialogs()[0].title == "SoC"
    # Qt close is the process-window boundary under test, not a dialog-opening shortcut.
    assert soc_session.window.close()
    assert ui.dialogs() == ()
