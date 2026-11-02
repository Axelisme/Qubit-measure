"""Diagnostic presentation through visible input with an offline synthetic host."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QPushButton
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._synthetic import SampleAdapter

TITLE = "Synthetic acquisition failed"
MESSAGE = (
    "Cannot save synthetic acquisition\nTraceback: synthetic writer\nOSError: disk full"
)


@pytest.fixture
def diagnostic_window(
    request: pytest.FixtureRequest, qapp: QApplication, tmp_path: Path
) -> Iterator[ComponentDriver]:
    """Inject external diagnostics without a socket, worker or instrument."""
    registry = Registry()
    registry.register("synthetic", SampleAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
        progress_transport=DirectProgressTransport(),
    )
    mode = request.param
    if mode != "local":
        ctrl.control_identity.connect("synthetic-client", "synthetic", "owner")
        if mode == "remote-offline":
            ctrl.control_identity.disconnect("synthetic-client")
    window = MainWindow(ctrl, templates=lambda: {})
    scope = LifetimeScope(window)
    # These fixture-only producers stand in for incoming diagnostics. The scenario
    # cannot issue business commands under Remote control, just like a real user.
    status_bar = window.statusBar()
    assert status_bar is not None
    for caption, severity, title, message in (
        ("Emit error", "error", TITLE, MESSAGE),
        ("Emit info", "info", "", "Synthetic progress"),
    ):
        button = QPushButton(caption, window)
        status_bar.addPermanentWidget(button)
        scope.connect(
            button.clicked,
            lambda checked=False, severity=severity, title=title, message=message: (
                window.notify_diagnostic(severity, title, message)
            ),
        )
    window.show()
    qapp.processEvents()
    try:
        yield ComponentDriver(window)
    finally:
        scope.close()
        ctrl.close()
        window.dispose()
        qapp.processEvents()


@pytest.fixture
def notice_timers(monkeypatch: pytest.MonkeyPatch) -> list[QTimer]:
    """Hold the notice's external timer until the scenario delivers its expiry."""
    pending: list[QTimer] = []
    real_start = QTimer.start

    def start(timer: QTimer, milliseconds: int | None = None) -> None:
        parent = timer.parent()
        if (
            parent is not None
            and parent.objectName() == "diagnostic.notice"
            and milliseconds != 0
        ):
            if milliseconds is not None:
                timer.setInterval(milliseconds)
            pending.append(timer)
        elif milliseconds is None:
            real_start(timer)
        else:
            real_start(timer, milliseconds)

    # Only the notice timer is held. start(0) explicitly releases a real Qt timeout;
    # qt_until's probe/watchdog and all other Qt collaborators retain real scheduling.
    monkeypatch.setattr(QTimer, "start", start)
    return pending


def _notice_visible(driver: ComponentDriver) -> bool:
    return any(w.object_name == "diagnostic.notice" for w in driver.visible().widgets)


@pytest.mark.parametrize("diagnostic_window", ["remote-online"], indirect=True)
def test_online_remote_error_expires_and_logs_full_content(
    diagnostic_window: ComponentDriver,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    notice_timers: list[QTimer],
) -> None:
    driver = diagnostic_window
    driver.screenshot(tmp_path / "01-online.png")
    assert driver.observe("Remote · synthetic · online").visible
    with caplog.at_level(logging.ERROR):
        driver.click("Emit error")
    driver.screenshot(tmp_path / "02-error.png")
    assert _notice_visible(driver), "online Remote errors need a transient notice"
    assert driver.observe(TITLE).visible
    assert driver.observe(MESSAGE.splitlines()[0]).visible
    assert not driver.dialogs()
    assert TITLE in caplog.text and MESSAGE in caplog.text
    (tmp_path / "error.log").write_text(caplog.text, encoding="utf-8")
    assert len(notice_timers) == 1
    driver.screenshot(tmp_path / "03-before-expiry.png")
    assert _notice_visible(driver)
    notice_timers.pop().start(0)
    # The released single-shot timeout must reach the notice on the owner loop.
    qt_until(lambda: not _notice_visible(driver), label="notice auto-dismissed")
    driver.screenshot(tmp_path / "04-expired.png")
    assert not driver.dialogs()
    driver.click("Emit info")
    driver.screenshot(tmp_path / "05-info.png")
    assert "Synthetic progress" in driver.status().text
    assert not _notice_visible(driver)


@pytest.mark.parametrize("diagnostic_window", ["remote-online"], indirect=True)
def test_remote_notice_details_remain_until_manually_closed(
    diagnostic_window: ComponentDriver, tmp_path: Path, notice_timers: list[QTimer]
) -> None:
    driver = diagnostic_window
    driver.click("Emit error")
    driver.screenshot(tmp_path / "01-summary.png")
    assert any(w.text == "Details…" for w in driver.visible().widgets), (
        "the transient notice needs a full-content inspection action"
    )
    driver.click("Details…")
    driver.screenshot(tmp_path / "02-details.png")
    assert driver.dialogs()[0].title == TITLE
    assert MESSAGE in driver.dialogs()[0].texts
    driver.click("Emit info")
    driver.screenshot(tmp_path / "03-non-modal-details.png")
    assert "Synthetic progress" in driver.status().text
    assert len(notice_timers) == 1
    notice_timers.pop().start(0)
    # The scheduled timeout hides only the summary, not the inspected report.
    qt_until(
        lambda: not _notice_visible(driver),
        label="summary expires without dismissing inspected details",
    )
    driver.screenshot(tmp_path / "04-expired-details-retained.png")
    assert MESSAGE in driver.dialogs()[0].texts
    driver.click("OK")
    driver.screenshot(tmp_path / "05-details-closed.png")
    assert not driver.dialogs()


@pytest.mark.parametrize(
    "diagnostic_window", ["local", "remote-offline"], indirect=True
)
def test_unattended_error_report_requires_manual_close(
    diagnostic_window: ComponentDriver, tmp_path: Path, notice_timers: list[QTimer]
) -> None:
    driver = diagnostic_window
    driver.screenshot(tmp_path / "01-control.png")
    assert any(
        w.text in ("Local", "Remote · synthetic · offline")
        for w in driver.visible().widgets
    )
    driver.click("Emit error")
    driver.screenshot(tmp_path / "02-report.png")
    assert driver.dialogs()[0].title == TITLE
    assert MESSAGE in driver.dialogs()[0].texts
    assert not _notice_visible(driver)
    driver.click("Emit info")
    driver.screenshot(tmp_path / "03-non-modal.png")
    assert "Synthetic progress" in driver.status().text
    # Manual reports do not schedule the online notice's auto-dismiss timer.
    assert not notice_timers
    driver.screenshot(tmp_path / "04-report-retained.png")
    assert MESSAGE in driver.dialogs()[0].texts
    driver.click("OK")
    driver.screenshot(tmp_path / "05-closed.png")
    assert not driver.dialogs()
