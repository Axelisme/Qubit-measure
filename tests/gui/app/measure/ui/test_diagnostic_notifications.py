"""Diagnostic presentation through visible input with an offline synthetic host."""

import logging
from collections.abc import Iterator
from pathlib import Path
from time import monotonic

import pytest
from qtpy.QtCore import QEventLoop, Qt, QTimer
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


def _elapse(milliseconds: int) -> None:
    """Pump Qt for a time-contract observation, not for background completion."""
    loop = QEventLoop()
    timer = QTimer(loop)
    timer.setSingleShot(True)
    timer.setTimerType(Qt.TimerType.PreciseTimer)
    timer.timeout.connect(loop.quit)
    timer.start(milliseconds)
    loop.exec()


def _notice_visible(driver: ComponentDriver) -> bool:
    return any(w.object_name == "diagnostic.notice" for w in driver.visible().widgets)


@pytest.mark.parametrize("diagnostic_window", ["remote-online"], indirect=True)
def test_online_remote_error_expires_and_logs_full_content(
    diagnostic_window: ComponentDriver, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    driver = diagnostic_window
    driver.screenshot(tmp_path / "01-online.png")
    assert driver.observe("Remote · synthetic · online").visible
    with caplog.at_level(logging.ERROR):
        started = monotonic()
        driver.click("Emit error")
    driver.screenshot(tmp_path / "02-error.png")
    assert _notice_visible(driver), "online Remote errors need a transient notice"
    assert driver.observe(TITLE).visible
    assert driver.observe(MESSAGE.splitlines()[0]).visible
    assert not driver.dialogs()
    assert TITLE in caplog.text and MESSAGE in caplog.text
    (tmp_path / "error.log").write_text(caplog.text, encoding="utf-8")
    # Time is the product contract here: the notice must survive before expiry.
    _elapse(7000)
    driver.screenshot(tmp_path / "03-before-expiry.png")
    assert _notice_visible(driver)
    qt_until(
        lambda: not _notice_visible(driver),
        label="notice auto-dismissed",
        watchdog_ms=4000,
    )
    elapsed = monotonic() - started
    driver.screenshot(tmp_path / "04-expired.png")
    assert 7.0 <= elapsed <= 11.0
    assert not driver.dialogs()
    (tmp_path / "timing.txt").write_text(
        f"notice_elapsed_seconds={elapsed:.3f}\n", encoding="utf-8"
    )
    driver.click("Emit info")
    driver.screenshot(tmp_path / "05-info.png")
    assert "Synthetic progress" in driver.status().text
    assert not _notice_visible(driver)


@pytest.mark.parametrize("diagnostic_window", ["remote-online"], indirect=True)
def test_remote_notice_details_remain_until_manually_closed(
    diagnostic_window: ComponentDriver, tmp_path: Path
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
    qt_until(
        lambda: not _notice_visible(driver),
        label="summary expires without dismissing inspected details",
        watchdog_ms=11_000,
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
    diagnostic_window: ComponentDriver, tmp_path: Path
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
    # Unlike online Remote, these reports must remain beyond the expiry interval.
    _elapse(8500)
    driver.screenshot(tmp_path / "04-report-retained.png")
    assert MESSAGE in driver.dialogs()[0].texts
    driver.click("OK")
    driver.screenshot(tmp_path / "05-closed.png")
    assert not driver.dialogs()
