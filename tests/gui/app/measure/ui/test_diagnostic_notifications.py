"""Diagnostic presentation through visible input with an offline synthetic host."""

import logging
from collections.abc import Iterator
from pathlib import Path
from time import monotonic

import pytest
from qtpy.QtTest import QTest
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
    for caption, severity, title, message in (
        ("Emit error", "error", TITLE, MESSAGE),
        ("Emit info", "info", "", "Synthetic progress"),
    ):
        button = QPushButton(caption, window)
        window.statusBar().addPermanentWidget(button)
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
    QTest.qWait(7000)
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
