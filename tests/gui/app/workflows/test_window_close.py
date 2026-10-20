"""Rejected Window close keeps the run, its diagnostic and its figure canvas.

A Stop journal fault during close must not destroy the producer's canvas in
the figure column or release run resources before owner delivery.
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QStackedWidget

from tests._filesystem import DIRECTORY_OPEN_ERROR

from ._start_fakes import StartHarness
from ._window_support import WindowSession, bind_point, form_editable, install_soc

RUN_ID = "close-fault"


def paused_with_canvas(session: WindowSession) -> tuple[int, Path, QStackedWidget]:
    """Pause the wide workflow at its first point, with one live canvas shown."""
    window, harness = session.window, session.harness
    entry = bind_point(harness, session.root)
    install_soc(harness)
    session.pause_at = 0
    token = window.start(RUN_ID)
    assert isinstance(token, int)
    harness.complete_segment()
    figures = window.findChild(QStackedWidget, "workflow.figures")
    assert isinstance(figures, QStackedWidget) and figures.count() == 2
    assert window.tuning_panel is not None
    assert harness.progress.bars_for_owner("workflow")
    return (
        token,
        entry.result_path / f"workflows/synthetic_wide_{RUN_ID}/journal.jsonl",
        figures,
    )


@contextmanager
def unwritable(journal: Path) -> Generator[None]:
    """Replace the run journal with a directory, restoring it afterwards."""
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        yield
    finally:
        journal.rmdir()
        journal.write_bytes(original)


def assert_close_rejected_with_diagnostic(
    session: WindowSession, journal: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The rejected close leaves the native diagnostic visible to the user."""
    window = session.window
    assert not window.close()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "failed"
    message = window.findChild(QLabel, "workflow.run.message")
    assert message is not None and str(journal) in message.text()
    assert str(journal) in caplog.text
    assert any(
        record.name == "zcu_tools.gui.app.workflows.window"
        and record.exc_info is not None
        and isinstance(record.exc_info[1], DIRECTORY_OPEN_ERROR)
        and record.exc_info[1].filename == str(journal)
        for record in caplog.records
    )


def assert_failed_settlement(harness: StartHarness, token: int, journal: Path) -> None:
    """One failed settlement keeps the journal cause and releases resources."""
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert outcome.error is not None and str(journal) in outcome.error
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("workflow")


def assert_closes_and_releases_canvas(
    session: WindowSession, qapp: QApplication, figures: QStackedWidget
) -> None:
    window = session.window
    assert not window.controller.has_pending_segment
    assert window.close() and window.close()
    qapp.processEvents()
    assert figures.count() == 1


def test_close_of_paused_run_settles_stop_journal_failure(
    window_session: WindowSession,
    qapp: QApplication,
    caplog: pytest.LogCaptureFixture,
) -> None:
    window, harness = window_session.window, window_session.harness
    token, journal, figures = paused_with_canvas(window_session)
    with unwritable(journal):
        assert_close_rejected_with_diagnostic(window_session, journal, caplog)
        assert not window.controller.has_pending_segment
        assert harness.background.work is None
        assert form_editable(window.plan_panel)
        assert harness.handles.known_outcome(token) is not None
    assert_failed_settlement(harness, token, journal)
    assert_closes_and_releases_canvas(window_session, qapp, figures)


def test_close_of_active_run_keeps_canvas_until_owner_delivery(
    window_session: WindowSession,
    qapp: QApplication,
    caplog: pytest.LogCaptureFixture,
) -> None:
    window, harness = window_session.window, window_session.harness
    token, journal, figures = paused_with_canvas(window_session)
    canvas = figures.currentWidget()
    panel = window.tuning_panel
    assert panel is not None and window.resume(RUN_ID) == token
    with unwritable(journal):
        assert_close_rejected_with_diagnostic(window_session, journal, caplog)
        controller = window.controller
        assert controller.has_pending_segment and controller.binding is panel.host
        assert figures.count() == 2 and figures.currentWidget() is canvas
        assert not form_editable(window.plan_panel)
        assert harness.background.work is not None and harness.gate.snapshot()
        assert harness.handles.live_count() == 1
        assert harness.handles.known_outcome(token) is None

    harness.background.execute()
    assert harness.gate.snapshot() and harness.handles.known_outcome(token) is None
    assert figures.count() == 2 and figures.currentWidget() is canvas
    harness.background.deliver()
    assert_failed_settlement(harness, token, journal)
    assert_closes_and_releases_canvas(window_session, qapp, figures)
