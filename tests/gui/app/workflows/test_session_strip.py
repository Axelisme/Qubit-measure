"""The session strip above the workflow columns shows only the Entry chip.

The chip follows the shared session's entry and point. Its text is read from
the shown QLabel.
"""

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QWidget
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import ResultEntry

from ._window_support import WindowSession, bind_point


def strip_chips(window: QWidget) -> list[QLabel]:
    strip = window.findChild(QWidget, "workflow.session")
    assert strip is not None
    return strip.findChildren(QLabel)


def entry_text(window: QWidget) -> str:
    chip = window.findChild(QLabel, "workflow.session.entry")
    assert chip is not None
    return chip.text()


def test_strip_shows_exactly_the_entry_chip(window_session: WindowSession) -> None:
    chips = strip_chips(window_session.window)
    assert [chip.objectName() for chip in chips] == ["workflow.session.entry"]
    assert chips[0].text() == "Entry: none"


def test_entry_chip_follows_opened_entry_and_selected_point(
    window_session: WindowSession,
) -> None:
    window, harness = window_session.window, window_session.harness
    entry = ResultEntry.create(
        "sample",
        result_root=window_session.root / "result",
        database_root=window_session.root / "Database",
    )
    entry.new_point("source", description="strip")
    entry.new_point("other", description="strip")
    context = harness.services.context
    context.install_entry(entry)
    assert entry_text(window) == "Entry: sample / no point"
    context.use_point(PointUseRequest("source"))
    assert entry_text(window) == "Entry: sample / source"
    context.use_point(PointUseRequest("other"))
    assert entry_text(window) == "Entry: sample / other"
    assert len(strip_chips(window)) == 1


def test_closed_window_stops_following_session_events(
    window_session: WindowSession,
    qapp: QApplication,
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = window_session.window
    entry = bind_point(window_session.harness, window_session.root)
    assert window.close()
    qapp.processEvents()
    entry.new_point("later", description="after close")
    window_session.harness.services.context.use_point(PointUseRequest("later"))
    assert entry_text(window) == "Entry: sample / source"
    assert "raised" not in caplog.text
