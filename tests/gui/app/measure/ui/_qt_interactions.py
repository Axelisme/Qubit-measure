"""Shared real-Qt interaction scaffolding for measure UI tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QDialog


def wait_until(qapp: QApplication, condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 4
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert condition()


def modal_action(
    qapp: QApplication,
    action: Callable[[], object],
    interactions: list[tuple[type[QDialog], Callable[[Any], None]]],
) -> None:
    """Deliver real nested modal decisions, with bounded cleanup on test failure."""
    pending = iter(interactions)
    errors: list[Exception] = []
    visited: list[type[QDialog]] = []
    timer = QTimer()
    timer.setSingleShot(True)

    def abort() -> None:
        errors.append(AssertionError("modal interaction timed out"))
        for widget in qapp.topLevelWidgets():
            if isinstance(widget, QDialog) and widget.isVisible():
                widget.reject()

    def schedule() -> None:
        item = next(pending, None)
        if item is None:
            return
        expected, interact = item

        def deliver() -> None:
            dialog = qapp.activeModalWidget()
            try:
                assert isinstance(dialog, expected), (expected, dialog)
                visited.append(expected)
                schedule()
                interact(dialog)
            except AssertionError as exc:
                errors.append(exc)
                if isinstance(dialog, QDialog):
                    dialog.reject()

        QTimer.singleShot(0, deliver)

    timer.timeout.connect(abort)
    timer.start(2000)
    schedule()
    try:
        action()
        wait_until(qapp, lambda: bool(errors) or len(visited) == len(interactions))
        qapp.processEvents()
    finally:
        timer.stop()
    if errors:
        raise errors[0]
    assert len(visited) == len(interactions)
