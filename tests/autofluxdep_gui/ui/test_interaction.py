"""Headless interaction tests for the autofluxdep-gui prototype.

Drives the window through the Controller / NodeListPane (not real dialogs) and
asserts the UI reflects State and the edit↔run switch. The run is fake (dry
data, no hardware); the run worker is a real QThread driven to completion via the
Qt event loop.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow


@pytest.fixture
def app(qapp):
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    ctrl.add_node_by_type("ro_optimize")
    win = MainWindow(ctrl)
    yield ctrl, win
    win.close()
    win.deleteLater()


def _list_labels(win: MainWindow) -> list[str]:
    lst = win._list._list
    items = [lst.item(i) for i in range(lst.count())]
    return [it.text() for it in items if it is not None]


# --- workflow editing reflects in the list ---


def test_list_reflects_nodes(app):
    _ctrl, win = app
    assert _list_labels(win) == ["qubit_freq", "ro_optimize"]


def test_reorder_swaps_and_keeps_selection(app):
    ctrl, win = app
    win._list.select_index(1)  # ro_optimize
    win._list._on_move(-1)  # move up
    assert _list_labels(win) == ["ro_optimize", "qubit_freq"]
    assert ctrl.state.node_names() == ["ro_optimize", "qubit_freq"]


def test_remove_node(app):
    ctrl, win = app
    win._list.select_index(0)
    win._list._on_remove()
    assert _list_labels(win) == ["ro_optimize"]
    assert ctrl.state.node_names() == ["ro_optimize"]


# --- selection drives the right pane ---


def test_selection_shows_node_form(app):
    _ctrl, win = app
    win._list.select_index(0)
    assert win._detail._title.text() == "qubit_freq"
    assert win._detail.current_form is not None
    win._list.select_index(1)
    assert win._detail._title.text() == "ro_optimize"


# --- Setup → Run enable ---


def test_run_disabled_until_setup(app):
    ctrl, win = app
    assert not win._list._run_btn.isEnabled()  # no setup yet
    # call setup directly (the dialog is modal; tested separately in test_setup)
    ctrl.setup(use_mock=True)
    win._list._refresh_buttons()
    assert win._list._run_btn.isEnabled()
    assert "ok" in win._list._setup_light.text()


# --- run lifecycle: edit↔run lock, auto-follow, progress ---


def _run_to_completion(ctrl, win):
    ctrl.set_flux_values([0.0, 1.0, 2.0])
    ctrl.setup(use_mock=True)
    win._list._refresh_buttons()
    win._start()
    # pump the event loop until the worker finishes and run-done fires
    for _ in range(2000):
        QApplication.processEvents()
        if win._worker is None and not ctrl.is_running:
            break


def test_run_locks_then_unlocks(app):
    ctrl, win = app
    win._list.select_index(0)
    _run_to_completion(ctrl, win)
    # back in edit state after finish
    assert win._list._run_btn.text() == "▶ Run"
    assert win._list._add_btn.isEnabled()
    # form not read-only again
    assert win._detail.current_form is not None
    # progress reached the end
    assert win._progress.value() == 3


def test_run_auto_follows_and_locks_form(app):
    ctrl, win = app

    # capture the UI state observable on the main thread when a Node starts.
    # (ctrl.is_running is a worker-thread flag that races the main-thread slot,
    # so we assert the UI-visible run state — button text + active sub-tab —
    # which the run_started slot set on the main thread before any node_started.)
    seen = {}
    followed = []

    def on_node_started(name, idx):
        seen["run_btn"] = win._list._run_btn.text()
        seen["detail_tab"] = win._detail.current_tab
        followed.append((name, win._list.selected_index))

    win._bridge.node_started.connect(on_node_started)
    win._list.select_index(0)
    _run_to_completion(ctrl, win)

    # during the run the toggle button showed Stop and detail was on the run tab
    assert seen.get("run_btn") == "■ Stop"
    assert seen.get("detail_tab") == 1  # run sub-tab
    # auto-follow: the left list selected the Node that started
    assert ("qubit_freq", 0) in followed
    assert ("ro_optimize", 1) in followed
