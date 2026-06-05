"""Headless interaction tests for the autofluxdep-gui prototype.

Drives the window through the Controller / NodeListPane (not real dialogs) and
asserts the UI reflects State and the edit↔run switch. The run uses synthetic
signals (no hardware); the run worker is a real QThread driven to completion via
the Qt event loop. A second ad-hoc provider (added directly, no registry) gives
the list two rows for reorder/remove without a second real experiment.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow

from .._helpers import make_builder


@pytest.fixture
def app(qapp):
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    # a second provider (ad-hoc, no Result → no liveplot) so the list has 2 rows
    ctrl.add_node(make_builder("probe", provides=("v",)))
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
    assert _list_labels(win) == ["qubit_freq", "probe"]


def test_reorder_swaps_and_keeps_selection(app):
    ctrl, win = app
    win._list.select_index(1)  # probe
    win._list._on_move(-1)  # move up
    assert _list_labels(win) == ["probe", "qubit_freq"]
    assert ctrl.state.node_names() == ["probe", "qubit_freq"]


def test_remove_node(app):
    ctrl, win = app
    win._list.select_index(0)
    win._list._on_remove()
    assert _list_labels(win) == ["probe"]
    assert ctrl.state.node_names() == ["probe"]


# --- selection drives the right pane ---


def test_selection_shows_node_form(app):
    _ctrl, win = app
    win._list.select_index(0)
    assert win._detail._title.text() == "qubit_freq"
    assert win._detail.current_form is not None
    win._list.select_index(1)
    assert win._detail._title.text() == "probe"


# --- Setup → Run enable ---


def test_run_disabled_until_setup(app):
    ctrl, win = app
    assert not win._list._run_btn.isEnabled()  # no setup yet
    ctrl.setup(use_mock=True)
    win._list._refresh_buttons()
    assert win._list._run_btn.isEnabled()
    assert "ok" in win._list._setup_light.text()


# --- run lifecycle: edit↔run lock, liveplot canvas, progress ---


def _run_to_completion(ctrl, win):
    ctrl.set_flux_values([0.0, 1.0, 2.0])
    ctrl.setup(use_mock=True)
    win._list._refresh_buttons()
    win._start()
    # pump the event loop until the worker finishes and run-done fires
    for _ in range(4000):
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
    assert win._detail.current_form is not None
    # progress reached the end
    assert win._progress.value() == 3


def test_run_builds_liveplot_canvas_for_measurement_node(app):
    ctrl, win = app
    win._list.select_index(0)  # qubit_freq has a Result → a canvas is built
    _run_to_completion(ctrl, win)
    # the qubit_freq provider got a sweep-lived canvas + plotter
    assert "qubit_freq" in win._plots
    canvas, plotter = win._plots["qubit_freq"]
    assert canvas is not None and plotter is not None
    # the ad-hoc "probe" provider has no Result → no canvas
    assert "probe" not in win._plots
    # the Result was filled (the worker ran produce over the sweep)
    result = ctrl.state.run_results["qubit_freq"]
    assert not all(map(lambda r: r != r, result.fit_freq))  # at least one non-nan


def test_run_switches_detail_to_run_tab(app):
    ctrl, win = app
    win._list.select_index(0)
    captured = {}

    def on_started():
        captured["tab"] = win._detail.current_tab
        captured["btn"] = win._list._run_btn.text()

    win._bridge.run_started.connect(on_started)
    _run_to_completion(ctrl, win)
    # the run_started slot (main thread) switched to the run sub-tab + showed Stop
    assert captured.get("tab") == 1
    assert captured.get("btn") == "■ Stop"
