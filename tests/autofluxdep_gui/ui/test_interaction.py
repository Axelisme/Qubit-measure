"""Headless interaction tests for the autofluxdep-gui prototype.

Drives the window through the Controller / NodeListPane (not real dialogs) and
asserts the UI reflects State and the edit↔run switch. The run uses FAKE
measurement Nodes (deterministic produce, no acquire): these tests exercise the
UI's run wiring (lock → fill → unlock, build canvases, auto-follow), not the
experiment physics — the real-acquire path is covered by the ``test_*_acquire``
integration tests. The run worker is a real QThread driven to completion via the
Qt event loop. A second ad-hoc provider (no Result → no liveplot) gives the list
two rows for reorder/remove.
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow

from .._helpers import connect_mock, make_builder, make_measurement_builder


@pytest.fixture
def app(qapp):
    ctrl = build_core()
    ctrl.add_node(make_measurement_builder("qubit_freq"))
    # a second provider (ad-hoc, no Result → no liveplot) so the list has 2 rows
    ctrl.add_node(make_builder("probe", provides=("v",)))
    win = MainWindow(ctrl)
    yield ctrl, win
    # Quiesce the controller's BackgroundService before GC: device setup and
    # other session operations submit pool workers via BackgroundService._runner;
    # a pending queued delivery to a GC'd runner segfaults a later test's event pump.
    ctrl._background_svc.quiesce()
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


def test_devices_button_opens_shared_device_dialog(app):
    # the Devices… button opens the shared session DeviceDialog, which depends
    # only on the SessionControllerPort the autofluxdep Controller implements —
    # constructing it with the live controller is the runtime conformance check.
    from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

    ctrl, win = app
    assert hasattr(win._list, "_devices_btn")
    dlg = DeviceDialog(ctrl, win)  # must not raise
    assert dlg._ctrl is ctrl
    dlg.deleteLater()


def test_flux_source_picker_records_selection(app):
    # the flux-source combo records which connected device the sweep is applied
    # through (its unit labels the flux axis); with no devices it is just "(none)".
    ctrl, win = app
    assert hasattr(win._list, "_flux_source")
    assert win._list._flux_source.count() == 1  # only "(none)"
    assert ctrl.get_flux_device() is None
    ctrl.set_flux_device("flux_dev")
    assert ctrl.get_flux_device() == "flux_dev"
    ctrl.set_flux_device(None)  # clearing → bare flux numbers
    assert ctrl.get_flux_device() is None


def test_predictor_button_opens_shared_predictor_dialog(app):
    # the Predictor… button opens the shared session PredictorDialog (load a
    # FluxoniumPredictor into the context) — runtime port conformance check.
    from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog

    ctrl, win = app
    assert hasattr(win._list, "_predictor_btn")
    dlg = PredictorDialog(ctrl, win)  # must not raise
    assert dlg._ctrl is ctrl
    dlg.deleteLater()


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
    connect_mock(ctrl)
    win._list._refresh_buttons()
    assert win._list._run_btn.isEnabled()
    assert "ok" in win._list._setup_light.text()


# --- run lifecycle: edit↔run lock, liveplot canvas, progress ---


def _zero_delays(ctrl):
    # the fake measurement Nodes produce instantly (no acquire / no delay), so a UI
    # run completes within the event pump without any pacing to neutralise. Kept as
    # a no-op so the run helpers read the same with or without a delay seam.
    del ctrl


def _pump_until_done(ctrl, win):
    for _ in range(20000):
        QApplication.processEvents()
        if win._worker is None and not ctrl.is_running:
            break


def _run_to_completion(ctrl, win):
    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 1.0, 2.0])
    connect_mock(ctrl)
    win._list._refresh_buttons()
    win._start()
    _pump_until_done(ctrl, win)


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


def test_produce_exception_during_gui_run_does_not_crash_and_unlocks(qapp, monkeypatch):
    # a Node whose produce raises mid-run (e.g. an unconfigured real acquire) must
    # NOT abort the run worker QThread: the orchestrator catches it, the run ends on
    # RUN_FAILED, and the UI unlocks back to edit state. The warning dialog is
    # stubbed so the headless test does not block on a modal.
    from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]
    from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency

    from .._helpers import make_builder

    def boom(env, snapshot):
        del env, snapshot
        raise RuntimeError("node not configured")

    ctrl = build_core()
    ctrl.add_node(
        make_builder("broken", requires=(Dependency("predict_freq"),), produce_fn=boom)
    )
    win = MainWindow(ctrl)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )

    ctrl.set_flux_values([0.0, 1.0])
    connect_mock(ctrl)
    win._list._refresh_buttons()
    win._start()
    _pump_until_done(ctrl, win)

    # the run ended (worker quiesced, not aborted) and the UI is back in edit state
    assert win._worker is None
    assert not ctrl.is_running
    assert win._list._run_btn.text() == "▶ Run"
    assert win._list._add_btn.isEnabled()
    # the failure surfaced to the user
    assert any("node not configured" in m for m in shown)
    ctrl._background_svc.quiesce()
    win.close()
    win.deleteLater()


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
    import numpy as np

    result = ctrl.state.run_results["qubit_freq"]
    assert not np.isnan(result.signal[-1]).any()  # the last row was filled


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


def test_multiple_real_experiments_each_get_a_liveplot(qapp):
    # a real multi-experiment workflow: each measurement provider gets its own
    # sweep-lived canvas + plotter, and the LivePlot-backed Plotter redraws on the
    # main thread (every experiment shares the same notify→update→draw wiring).
    ctrl = build_core()
    for t in ("qubit_freq", "t1", "ro_optimize", "mist"):
        ctrl.add_node(make_measurement_builder(t))
    win = MainWindow(ctrl)
    win._list.select_index(1)  # follow t1's plot
    redraws = {"t1": 0}

    def patch_counter():
        if "t1" in win._plots:
            _canvas, plotter = win._plots["t1"]
            orig = plotter.update

            def wrapped(result, idx, _o=orig):
                redraws["t1"] += 1
                _o(result, idx)

            plotter.update = wrapped

    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5, 1.0])
    connect_mock(ctrl)
    win._list._refresh_buttons()
    win._start()
    patch_counter()  # wrap after _build_plots created the plotter
    _pump_until_done(ctrl, win)

    # every measurement provider built a canvas + plotter (predictor Service none)
    assert set(win._plots) == {"qubit_freq", "t1", "ro_optimize", "mist"}
    for name in win._plots:
        canvas, plotter = win._plots[name]
        assert canvas is not None and plotter is not None
    # t1's Plotter redrew on the main thread as rows filled
    assert redraws["t1"] >= 1
    win.close()
    win.deleteLater()


def test_run_auto_follows_each_entered_node(qapp):
    # as the sweep enters each provider, the left list selects it + the detail
    # pane switches to its run tab (the canvas it shows follows the selection).
    ctrl = build_core()
    for t in ("qubit_freq", "t1", "mist"):
        ctrl.add_node(make_measurement_builder(t))
    win = MainWindow(ctrl)
    win._list.select_index(0)

    # record which row was selected + the sub-tab when each Node was entered
    followed = []

    def on_entered(name, _idx):
        followed.append((name, win._list.selected_index, win._detail.current_tab))

    win._bridge.node_entered.connect(on_entered)
    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5])
    connect_mock(ctrl)
    win._list._refresh_buttons()
    win._start()
    _pump_until_done(ctrl, win)

    nav = {name: (row, tab) for name, row, tab in followed}
    # each Node, when entered, selected its own list row and showed the run tab
    assert nav["qubit_freq"] == (0, 1)
    assert nav["t1"] == (1, 1)
    assert nav["mist"] == (2, 1)
    # the predictor Service never drives navigation (filtered by the controller)
    assert "predictor" not in nav
    win.close()
    win.deleteLater()


def test_rename_updates_list_and_keeps_canvas_key(qapp):
    # renaming two mist placements to g_mist / e_mist relabels the list and keys
    # each one's liveplot canvas under its instance name.
    ctrl = build_core()
    ctrl.add_node(make_measurement_builder("mist"))
    ctrl.add_node(make_measurement_builder("mist"))
    win = MainWindow(ctrl)
    ctrl.rename_node(0, "g_mist")
    ctrl.rename_node(1, "e_mist")
    win._list.refresh_list()
    assert _list_labels(win) == ["g_mist", "e_mist"]

    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5])
    connect_mock(ctrl)
    win._list._refresh_buttons()
    win._start()
    _pump_until_done(ctrl, win)

    # canvases are keyed by instance name, independent per placement
    assert set(win._plots) == {"g_mist", "e_mist"}
    assert win._plots["g_mist"][0] is not win._plots["e_mist"][0]
    win.close()
    win.deleteLater()


def test_no_canvas_is_ever_a_toplevel_window(qapp):
    # every Node's Plotter redraws each run point, even off-screen ones; a
    # parentless canvas becomes a top-level window the moment it draws (the
    # "stray window flashing" bug). All canvases must stay parented — the shown
    # one in the run tab, the rest under the hidden park — so none is a window.
    ctrl = build_core()
    for t in ("qubit_freq", "t1", "mist"):
        ctrl.add_node(make_measurement_builder(t))
    win = MainWindow(ctrl)
    win.show()
    win._list.select_index(0)
    _zero_delays(ctrl)
    ctrl.set_flux_values([0.0, 0.5])
    connect_mock(ctrl)
    win._list._refresh_buttons()
    win._start()
    _pump_until_done(ctrl, win)

    park = win._canvas_park
    for name, (canvas, _plotter) in win._plots.items():
        assert not canvas.isWindow(), f"{name} canvas is a top-level window"
        assert canvas.parent() is not None, f"{name} canvas is parentless"

    # de-selecting a Node parks its canvas (never leaves it parentless)
    win._list.select_index(1)  # switch away from whatever is shown
    QApplication.processEvents()
    for name, (canvas, _plotter) in win._plots.items():
        assert not canvas.isWindow(), f"{name} canvas became a window after switch"
        # the de-selected canvases sit under the park
        if win._detail._canvas is not canvas:
            assert canvas.parent() is park, f"{name} not parked after de-select"
    win.close()
    win.deleteLater()
