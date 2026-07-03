"""Cooperative run-cancellation — the controller honours stop_run() mid-sweep.

These exercise the stateful cancel path: stop_run() flips the cooperative flag,
start_run breaks the sweep early at the next should_stop poll, and the run ends on
RUN_STOPPED (not RUN_FINISHED) leaving a partial InfoStore. The cancel mechanic is
independent of experiment physics, so a fake measurement Node (a deterministic
``make_builder`` double) drives the run — fast, no acquire, no Qt.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.events.run import (
    PointDonePayload,
    RunEvent,
    RunFinishedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch

from ._helpers import (
    make_builder,
    make_measurement_builder,
    pump_controller_until_idle,
    run_controller_to_completion,
)

_FLUX = [0.0, 0.25, 0.5, 0.75, 1.0]


def _build_ready_controller():
    ctrl = build_core()
    ctrl.add_node(make_measurement_builder("probe"))
    ctrl.set_flux_values(_FLUX)
    return ctrl


def test_stop_run_mid_sweep_emits_run_stopped_not_finished(monkeypatch):
    ctrl = _build_ready_controller()
    persist_all = MagicMock()
    monkeypatch.setattr(ctrl, "persist_all", persist_all)

    events: list[RunEvent] = []
    ctrl.bus.subscribe(RunStoppedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append(p.EVENT))

    points: list[int] = []

    def on_point_done(p: PointDonePayload) -> None:
        points.append(p.idx)
        if len(points) == 2:  # request cancel after the 2nd point completes
            ctrl.stop_run("test cancellation")

    ctrl.bus.subscribe(PointDonePayload, on_point_done)

    token = ctrl.start_run()
    pump_controller_until_idle(ctrl)
    info = ctrl.last_run_info
    assert info is not None

    # the run ended on RUN_STOPPED, and RUN_FINISHED never fired
    assert events == [RunEvent.RUN_STOPPED]
    result = ctrl.await_operation(token, timeout=0.0)
    assert result is not None
    assert result.outcome is not None
    assert result.outcome.status == "cancelled"
    assert result.feedback == "test cancellation"
    # only points 0 and 1 ran; the sweep broke before point 2 of 5
    assert points == [0, 1]
    # the returned InfoStore is partial: it reflects the last completed point
    assert info.point["flux_idx"] == 1
    # completed rows stay in the pre-allocated Result; later rows remain untouched
    sweep_result = ctrl.state.run_results["probe"]
    assert not np.isnan(sweep_result.signal[0]).any()
    assert not np.isnan(sweep_result.signal[1]).any()
    assert np.isnan(sweep_result.signal[2]).all()
    persist_all.assert_called_once_with()


def test_full_run_without_stop_emits_run_finished(monkeypatch):
    # control case: an un-cancelled run ends on RUN_FINISHED over every point.
    ctrl = _build_ready_controller()
    persist_all = MagicMock()
    monkeypatch.setattr(ctrl, "persist_all", persist_all)

    events: list[RunEvent] = []
    ctrl.bus.subscribe(RunStoppedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append(p.EVENT))

    points: list[int] = []
    ctrl.bus.subscribe(PointDonePayload, lambda p: points.append(p.idx))

    info = run_controller_to_completion(ctrl)

    assert events == [RunEvent.RUN_FINISHED]
    assert points == [0, 1, 2, 3, 4]
    assert info.point["flux_idx"] == 4
    persist_all.assert_called_once_with()


def test_run_operation_progress_is_live_until_terminal():
    ctrl = build_core()

    def wait_for_cancel(env, snapshot):
        del snapshot
        while env.should_stop is not None and not env.should_stop():
            time.sleep(0.001)
        return Patch()

    ctrl.add_node(make_builder("slow_probe", produce_fn=wait_for_cancel))
    ctrl.set_flux_values([0.0])
    token = ctrl.start_run()

    app = QApplication.instance()
    assert app is not None
    deadline = time.monotonic() + 3.0
    bars = ()
    while time.monotonic() < deadline:
        app.processEvents()
        bars = ctrl.get_operation_progress(token)
        if bars:
            break
        time.sleep(0.001)

    assert ctrl.is_running
    assert bars

    ctrl.stop_run("progress test cleanup")
    pump_controller_until_idle(ctrl)

    assert ctrl.get_operation_progress(token) == ()
