"""Cooperative run-cancellation — the controller honours stop_run() mid-sweep.

These exercise the stateful cancel path: stop_run() flips the cooperative flag,
start_run breaks the sweep early at the next should_stop poll, and the run ends on
RUN_STOPPED (not RUN_FINISHED) leaving a partial InfoStore. The cancel mechanic is
independent of experiment physics, so a fake measurement Node (a deterministic
``make_builder`` double) drives the run — fast, no acquire, no Qt.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.events.run import (
    PointDonePayload,
    RunEvent,
    RunFinishedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch

from ._helpers import make_builder

_FLUX = [0.0, 0.25, 0.5, 0.75, 1.0]


def _build_ready_controller():
    ctrl = build_core()
    ctrl.add_node(make_builder("probe", produce_fn=lambda env, snap: Patch()))
    ctrl.set_flux_values(_FLUX)
    return ctrl


def test_stop_run_mid_sweep_emits_run_stopped_not_finished():
    ctrl = _build_ready_controller()

    events: list[RunEvent] = []
    ctrl.bus.subscribe(RunStoppedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append(p.EVENT))

    points: list[int] = []

    def on_point_done(p: PointDonePayload) -> None:
        points.append(p.idx)
        if len(points) == 2:  # request cancel after the 2nd point completes
            ctrl.stop_run()

    ctrl.bus.subscribe(PointDonePayload, on_point_done)

    info = ctrl.start_run()

    # the run ended on RUN_STOPPED, and RUN_FINISHED never fired
    assert events == [RunEvent.RUN_STOPPED]
    # only points 0 and 1 ran; the sweep broke before point 2 of 5
    assert points == [0, 1]
    # the returned InfoStore is partial: it reflects the last completed point
    assert info.point["flux_idx"] == 1


def test_full_run_without_stop_emits_run_finished():
    # control case: an un-cancelled run ends on RUN_FINISHED over every point.
    ctrl = _build_ready_controller()

    events: list[RunEvent] = []
    ctrl.bus.subscribe(RunStoppedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append(p.EVENT))

    points: list[int] = []
    ctrl.bus.subscribe(PointDonePayload, lambda p: points.append(p.idx))

    info = ctrl.start_run()

    assert events == [RunEvent.RUN_FINISHED]
    assert points == [0, 1, 2, 3, 4]
    assert info.point["flux_idx"] == 4
