"""Cooperative run-cancellation — the controller honours stop_run() mid-sweep.

These exercise the stateful cancel path: stop_run() flips the cooperative flag,
start_run breaks the sweep early at the next should_stop poll, and the run ends on
RUN_STOPPED (not RUN_FINISHED) leaving a partial InfoStore. The cancel mechanic is
independent of experiment physics, so a fake measurement Node (a deterministic
``make_builder`` double) drives the run — fast, no acquire, no Qt.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.events.run import (
    PointDonePayload,
    RunContinuedPayload,
    RunEvent,
    RunFailedPayload,
    RunFinishedPayload,
    RunPausedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency
from zcu_tools.gui.app.autofluxdep.services.run_store import (
    load_journal_events,
    load_manifest,
)

from ._helpers import (
    ensure_test_project,
    make_builder,
    make_measurement_builder,
    pump_controller_until_idle,
    run_controller_to_completion,
)

_FLUX = [0.0, 0.25, 0.5, 0.75, 1.0]


def _build_ready_controller():
    ctrl = build_core()
    ensure_test_project(ctrl)
    ctrl.add_node(make_measurement_builder("probe"))
    ctrl.set_flux_values(_FLUX)
    return ctrl


def _latest_run_dir(ctrl) -> Path:
    project = ctrl.state.project
    assert project is not None
    runs = sorted((Path(project.result_dir) / "autofluxdep_runs").glob("*"))
    assert runs
    return runs[-1]


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
    ensure_test_project(ctrl)

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
    with pytest.raises(RuntimeError, match="context is locked while a run is active"):
        ctrl.context_control.set_md_attr("locked", 1)
    with pytest.raises(RuntimeError, match="device is locked while a run is active"):
        ctrl.device_control.forget_device("fake_flux")
    with pytest.raises(RuntimeError, match="predictor is locked while a run is active"):
        ctrl.predictor_control.clear_predictor()

    ctrl.stop_run("progress test cleanup")
    pump_controller_until_idle(ctrl)

    assert ctrl.get_operation_progress(token) == ()


def test_pause_then_continue_preserves_results_and_finalizes_artifact():
    ctrl = _build_ready_controller()

    events: list[RunEvent] = []
    ctrl.bus.subscribe(RunPausedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunContinuedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append(p.EVENT))

    points: list[int] = []

    def on_point_done(p: PointDonePayload) -> None:
        points.append(p.idx)
        if p.idx == 1:
            assert ctrl.request_pause()

    ctrl.bus.subscribe(PointDonePayload, on_point_done)

    token = ctrl.start_run()
    pump_controller_until_idle(ctrl)

    assert not ctrl.is_running
    assert ctrl.is_paused
    assert ctrl.run_status == "paused"
    assert ctrl.next_flux_idx == 2
    paused = ctrl.await_operation(token, timeout=0.0)
    assert paused is not None
    assert paused.outcome is not None
    assert paused.outcome.status == "cancelled"

    sweep_result = ctrl.state.run_results["probe"]
    assert not np.isnan(sweep_result.signal[0]).any()
    assert not np.isnan(sweep_result.signal[1]).any()
    assert np.isnan(sweep_result.signal[2]).all()

    with pytest.raises(RuntimeError, match="locked while a run is paused"):
        ctrl.set_flux_values([9.0])
    with pytest.raises(RuntimeError, match="setup is locked while a run is paused"):
        ctrl.setup_control.new_context()
    with pytest.raises(RuntimeError, match="context is locked while a run is paused"):
        ctrl.context_control.set_md_attr("locked", 1)
    with pytest.raises(RuntimeError, match="device is locked while a run is paused"):
        ctrl.device_control.forget_device("fake_flux")
    with pytest.raises(RuntimeError, match="predictor is locked while a run is paused"):
        ctrl.predictor_control.clear_predictor()

    token2 = ctrl.continue_run()
    assert ctrl.is_running
    pump_controller_until_idle(ctrl)

    assert not ctrl.is_running
    assert not ctrl.is_paused
    assert points == [0, 1, 2, 3, 4]
    assert events == [
        RunEvent.RUN_PAUSED,
        RunEvent.RUN_CONTINUED,
        RunEvent.RUN_FINISHED,
    ]
    finished = ctrl.await_operation(token2, timeout=0.0)
    assert finished is not None
    assert finished.outcome is not None
    assert finished.outcome.status == "finished"
    assert not np.isnan(sweep_result.signal).any()

    run_dir = _latest_run_dir(ctrl)
    manifest = load_manifest(run_dir / "manifest.json")
    assert manifest["terminal"]["status"] == "finished"
    assert manifest["lifecycle"]["status"] == "finished"
    assert manifest["lifecycle"]["next_flux_idx"] == len(_FLUX)
    event_types = [
        event["type"] for event in load_journal_events(run_dir / "journal.jsonl")
    ]
    assert "run_paused" in event_types
    assert "run_continued" in event_types
    assert event_types[-1] == "run_finalized"


def test_pause_then_continue_preserves_smoothing_state():
    ctrl = build_core()
    ensure_test_project(ctrl)
    raw_values = [10.0, 20.0, 30.0]

    def produce_x(env, _snapshot):
        return Patch({"x": raw_values[env.flux_idx]})

    ctrl.add_node(make_builder("raw_x", provides=("x",), produce_fn=produce_x))
    ctrl.add_node(
        make_builder(
            "smooth_consumer",
            optional=(Dependency("x", smooth="ewma", default=lambda: None),),
        )
    )
    ctrl.set_flux_values([0.0, 0.5, 1.0])

    def pause_after_second_point(p: PointDonePayload) -> None:
        if p.idx == 1:
            ctrl.request_pause()

    ctrl.bus.subscribe(PointDonePayload, pause_after_second_point)

    ctrl.start_run()
    pump_controller_until_idle(ctrl)
    assert ctrl.is_paused

    ctrl.continue_run()
    pump_controller_until_idle(ctrl)

    info = ctrl.last_run_info
    assert info is not None
    assert info.point_smoothed["x"] == pytest.approx(22.5)


def test_pause_flush_failure_records_terminal_failed_artifact(monkeypatch):
    from zcu_tools.gui.app.autofluxdep.run_session import RunSession

    ctrl = _build_ready_controller()
    events: list[RunEvent] = []
    ctrl.bus.subscribe(RunFailedPayload, lambda p: events.append(p.EVENT))

    def fail_mark_paused(self: RunSession) -> None:
        raise RuntimeError("pause flush failed")

    monkeypatch.setattr(RunSession, "mark_paused", fail_mark_paused)

    def pause_after_first_point(p: PointDonePayload) -> None:
        if p.idx == 0:
            ctrl.request_pause()

    ctrl.bus.subscribe(PointDonePayload, pause_after_first_point)
    token = ctrl.start_run()
    pump_controller_until_idle(ctrl)

    assert not ctrl.is_running
    assert not ctrl.is_paused
    failed = ctrl.await_operation(token, timeout=0.0)
    assert failed is not None
    assert failed.outcome is not None
    assert failed.outcome.status == "failed"
    assert events == [RunEvent.RUN_FAILED]

    run_dir = _latest_run_dir(ctrl)
    manifest = load_manifest(run_dir / "manifest.json")
    assert manifest["terminal"]["status"] == "failed"
    assert manifest["lifecycle"]["status"] == "failed"
    event_types = [
        event["type"] for event in load_journal_events(run_dir / "journal.jsonl")
    ]
    assert "run_failed" in event_types
    assert event_types[-1] == "run_finalized"


def test_abort_paused_run_finalizes_and_prevents_continue():
    ctrl = _build_ready_controller()

    def pause_after_first_point(p: PointDonePayload) -> None:
        if p.idx == 0:
            ctrl.request_pause()

    ctrl.bus.subscribe(PointDonePayload, pause_after_first_point)

    ctrl.start_run()
    pump_controller_until_idle(ctrl)
    assert ctrl.is_paused

    assert ctrl.stop_run("abort paused")
    assert not ctrl.is_paused
    assert ctrl.run_status == "idle"
    with pytest.raises(RuntimeError, match="not paused"):
        ctrl.continue_run()

    manifest = load_manifest(_latest_run_dir(ctrl) / "manifest.json")
    assert manifest["terminal"]["status"] == "stopped"
    assert manifest["lifecycle"]["status"] == "stopped"
