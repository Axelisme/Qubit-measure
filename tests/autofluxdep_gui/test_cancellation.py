"""Cooperative run-cancellation — the controller honours stop_run() mid-sweep.

These exercise the stateful cancel path: stop_run() flips the cooperative flag,
start_run breaks the sweep early at the next should_stop poll, and the run ends on
RUN_STOPPED (not RUN_FINISHED) leaving a partial InfoStore. The cancel mechanic is
independent of experiment physics, so a fake measurement Node (a deterministic
``make_builder`` double) drives the run — fast, no acquire, no Qt.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import ScalarSpec
from zcu_tools.gui.app.autofluxdep.events.run import (
    PointDonePayload,
    RunContinuedPayload,
    RunEvent,
    RunFailedPayload,
    RunFinishedPayload,
    RunPausedPayload,
    RunStartedPayload,
    RunStoppedPayload,
)
from zcu_tools.gui.app.autofluxdep.feedback.runtime import FeedbackSlotDecl
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency
from zcu_tools.gui.app.autofluxdep.services.result_io import load_node_result
from zcu_tools.gui.app.autofluxdep.services.run_store import (
    load_journal_events,
    load_manifest,
)
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.state import FluxDepState

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


def _feedback_age_builder(ages: list[tuple[int, int]]):
    slot = FeedbackSlotDecl(
        key="freq",
        kind="estimator",
        prefix="freq",
        default_strategy="last_good",
    )

    def produce(env, snapshot):
        del snapshot
        estimator = env.feedback.estimator("freq")
        assert estimator is not None
        if env.flux_idx == 0:
            estimator.observe(env.flux, 10.0)
        sample = estimator.estimate(env.flux)
        assert sample is not None
        ages.append((env.flux_idx, sample.age_queries))
        return Patch()

    builder = make_builder(
        "feedback_probe",
        schema_fields=(
            ("freq_enabled", ScalarSpec(label="enabled", type=bool), True),
            ("freq_strategy", ScalarSpec(label="strategy", type=str), "last_good"),
            ("freq_idw_k", ScalarSpec(label="idw k", type=int), 4),
            ("freq_idw_epsilon", ScalarSpec(label="idw epsilon", type=float), 1e-6),
            ("freq_decay_points", ScalarSpec(label="decay", type=float), 3.0),
        ),
        produce_fn=produce,
    )
    builder.feedback_slots = (slot,)
    return builder


def _deterministic_qubit_freq_builder():
    def result_factory(_schema, flux):
        return QubitFreqResult.allocate(
            np.asarray(flux, dtype=float),
            np.array([-1.0, 0.0, 1.0], dtype=float),
        )

    def produce(env, snapshot):
        del snapshot
        result = cast(QubitFreqResult, env.result)
        idx = int(env.flux_idx)
        signal = np.array([idx + 1.0, idx + 2.0, idx + 3.0], dtype=float)
        predict = 5000.0 + float(idx)
        result.signal[idx] = signal
        result.fit_curve[idx] = signal + 0.25
        result.predict_freq[idx] = predict
        result.fit_freq[idx] = predict + 0.125
        result.snr[idx] = 100.0 + float(idx)
        if env.round_hook is not None:
            env.round_hook(idx)
        return Patch({"qubit_freq": float(result.fit_freq[idx])})

    return make_builder(
        "qubit_freq",
        provides=("qubit_freq",),
        produce_fn=produce,
        result_factory=result_factory,
    )


def _stable_events(
    events: list[dict[str, Any]], event_type: str, keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {key: event[key] for key in keys}
        for event in events
        if event["type"] == event_type
    ]


def _load_fluxdep_export(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data_root = Path(manifest["paths"]["data_root"])
    export_path = data_root / manifest["exports"]["fluxdep_spectrum"]
    state = FluxDepState()
    name = LoadService(state).load_spectrum(str(export_path), spec_type="TwoTone")
    raw = state.spectrums[name].raw
    return np.asarray(raw["dev_values"], dtype=float), np.asarray(raw["signals"])


def _artifact_snapshot(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir / "manifest.json")
    events = load_journal_events(run_dir / "journal.jsonl")
    data_root = Path(manifest["paths"]["data_root"])
    node_file = data_root / manifest["files"]["nodes"][0]["path"]
    result = load_node_result(node_file, "qubit_freq")
    assert isinstance(result, QubitFreqResult)
    export_flux, export_signal = _load_fluxdep_export(manifest)
    return {
        "manifest": manifest,
        "node_rows": _stable_events(
            events,
            "node_row_written",
            (
                "flux_idx",
                "flux_value",
                "node",
                "node_type",
                "roles_written",
                "measurement_status",
                "provide_status",
                "patch",
                "provided_modules",
                "row_summary",
            ),
        ),
        "flux_commits": _stable_events(
            events,
            "flux_committed",
            (
                "flux_idx",
                "flux_value",
                "node_rows_written",
                "nodes_skipped",
                "info_keys",
            ),
        ),
        "signal": result.signal.copy(),
        "fit_curve": result.fit_curve.copy(),
        "fit_freq": result.fit_freq.copy(),
        "predict_freq": result.predict_freq.copy(),
        "snr": result.snr.copy(),
        "export_flux": export_flux,
        "export_signal": export_signal,
    }


def _run_deterministic_qubit_freq_artifact(
    *, pause_after_idx: int | None = None
) -> dict[str, Any]:
    ctrl = build_core()
    ensure_test_project(ctrl)
    ctrl.add_node(_deterministic_qubit_freq_builder())
    ctrl.set_flux_values([0.0, 0.25, 0.5])

    if pause_after_idx is not None:

        def request_pause_at_boundary(p: PointDonePayload) -> None:
            if p.idx == pause_after_idx:
                assert ctrl.request_pause()

        ctrl.bus.subscribe(PointDonePayload, request_pause_at_boundary)

    ctrl.start_run()
    pump_controller_until_idle(ctrl)
    if pause_after_idx is not None:
        assert ctrl.is_paused
        ctrl.continue_run()
        pump_controller_until_idle(ctrl)

    assert not ctrl.is_running
    assert not ctrl.is_paused
    return _artifact_snapshot(_latest_run_dir(ctrl))


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


def test_stop_after_operation_token_before_worker_start_is_honored(monkeypatch):
    ctrl = _build_ready_controller()
    captured: dict[str, Any] = {}
    produced: list[int] = []

    def produce(env, snapshot):
        del snapshot
        produced.append(env.flux_idx)
        return Patch()

    ctrl.state.nodes = []
    ctrl.add_node(make_builder("probe", produce_fn=produce))
    ctrl.set_flux_values([0.0, 0.5])

    def capture_submit(
        work: Callable[[], Any],
        *,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        run_in_pool: bool = True,
        enter: object = None,
    ) -> None:
        del run_in_pool, enter
        captured["work"] = work
        captured["on_done"] = on_done
        captured["on_error"] = on_error

    monkeypatch.setattr(ctrl._background_svc, "submit", capture_submit)

    events: list[RunEvent] = []
    ctrl.bus.subscribe(PointDonePayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunStartedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunStoppedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunFailedPayload, lambda p: events.append(p.EVENT))

    token = ctrl.start_run()
    assert ctrl.is_running
    assert produced == []

    assert ctrl.stop_run("cancel before worker starts")

    work = cast(Callable[[], object], captured["work"])
    on_done = cast(Callable[[object], None], captured["on_done"])
    on_done(work())

    assert produced == []
    assert events == [RunEvent.RUN_STARTED, RunEvent.RUN_STOPPED]
    result = ctrl.await_operation(token, timeout=0.0)
    assert result is not None
    assert result.outcome is not None
    assert result.outcome.status == "cancelled"
    assert result.feedback == "cancel before worker starts"

    manifest = load_manifest(_latest_run_dir(ctrl) / "manifest.json")
    assert manifest["terminal"]["status"] == "stopped"
    assert manifest["lifecycle"] == {"status": "stopped", "next_flux_idx": 0}


def test_stop_after_produce_before_commit_keeps_current_flux_cursor():
    ctrl = _build_ready_controller()
    produced: list[int] = []
    events: list[RunEvent] = []

    def produce(env, snapshot):
        del snapshot
        produced.append(env.flux_idx)
        ctrl.stop_run("stop before row commit")
        return Patch()

    ctrl.state.nodes = []
    ctrl.add_node(make_builder("probe", produce_fn=produce))
    ctrl.set_flux_values([0.0, 0.5])
    ctrl.bus.subscribe(PointDonePayload, lambda p: events.append(p.EVENT))
    ctrl.bus.subscribe(RunStoppedPayload, lambda p: events.append(p.EVENT))

    token = ctrl.start_run()
    pump_controller_until_idle(ctrl)

    assert produced == [0]
    assert events == [RunEvent.RUN_STOPPED]
    result = ctrl.await_operation(token, timeout=0.0)
    assert result is not None
    assert result.outcome is not None
    assert result.outcome.status == "cancelled"
    assert result.feedback == "stop before row commit"

    run_dir = _latest_run_dir(ctrl)
    manifest = load_manifest(run_dir / "manifest.json")
    assert manifest["terminal"]["status"] == "stopped"
    assert manifest["lifecycle"] == {"status": "stopped", "next_flux_idx": 0}
    event_types = [
        event["type"] for event in load_journal_events(run_dir / "journal.jsonl")
    ]
    assert "flux_committed" not in event_types


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


def test_pause_continue_preserves_feedback_query_age():
    paused_ages: list[tuple[int, int]] = []
    ctrl = build_core()
    ensure_test_project(ctrl)
    ctrl.add_node(_feedback_age_builder(paused_ages))
    ctrl.set_flux_values([0.0, 0.5, 1.0])

    def pause_after_second_point(p: PointDonePayload) -> None:
        if p.idx == 1:
            assert ctrl.request_pause()

    ctrl.bus.subscribe(PointDonePayload, pause_after_second_point)
    ctrl.start_run()
    pump_controller_until_idle(ctrl)
    assert ctrl.is_paused

    ctrl.continue_run()
    pump_controller_until_idle(ctrl)

    uninterrupted_ages: list[tuple[int, int]] = []
    uninterrupted = build_core()
    ensure_test_project(uninterrupted)
    uninterrupted.add_node(_feedback_age_builder(uninterrupted_ages))
    uninterrupted.set_flux_values([0.0, 0.5, 1.0])
    run_controller_to_completion(uninterrupted)

    assert paused_ages == uninterrupted_ages == [(0, 0), (1, 1), (2, 2)]


def test_pause_continue_artifact_matches_uninterrupted_run_for_committed_rows():
    uninterrupted = _run_deterministic_qubit_freq_artifact()
    paused = _run_deterministic_qubit_freq_artifact(pause_after_idx=1)

    assert uninterrupted["manifest"]["terminal"]["status"] == "finished"
    assert paused["manifest"]["terminal"]["status"] == "finished"
    assert uninterrupted["manifest"]["lifecycle"] == {
        "status": "finished",
        "next_flux_idx": 3,
    }
    assert paused["manifest"]["lifecycle"] == uninterrupted["manifest"]["lifecycle"]
    assert paused["node_rows"] == uninterrupted["node_rows"]
    assert paused["flux_commits"] == uninterrupted["flux_commits"]
    assert paused["manifest"]["exports"] == uninterrupted["manifest"]["exports"]

    np.testing.assert_allclose(paused["signal"], uninterrupted["signal"])
    np.testing.assert_allclose(paused["fit_curve"], uninterrupted["fit_curve"])
    np.testing.assert_allclose(paused["fit_freq"], uninterrupted["fit_freq"])
    np.testing.assert_allclose(paused["predict_freq"], uninterrupted["predict_freq"])
    np.testing.assert_allclose(paused["snr"], uninterrupted["snr"])
    np.testing.assert_allclose(paused["export_flux"], uninterrupted["export_flux"])
    np.testing.assert_allclose(
        paused["export_signal"], uninterrupted["export_signal"], equal_nan=True
    )


def test_pause_continue_progress_cursor_resumes_at_original_next_flux_idx():
    ctrl = _build_ready_controller()
    paused_next_indices: list[int] = []
    points_after_continue: list[int] = []
    continuing = False

    def on_point_done(p: PointDonePayload) -> None:
        if continuing:
            points_after_continue.append(p.idx)
        if p.idx == 1:
            assert ctrl.request_pause()

    ctrl.bus.subscribe(PointDonePayload, on_point_done)
    ctrl.bus.subscribe(
        RunPausedPayload, lambda p: paused_next_indices.append(p.next_flux_idx)
    )

    ctrl.start_run()
    pump_controller_until_idle(ctrl)

    assert ctrl.is_paused
    assert ctrl.next_flux_idx == 2
    assert paused_next_indices == [2]

    continuing = True
    ctrl.continue_run()
    pump_controller_until_idle(ctrl)

    assert points_after_continue == [2, 3, 4]
    manifest = load_manifest(_latest_run_dir(ctrl) / "manifest.json")
    assert manifest["lifecycle"] == {
        "status": "finished",
        "next_flux_idx": len(_FLUX),
    }


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
