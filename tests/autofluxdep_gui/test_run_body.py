"""Run-body mechanics — the controller drives the orchestrator over providers.

These exercise the run *wiring* (predictor Service prepended, RunEnv threaded,
Patch merged into the InfoStore, the ModuleSource bridge), NOT experiment
physics: the per-Node real-acquire fit is covered against the flux-aware MockSoc
by the ``test_*_acquire.py`` integration tests. A fake measurement Node (a
``make_builder`` double whose ``produce`` returns a deterministic Patch) keeps the
mechanics fast and decoupled from any acquire.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency
from zcu_tools.gui.app.autofluxdep.services.result_io import load_node_result
from zcu_tools.gui.app.autofluxdep.services.run_store import (
    load_journal_events,
    load_manifest,
)
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo

from ._helpers import (
    make_builder,
    make_measurement_builder,
    run_controller_to_completion,
)


def _consume_predict(env, snapshot):
    """A fake measurement Node: echo the Service's predict_freq into a Patch.

    Reads ``predict_freq`` (provided by the prepended predictor Service) and the
    flux from the env, and emits a deterministic ``measured`` key so the run wiring
    (snapshot projection → produce → Patch merge) is observable without physics."""
    predicted = float(snapshot["predict_freq"])
    patch = Patch()
    patch.set("measured", predicted + 0.5)  # a fixed, deterministic offset
    return patch


def _fake_consumer():
    return make_builder(
        "consumer",
        provides=("measured",),
        requires=(Dependency("predict_freq"),),
        produce_fn=_consume_predict,
    )


def _project(tmp_path):
    return ProjectInfo(
        chip_name="chip",
        qub_name="q1",
        result_dir=str(tmp_path),
        database_path=str(tmp_path / "Database" / "chip" / "q1"),
        params_path=str(tmp_path / "params.json"),
    )


def _latest_run_dir(tmp_path):
    runs = sorted((tmp_path / "autofluxdep_runs").glob("*"))
    assert runs
    return runs[-1]


def test_controller_run_drives_predictor_service_then_consumer():
    # the controller prepends the predictor Service and runs the user node after it;
    # the consumer reads the Service's predict_freq and the final InfoStore carries
    # both keys — the full predictor-Service-then-Node run wiring.
    ctrl = build_core()
    ctrl.add_node(_fake_consumer())
    ctrl.set_flux_values([0.0, 1.0])
    info = run_controller_to_completion(ctrl)
    # the predictor Service produced predict_freq at the last point, and the
    # consumer produced its derived key off it
    assert "predict_freq" in info.point
    assert info.point["measured"] == info.point["predict_freq"] + 0.5


def test_controller_run_writes_artifact_manifest_journal_and_node_hdf5(tmp_path):
    ctrl = build_core(project=_project(tmp_path))
    ctrl.add_node(make_measurement_builder("probe"))
    ctrl.set_flux_values([0.0, 1.0])

    run_controller_to_completion(ctrl)

    run_dir = _latest_run_dir(tmp_path)
    manifest = load_manifest(run_dir / "manifest.json")
    assert manifest["terminal"]["status"] == "finished"
    assert manifest["files"]["nodes"][0]["name"] == "probe"
    assert manifest["paths"]["metadata_root"] == str(run_dir)
    data_root = Path(manifest["paths"]["data_root"])
    assert str(data_root).endswith("Database/chip/q1/autofluxdep_runs/" + run_dir.name)
    events = load_journal_events(run_dir / "journal.jsonl")
    assert [event["type"] for event in events] == [
        "node_row_written",
        "flux_committed",
        "node_row_written",
        "flux_committed",
        "run_finalized",
    ]
    node_result = load_node_result(
        data_root / manifest["files"]["nodes"][0]["path"], "probe"
    )
    assert not np.isnan(node_result.signal[0]).any()


def test_disabled_node_is_omitted_from_run_results_and_artifact(tmp_path):
    ctrl = build_core(project=_project(tmp_path))
    ctrl.add_node(make_measurement_builder("enabled_probe"))
    ctrl.add_node(make_measurement_builder("disabled_probe"))
    ctrl.set_node_enabled(1, False)
    ctrl.set_flux_values([0.0, 1.0])

    run_controller_to_completion(ctrl)

    assert set(ctrl.state.run_results) == {"enabled_probe"}
    run_dir = _latest_run_dir(tmp_path)
    manifest = load_manifest(run_dir / "manifest.json")
    assert [node["name"] for node in manifest["workflow"]["nodes"]] == ["enabled_probe"]
    assert [node["name"] for node in manifest["files"]["nodes"]] == ["enabled_probe"]
    events = load_journal_events(run_dir / "journal.jsonl")
    event_nodes = {event.get("node") for event in events if "node" in event}
    assert event_nodes == {"enabled_probe"}


def test_dry_run_omits_disabled_nodes():
    called: list[str] = []

    def record(env, snapshot):
        del env, snapshot
        called.append("disabled")
        return Patch()

    ctrl = build_core()
    ctrl.add_node(make_builder("disabled", produce_fn=record))
    ctrl.set_node_enabled(0, False)
    ctrl.set_flux_values([0.0])

    ctrl.dry_run()

    assert called == []


def test_run_event_bus_payloads_emit_on_main_thread(qapp):
    from zcu_tools.gui.app.autofluxdep.events.run import (
        NodeEnteredPayload,
        PointDonePayload,
        RunFailedPayload,
        RunFinishedPayload,
        RunStartedPayload,
        RunStoppedPayload,
    )

    main_thread = threading.get_ident()
    ctrl = build_core()
    ctrl.add_node(_fake_consumer())
    ctrl.set_flux_values([0.0, 1.0])
    seen: list[tuple[str, int]] = []

    def record(label: str) -> Callable[[object], None]:
        def _inner(_payload: object) -> None:
            seen.append((label, threading.get_ident()))

        return _inner

    ctrl.bus.subscribe(RunStartedPayload, record("started"))
    ctrl.bus.subscribe(NodeEnteredPayload, record("node"))
    ctrl.bus.subscribe(PointDonePayload, record("point"))
    ctrl.bus.subscribe(RunFinishedPayload, record("finished"))
    ctrl.bus.subscribe(RunStoppedPayload, record("stopped"))
    ctrl.bus.subscribe(RunFailedPayload, record("failed"))

    run_controller_to_completion(ctrl)
    qapp.processEvents()
    qapp.processEvents()

    labels = [label for label, _thread_id in seen]
    assert "started" in labels
    assert "node" in labels
    assert labels.count("point") == 2
    assert "finished" in labels
    assert all(thread_id == main_thread for _label, thread_id in seen)


def test_run_event_emitter_uses_direct_path_on_owner_thread(qapp):
    from zcu_tools.gui.app.autofluxdep.controller import _RunEventEmitter
    from zcu_tools.gui.app.autofluxdep.events.run import (
        NodeEnteredPayload,
        PointDonePayload,
    )

    main_thread = threading.get_ident()
    ctrl = build_core()
    seen: list[tuple[str, int | str, int]] = []
    ctrl.bus.subscribe(
        PointDonePayload,
        lambda p: seen.append(("point", p.idx, threading.get_ident())),
    )
    ctrl.bus.subscribe(
        NodeEnteredPayload,
        lambda p: seen.append(("node", p.name, threading.get_ident())),
    )

    emitter = _RunEventEmitter(ctrl)
    emitter.emit_point_done(3)
    emitter.emit_node_entered("consumer", 3)

    assert seen == [("point", 3, main_thread), ("node", "consumer", main_thread)]
    assert ctrl._cur_idx == 3


def test_run_threads_flux_into_env():
    # the run threads each flux point's value into the Node's RunEnv: the consumer
    # records env.flux per point, so the recorded sequence matches the sweep.
    seen: list[float] = []

    def record_flux(env, snapshot):
        del snapshot
        seen.append(env.flux)
        return Patch()

    ctrl = build_core()
    ctrl.add_node(
        make_builder(
            "recorder", requires=(Dependency("predict_freq"),), produce_fn=record_flux
        )
    )
    ctrl.set_flux_values([0.0, 0.5, 1.0])
    run_controller_to_completion(ctrl)
    assert seen == [0.0, 0.5, 1.0]


def test_produce_exception_fails_run_gracefully(monkeypatch):
    # a Node whose produce raises (e.g. an unconfigured real acquire Fast-Failing)
    # must NOT propagate out of the run: the orchestrator catches it, the run ends
    # on RUN_FAILED (not RUN_FINISHED), the controller unlocks, and the error is
    # carried on the payload. This is what stops a GUI run worker QThread aborting.
    from zcu_tools.gui.app.autofluxdep.events.run import (
        RunFailedPayload,
        RunFinishedPayload,
    )

    def boom(env, snapshot):
        del env, snapshot
        raise RuntimeError("node not configured")

    ctrl = build_core()
    ctrl.add_node(
        make_builder("broken", requires=(Dependency("predict_freq"),), produce_fn=boom)
    )
    ctrl.set_flux_values([0.0, 1.0])
    persist_all = MagicMock()
    monkeypatch.setattr(ctrl, "persist_all", persist_all)

    events: list[str] = []
    ctrl.bus.subscribe(RunFailedPayload, lambda p: events.append(f"failed:{p.message}"))
    ctrl.bus.subscribe(RunFinishedPayload, lambda p: events.append("finished"))

    run_controller_to_completion(ctrl)  # must not raise

    assert len(events) == 1 and events[0].startswith("failed:")
    assert "node not configured" in events[0]
    assert not ctrl.is_running  # the controller unlocked
    persist_all.assert_called_once_with()


def test_controller_failed_run_finalizes_artifact(tmp_path):
    def boom(env, snapshot):
        del env, snapshot
        raise RuntimeError("node not configured")

    ctrl = build_core(project=_project(tmp_path))
    ctrl.add_node(
        make_builder("broken", requires=(Dependency("predict_freq"),), produce_fn=boom)
    )
    ctrl.set_flux_values([0.0])

    run_controller_to_completion(ctrl)

    run_dir = _latest_run_dir(tmp_path)
    assert load_manifest(run_dir / "manifest.json")["terminal"]["status"] == "failed"
    events = load_journal_events(run_dir / "journal.jsonl")
    assert [event["type"] for event in events] == [
        "node_failed",
        "run_failed",
        "run_finalized",
    ]


def test_operation_begin_failure_finalizes_created_artifact(tmp_path, monkeypatch):
    ctrl = build_core(project=_project(tmp_path))
    ctrl.add_node(make_measurement_builder("probe"))
    ctrl.set_flux_values([0.0])

    def fail_begin(_spec):
        raise RuntimeError("operation gate closed")

    monkeypatch.setattr(ctrl._runner, "begin", fail_begin)

    with pytest.raises(RuntimeError, match="operation gate closed"):
        ctrl.start_run()

    run_dir = _latest_run_dir(tmp_path)
    manifest = load_manifest(run_dir / "manifest.json")
    assert manifest["terminal"]["status"] == "failed"
    assert not ctrl.is_running
    events = load_journal_events(run_dir / "journal.jsonl")
    assert [event["type"] for event in events] == ["run_failed", "run_finalized"]


def test_ml_module_source_returns_none_on_absent():
    # the orchestrator's ModuleSource contract is "None if absent", but
    # ModuleLibrary.get_module raises — the adapter start_run threads in must
    # bridge that so an absent module dep falls back instead of crashing the run.
    from zcu_tools.gui.app.autofluxdep.controller import _MlModuleSource
    from zcu_tools.meta_tool import ModuleLibrary

    source = _MlModuleSource(ModuleLibrary())
    assert source.get_module("not_a_module") is None  # must not raise
