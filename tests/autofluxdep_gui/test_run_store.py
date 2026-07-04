"""RunStore manifest/journal artifact tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.cfg import ScalarSpec
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult, Sweep1DResult
from zcu_tools.gui.app.autofluxdep.orchestrator import InfoStore, SkipReason
from zcu_tools.gui.app.autofluxdep.services import run_store as run_store_module
from zcu_tools.gui.app.autofluxdep.services.result_io import load_node_result
from zcu_tools.gui.app.autofluxdep.services.run_store import (
    RunStore,
    load_journal_events,
    load_manifest,
)
from zcu_tools.gui.app.autofluxdep.state import ProjectInfo
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.state import FluxDepState

from ._helpers import make_builder, place


def _project(tmp_path: Path) -> ProjectInfo:
    return ProjectInfo(
        chip_name="chip",
        qub_name="q1",
        result_dir=str(tmp_path),
        database_path=str(tmp_path / "Database" / "chip" / "q1"),
        params_path=str(tmp_path / "params.json"),
    )


def _node_and_result():
    node = place(make_builder("probe", provides=("fit",)))
    result = Sweep1DResult.allocate(
        np.array([0.0, 0.5], dtype=float),
        np.array([1.0, 2.0], dtype=float),
        x_label="time",
    )
    result.signal[0] = [1.0, 2.0]
    result.fit_curve[0] = [1.5, 2.5]
    result.fit_value[0] = 3.0
    result.snr[0] = 4.0
    return node, result


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_run_store_writes_manifest_node_row_journal_and_finalize(tmp_path):
    node, result = _node_and_result()
    cfg_snapshot = {
        "base_cfg": {"acquire": {"reps": 10}},
        "override_plan": [
            {
                "path": "acquire.reps",
                "mode": "all_points",
                "source": "generation.test",
                "reason": "test override",
            }
        ],
    }
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0, 0.5],
        flux_device_name="fake_flux",
        nodes=[node],
        results={"probe": result},
        cfg_snapshots={"probe": cfg_snapshot},
    )
    patch = Patch({"fit": 3.0})

    store.write_node_row("probe", 0, patch, InfoStore(point={"flux_idx": 0}))
    store.commit_flux(0, 0.0, InfoStore(point={"flux_idx": 0, "fit": 3.0}))
    store.finalize("finished")

    manifest = load_manifest(store.manifest_path)
    assert manifest["format_version"] == 1
    assert manifest["artifact_kind"] == "zcu_tools.autofluxdep.run"
    assert manifest["terminal"]["status"] == "finished"
    assert manifest["project"]["result_dir"] == str(tmp_path)
    assert manifest["project"]["database_path"] == str(
        tmp_path / "Database" / "chip" / "q1"
    )
    assert manifest["paths"]["metadata_root"] == str(store.run_dir)
    assert manifest["paths"]["data_root"] == str(store.data_dir)
    assert manifest["files"]["nodes"][0]["path"].startswith("nodes/000-probe")
    workflow_node = manifest["workflow"]["nodes"][0]
    assert workflow_node["cfg"] == node.schema.to_persisted_raw()
    assert workflow_node["cfg_hash"] == f"sha256:{_sha256_json(workflow_node['cfg'])}"
    assert workflow_node["base_cfg"] == cfg_snapshot["base_cfg"]
    assert workflow_node["override_plan"] == cfg_snapshot["override_plan"]
    assert manifest["workflow"]["workflow_hash"] == (
        "sha256:"
        + _sha256_json(
            {
                "nodes": manifest["workflow"]["nodes"],
                "flux": manifest["workflow"]["flux"],
            }
        )
    )
    assert manifest["reports"]["markdown"] == "report.md"
    report = (store.run_dir / "report.md").read_text(encoding="utf-8")
    assert "- Run finalized events: 0" in report
    assert "- Report markdown: report.md" in report
    assert f"- Data root: {store.data_dir}" in report

    events = load_journal_events(store.run_dir / "journal.jsonl")
    assert [event["type"] for event in events] == [
        "node_row_written",
        "flux_committed",
        "run_finalized",
    ]
    assert events[0]["patch"] == {"fit": 3.0}
    assert events[-1]["terminal_status"] == "finished"
    assert events[-1]["exports"] == manifest["exports"]
    assert events[-1]["reports"] == manifest["reports"]
    assert events[-1]["writer_errors"] == []
    assert events[-1]["export_errors"] == []
    assert events[-1]["report_errors"] == []

    assert not (store.run_dir / manifest["files"]["nodes"][0]["path"]).exists()
    assert (store.data_dir / manifest["files"]["nodes"][0]["path"]).is_file()
    loaded = load_node_result(
        store.data_dir / manifest["files"]["nodes"][0]["path"], "probe"
    )
    assert isinstance(loaded, Sweep1DResult)
    np.testing.assert_allclose(loaded.signal[0], [1.0, 2.0])
    assert (store.run_dir / "report.md").is_file()


def test_run_store_mark_paused_flushes_without_finalizing(tmp_path):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0, 0.5],
        flux_device_name="fake_flux",
        nodes=[node],
        results={"probe": result},
    )
    store.write_node_row("probe", 0, Patch({"fit": 3.0}), InfoStore())
    store.commit_flux(0, 0.0, InfoStore(point={"flux_idx": 0, "fit": 3.0}))

    store.mark_paused(1)

    manifest = load_manifest(store.manifest_path)
    assert manifest["terminal"]["status"] == "running"
    assert manifest["lifecycle"] == {"status": "paused", "next_flux_idx": 1}
    assert manifest["reports"] == {}
    assert not (store.run_dir / "report.md").exists()
    events = load_journal_events(store.run_dir / "journal.jsonl")
    assert [event["type"] for event in events] == [
        "node_row_written",
        "flux_committed",
        "run_paused",
    ]
    assert events[-1]["next_flux_idx"] == 1

    store.finalize("stopped", next_flux_idx=1)


def test_run_store_rejects_nonfinite_flux_values_before_manifest(tmp_path):
    with pytest.raises(TypeError, match="workflow flux is not strict JSON-safe"):
        RunStore.create(
            project=_project(tmp_path),
            flux_values=[np.nan],
            flux_device_name="fake_flux",
            nodes=[],
            results={},
        )


def test_run_store_rejects_nonfinite_manifest_cfg(tmp_path):
    node = place(
        make_builder(
            "probe",
            schema_fields=(("gain", ScalarSpec(label="gain", type=float), 0.1),),
        ),
        gain=np.nan,
    )

    with pytest.raises(TypeError, match="node cfg 'probe' is not strict JSON-safe"):
        RunStore.create(
            project=_project(tmp_path),
            flux_values=[0.0],
            flux_device_name="fake_flux",
            nodes=[node],
            results={},
        )


def test_run_store_records_skip_failure_and_run_failed(tmp_path):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0, 0.5],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )
    store.record_node_skipped(
        "probe",
        0,
        SkipReason(missing_info_keys=("qubit_freq",), missing_modules=("readout",)),
    )
    error = RuntimeError("boom")
    store.record_node_failed("probe", 1, error, "produce")
    store.record_run_failed(error, flux_idx=1, node="probe", stage="produce")
    store.finalize("failed", error=error)

    events = load_journal_events(store.run_dir / "journal.jsonl")
    assert [event["type"] for event in events] == [
        "node_skipped",
        "node_failed",
        "run_failed",
        "run_finalized",
    ]
    assert events[0]["reason"]["missing_info_keys"] == ["qubit_freq"]
    assert events[1]["stage"] == "produce"
    assert events[2]["message"] == "boom"
    assert load_manifest(store.manifest_path)["terminal"]["status"] == "failed"


def test_run_store_rejects_non_json_safe_module_snapshot(tmp_path):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )
    patch = Patch(modules={"bad_module": object()})

    with pytest.raises(TypeError, match="provided module"):
        store.write_node_row("probe", 0, patch, InfoStore())

    store.close_writers(finalize=True)
    manifest = load_manifest(store.manifest_path)
    loaded = load_node_result(
        store.data_dir / manifest["files"]["nodes"][0]["path"], "probe"
    )
    assert isinstance(loaded, Sweep1DResult)
    assert np.isnan(loaded.signal[0]).all()
    assert load_journal_events(store.run_dir / "journal.jsonl") == []


def test_run_store_finalize_records_report_failure_after_journal_event(
    tmp_path, monkeypatch
):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )

    def fail_report(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("report boom")

    monkeypatch.setattr(run_store_module, "write_markdown_report", fail_report)

    with pytest.raises(RuntimeError, match="report boom"):
        store.finalize("finished")

    manifest = load_manifest(store.manifest_path)
    assert manifest["terminal"]["status"] == "finished"
    assert "report boom" in manifest["terminal"]["error"]
    assert manifest["reports"] == {}
    events = load_journal_events(store.run_dir / "journal.jsonl")
    assert events[-1]["type"] == "run_finalized"
    assert events[-1]["reports"] == {}
    assert events[-1]["report_errors"] == ["report boom"]


def test_run_store_finalize_surfaces_writer_finalize_runtime_error(
    tmp_path, monkeypatch
):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )

    def fail_finalize() -> None:
        raise RuntimeError("flush failed")

    monkeypatch.setattr(store._writers["probe"], "finalize", fail_finalize)

    with pytest.raises(RuntimeError, match="flush failed"):
        store.finalize("finished")

    manifest = load_manifest(store.manifest_path)
    assert manifest["terminal"]["status"] == "finished"
    assert "flush failed" in manifest["terminal"]["error"]
    events = load_journal_events(store.run_dir / "journal.jsonl")
    assert events[-1]["type"] == "run_finalized"
    assert events[-1]["writer_errors"] == ["probe: flush failed"]


def test_run_store_close_writers_ignores_explicit_already_closed(tmp_path):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )

    store.close_writers(finalize=True)
    store.close_writers(finalize=True)


def test_run_store_records_raw_row_with_nan_fit_as_completed_measurement(tmp_path):
    node, result = _node_and_result()
    result.fit_value[0] = np.nan
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )

    store.write_node_row("probe", 0, Patch(), InfoStore())

    event = load_journal_events(store.run_dir / "journal.jsonl")[0]
    assert event["measurement_status"] == "completed"
    assert event["provide_status"] == "empty_patch"
    assert "signal" in event["roles_written"]
    assert event["row_summary"]["fit_value"] is None


def test_run_store_rejects_nonfinite_patch_value_before_hdf5_write(tmp_path):
    node, result = _node_and_result()
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[0.0],
        flux_device_name=None,
        nodes=[node],
        results={"probe": result},
    )
    patch = Patch({"fit": np.nan})

    with pytest.raises(TypeError, match="patch values"):
        store.write_node_row("probe", 0, patch, InfoStore())

    store.close_writers(finalize=True)
    manifest = load_manifest(store.manifest_path)
    loaded = load_node_result(
        store.data_dir / manifest["files"]["nodes"][0]["path"], "probe"
    )
    assert isinstance(loaded, Sweep1DResult)
    assert np.isnan(loaded.signal[0]).all()
    assert load_journal_events(store.run_dir / "journal.jsonl") == []


def test_run_store_create_rejects_unsupported_result_type(tmp_path):
    node, _result = _node_and_result()

    with pytest.raises(TypeError, match="unsupported autofluxdep Result type"):
        RunStore.create(
            project=_project(tmp_path),
            flux_values=[0.0],
            flux_device_name=None,
            nodes=[node],
            results={"probe": object()},
        )


def test_qubit_freq_export_excludes_memory_rows_without_journal_commit(tmp_path):
    node = place(make_builder("qubit_freq", provides=("qubit_freq",)))
    result = QubitFreqResult.allocate(
        np.array([1.0, 0.0], dtype=float),
        np.array([-1.0, 0.0, 1.0], dtype=float),
    )
    result.predict_freq[:] = [5001.0, 6001.0]
    result.signal[0] = [10.0, 11.0, 12.0]
    result.signal[1] = [20.0, 21.0, 22.0]
    store = RunStore.create(
        project=_project(tmp_path),
        flux_values=[1.0, 0.0],
        flux_device_name=None,
        nodes=[node],
        results={"qubit_freq": result},
    )

    store.write_node_row("qubit_freq", 0, Patch({"qubit_freq": 5001.0}), InfoStore())
    store.finalize("finished")

    manifest = load_manifest(store.manifest_path)
    export_path = store.data_dir / manifest["exports"]["fluxdep_spectrum"]
    state = FluxDepState()
    name = LoadService(state).load_spectrum(str(export_path), spec_type="TwoTone")
    raw = state.spectrums[name].raw

    np.testing.assert_allclose(raw["dev_values"], [0.0, 1.0])
    np.testing.assert_allclose(raw["signals"][0].real, np.nan, equal_nan=True)
    np.testing.assert_allclose(raw["signals"][1].real, [10.0, 11.0, 12.0])


def test_manifest_and_journal_unknown_versions_fast_fail(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"format_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_manifest(manifest)

    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps({"event_version": 2}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="event_version"):
        load_journal_events(journal)
