"""Saved workflow writers consume synthetic science payloads without experiments."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import h5py as h5
import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import DataVariable, load_grouped_labber_data, load_run_data, save_run_data
from zcu_tools.experiment.metadata import RunSnapshot
from zcu_tools.experiment import AxesSpec, Axis, ExpCfgModel, ZSpec
from zcu_tools.experiment.workflows import RunIdentity, RunPaths
from zcu_tools.experiment.workflows.artifacts import RunManifest
from zcu_tools.experiment.workflows.export import (
    PreparedFluxdepExport,
    load_trace,
    validate_destinations,
    write_outputs,
)

from tests._filesystem import symlink_or_skip
from tests._native_support import native_cfg, native_metadata, native_payload


@pytest.mark.parametrize("completion", ["complete", "unknown"])
def test_trace_requires_known_completion_but_not_known_start_time(
    tmp_path: Path, completion: str
) -> None:
    @dataclass
    class TraceResult:
        frequency: np.ndarray
        signal: np.ndarray

    spec = AxesSpec(
        axes=(Axis("frequency", "frequency", "Hz"),),
        z=ZSpec("signal", "signal", "V"),
        result_type=TraceResult,
        cfg_type=ExpCfgModel,
        tag="test/native",
        data_variable=DataVariable("readout"),
    )
    path = tmp_path / "trace.h5"
    save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(path, "r+") as file:
        file.attrs["format_version"] = "2.1"
        file.attrs["started_at"] = ""
        file.attrs["completion"] = completion
    if completion == "unknown":
        with pytest.raises(
            ValueError, match="Expected complete test/native acquisition"
        ):
            load_trace(tmp_path, path.name, spec)
    else:
        try:
            trace = load_trace(tmp_path, path.name, spec)
        except ValueError as error:
            pytest.fail(
                f"Complete trace does not require historical start evidence: {error}"
            )
        np.testing.assert_array_equal(trace.z, [1 + 2j, 3 - 4j])
        np.testing.assert_array_equal(trace.axes[0].values, [4e9, 5e9])


def inputs(
    tmp_path: Path,
    lifecycle: Literal["done", "cancelled", "failed"] = "done",
    run_id: str | None = None,
) -> tuple[RunManifest, PreparedFluxdepExport, RunSnapshot]:
    metadata = native_metadata()
    now = datetime(2026, 10, 12, tzinfo=UTC)
    manifest = TypeAdapter(RunManifest).validate_python(
        {
            "format": "zcu-workflow-run",
            "format_version": 1,
            "identity": RunIdentity(run_id or metadata.run_id, metadata.provenance),
            "workflow": "synthetic",
            "plan": {"points": 2},
            "initial_tunables": {},
            "roots": RunPaths(tmp_path / "metadata", tmp_path / "data"),
            "journal": tmp_path / "metadata" / "journal.jsonl",
            "started_at": now,
            "updated_at": now,
            "ended_at": now,
            "lifecycle": lifecycle,
            "reason": None,
        }
    )
    manifest.roots.data_root.mkdir(parents=True)
    payload = replace(native_payload(), representation="grouped")
    return manifest, PreparedFluxdepExport(payload, native_cfg()), metadata.snapshot


@pytest.mark.parametrize(
    "lifecycle,completion",
    [
        ("done", "complete"),
        ("cancelled", "stopped"),
        ("failed", "partial"),
    ],
)
def test_saved_export_roundtrip_overwrite_and_new_labber(
    tmp_path: Path,
    lifecycle: Literal["done", "cancelled", "failed"],
    completion: str,
):
    manifest, prepared, snapshot = inputs(tmp_path, lifecycle)
    attempts = [
        write_outputs(
            manifest, prepared, snapshot, merged=True, database=tmp_path / "database"
        )
        for _ in range(2)
    ]
    for sequence, (native, labber) in enumerate(attempts, start=1):
        assert native.status == labber.status == "saved"
        assert native.path == manifest.roots.data_root / "merged.h5"
        assert labber.path is not None
        assert labber.path.name == f"synthetic_a1b2c3_{sequence}.hdf5"
        assert native.path is not None
        stored = load_run_data(native.path)
        assert stored.metadata.completion == completion
        assert stored.metadata.snapshot == snapshot
        assert stored.metadata.experiment == "synthetic"
        assert stored.metadata.provenance == manifest.identity.provenance
        assert stored.cfg == prepared.cfg
        restored = load_grouped_labber_data(str(labber.path))
        for name, trace in prepared.payload.variables.items():
            np.testing.assert_array_equal(stored.payload.variables[name].z, trace.z)
            np.testing.assert_array_equal(restored.variables[name].z, trace.z)
    assert attempts[0][0].path == attempts[1][0].path
    assert attempts[0][1].path != attempts[1][1].path


@pytest.mark.parametrize("failure", ["native", "labber"])
def test_output_failures_retain_independent_successes(tmp_path: Path, failure: str):
    manifest, prepared, snapshot = inputs(tmp_path)
    database = tmp_path / "database"
    if failure == "native":
        (manifest.roots.data_root / "merged.h5").mkdir(parents=True)
    else:
        database.mkdir()
        (database / "Labber").write_text("blocked")
    outputs = write_outputs(
        manifest, prepared, snapshot, merged=True, database=database
    )
    assert [out.status for out in outputs] == {
        "native": ["failed", "saved"],
        "labber": ["saved", "failed"],
    }[failure]
    for out in outputs:
        assert out.path is not None
        assert (out.error is None) == (out.status == "saved")


def test_labber_allocation_failure_keeps_native_and_unallocated_path(tmp_path: Path):
    manifest, prepared, snapshot = inputs(tmp_path)
    database, outside = tmp_path / "database", tmp_path / "outside"
    database.mkdir()
    outside.mkdir()
    symlink_or_skip(database / "Labber", outside, target_is_directory=True)
    native, labber = write_outputs(
        manifest, prepared, snapshot, merged=True, database=database
    )
    assert native.status == "saved"
    assert labber.status == "failed" and labber.path is None and labber.error
    assert list(outside.iterdir()) == []


def test_opaque_workflow_id_keeps_native_writer_rejection_and_labber_success(
    tmp_path: Path,
):
    manifest, prepared, snapshot = inputs(tmp_path, run_id="opaque-workflow")
    native, labber = write_outputs(
        manifest, prepared, snapshot, merged=True, database=tmp_path / "database"
    )
    assert (
        native.status == "failed"
        and native.path == manifest.roots.data_root / "merged.h5"
    )
    assert native.error and "run_id" in native.error and "UTC timestamp" in native.error
    assert native.path is not None
    assert not native.path.exists()
    assert labber.status == "saved" and labber.path is not None and labber.error is None


def test_preflight_requires_only_selected_destinations(tmp_path: Path):
    validate_destinations(
        ("merged",), {"merged", "labber", "samples"}, result=None, database=None
    )
    with pytest.raises(ValueError, match="--database"):
        validate_destinations(("labber",), {"labber"}, result=None, database=None)
    with pytest.raises(ValueError, match="--result"):
        validate_destinations(("samples",), {"samples"}, result=None, database=None)
    with pytest.raises(ValueError, match="unique"):
        validate_destinations(
            ("merged", "merged"), {"merged"}, result=tmp_path, database=None
        )
