"""Independent Coherence conversion retains numerical and writer contracts."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.datafile import DataVariable, load_grouped_labber_data, load_run_data
from zcu_tools.experiment.workflows.saved_run import read_manifest, read_points

from zcu_lab.workflows.coherence import CoherencePoint
from zcu_lab.workflows.coherence_export import export_run, prepare_export

from ._export_inputs import RUN_ID, synthetic_run
from ._sim_inputs import Inputs


def test_t1_adapter_keeps_actual_delay_rows_and_rejected_fit(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, plan, _identity, paths = synthetic_run(inputs, tmp_path, "stopped")
    assert engine.execute(RUN_ID).lifecycle == "stopped"
    points, missing = read_points(
        read_manifest(paths.metadata_root, "coherence_fluxdep"),
        TypeAdapter(CoherencePoint),
    )
    assert len(points) == 3 and missing == ()
    prepared = prepare_export(plan, points)
    variables = prepared.payload.variables
    assert tuple(variables) == ("t1_signal", "t1_time", "t1")
    np.testing.assert_array_equal(
        variables[DataVariable("t1_signal")].z,
        [
            [1 + 2j, 2 + 3j, complex(np.nan, np.nan)],
            [4 + 5j, 6 + 7j, complex(np.nan, np.nan)],
            [complex(np.nan, np.nan)] * 3,
            [complex(np.nan, np.nan)] * 3,
        ],
    )
    np.testing.assert_array_equal(
        variables[DataVariable("t1_time")].z,
        [[0, 1e-6, 2e-6], [0, 3e-6, np.nan], [np.nan] * 3, [np.nan] * 3],
    )
    np.testing.assert_array_equal(
        variables[DataVariable("t1")].z, [20e-6, np.nan, np.nan, np.nan]
    )
    assert [(a.name, a.unit) for a in variables[DataVariable("t1_signal")].axes] == [
        ("t1 Time Index", ""),
        ("Flux", "native"),
    ]
    np.testing.assert_array_equal(
        variables[DataVariable("t1_signal")].axes[0].values, [0, 1, 2]
    )
    np.testing.assert_array_equal(
        variables[DataVariable("t1_signal")].axes[1].values, [1, 1.1, 1.2, 1.3]
    )
    assert [(a.name, a.unit) for a in variables[DataVariable("t1")].axes] == [
        ("Flux", "native")
    ]
    assert prepared.cfg.values == plan.model_dump(mode="json")
    assert (prepared.cfg.cfg_type, prepared.cfg.schema_version) == (
        "CoherencePlan",
        "1.0",
    )
    with pytest.raises(ValueError, match="usable committed"):
        prepare_export(plan, ())


@pytest.mark.parametrize("terminal", ["done", "stopped", "failed", "aborted"])
def test_saved_export_roundtrip_overwrite_and_new_labber(
    inputs: Inputs,
    tmp_path: Path,
    terminal: Literal["done", "stopped", "failed", "aborted"],
) -> None:
    engine, plan, identity, paths = synthetic_run(inputs, tmp_path, terminal)
    assert engine.execute(RUN_ID).lifecycle == terminal
    canonical = engine.records(RUN_ID)
    reports = []
    for _ in range(2):
        report = export_run(
            paths.metadata_root,
            outputs=("merged", "labber"),
            database=tmp_path / "database",
        )
        reports.append(report)
        assert [item.status for item in report.outputs] == ["saved", "saved"]
        native = load_run_data(paths.data_root / "merged.h5")
        labber = load_grouped_labber_data(
            str(report.outputs[1].path),
            required_variables=("t1_signal", "t1_time", "t1"),
        )
        for name in map(DataVariable, ("t1_signal", "t1_time", "t1")):
            a, b = native.payload.variables[name], labber.variables[name]
            np.testing.assert_array_equal(a.z, b.z)
            assert [(axis.name, axis.unit) for axis in a.axes] == [
                (axis.name, axis.unit) for axis in b.axes
            ]
            assert (a.data.name, a.data.unit) == (b.data.name, b.data.unit)
            for native_axis, labber_axis in zip(a.axes, b.axes, strict=True):
                np.testing.assert_array_equal(native_axis.values, labber_axis.values)
        assert (
            native.metadata.run_id == RUN_ID
            and native.metadata.provenance == identity.provenance
        )
        assert (
            native.metadata.snapshot == plan.snapshot
            and native.metadata.experiment == "coherence_fluxdep"
        )
        assert native.cfg is not None and native.cfg.values == plan.model_dump(
            mode="json"
        )
        assert native.metadata.device_snapshot is None
        np.testing.assert_array_equal(
            native.payload.variables[DataVariable("t1")].z,
            [20e-6, np.nan, np.nan, np.nan],
        )
        assert native.payload.variables[DataVariable("t1_signal")].z.shape == (4, 3)
    assert (
        reports[0].outputs[0].path
        == reports[1].outputs[0].path
        == paths.data_root / "merged.h5"
    )
    assert reports[0].outputs[1].path != reports[1].outputs[1].path
    assert datetime.now(UTC).strftime("Labber/%Y/%m/Data_%m%d") in str(
        reports[0].outputs[1].path
    )
    assert engine.records(RUN_ID) == canonical
    status = engine.status()
    assert status is not None and status.lifecycle == terminal


@pytest.mark.parametrize("failure", ["native", "labber"])
def test_output_failures_retain_independent_successes(
    inputs: Inputs, tmp_path: Path, failure: str
) -> None:
    engine, _plan, _identity, paths = synthetic_run(inputs, tmp_path)
    engine.execute(RUN_ID)
    database = tmp_path / "database"
    if failure == "native":
        (paths.data_root / "merged.h5").mkdir()
    else:
        database.mkdir()
        (database / "Labber").write_text("blocked")
    report = export_run(
        paths.metadata_root, outputs=("merged", "labber"), database=database
    )
    assert [item.status for item in report.outputs] == {
        "native": ["failed", "saved"],
        "labber": ["saved", "failed"],
    }[failure]
    for out in report.outputs:
        if out.status == "saved":
            assert out.error is None and out.path is not None
            if out.format == "data_h5":
                assert load_run_data(out.path).metadata.finished_at is not None
            else:
                assert load_grouped_labber_data(str(out.path)).variables[
                    DataVariable("t1_signal")
                ].z.shape == (4, 3)
        else:
            assert out.error
    status = engine.status()
    assert status is not None and status.lifecycle == "done"


def test_labber_allocation_failure_keeps_native_and_unallocated_path(
    inputs: Inputs, tmp_path: Path
) -> None:
    engine, _plan, _identity, paths = synthetic_run(inputs, tmp_path)
    engine.execute(RUN_ID)
    database, outside = tmp_path / "database", tmp_path / "outside"
    database.mkdir()
    outside.mkdir()
    (database / "Labber").symlink_to(outside, target_is_directory=True)
    report = export_run(
        paths.metadata_root, outputs=("merged", "labber"), database=database
    )
    assert (
        report.outputs[0].status == "saved"
        and report.outputs[0].path == paths.data_root / "merged.h5"
    )
    assert load_run_data(paths.data_root / "merged.h5").payload.variables[
        DataVariable("t1_signal")
    ].z.shape == (4, 3)
    assert report.outputs[1].path is None
    assert report.outputs[1].status == "failed" and report.outputs[1].error
    assert list(outside.iterdir()) == []


def test_opaque_workflow_id_keeps_native_writer_rejection_and_labber_success(
    inputs: Inputs, tmp_path: Path
) -> None:
    run_id = "opaque-workflow"
    engine, _plan, _identity, paths = synthetic_run(inputs, tmp_path, run_id=run_id)
    assert engine.execute(run_id).lifecycle == "done"
    canonical = engine.records(run_id)
    native, labber = export_run(
        paths.metadata_root,
        outputs=("merged", "labber"),
        database=tmp_path / "database",
    ).outputs
    assert native.status == "failed" and native.path == paths.data_root / "merged.h5"
    assert (
        native.error
        and "ValueError" in native.error
        and "run_id" in native.error
        and "UTC timestamp" in native.error
    )
    assert not (paths.data_root / "merged.h5").exists()
    assert labber.status == "saved" and labber.path is not None and labber.error is None
    loaded = load_grouped_labber_data(
        str(labber.path), required_variables=("t1_signal", "t1_time", "t1")
    )
    np.testing.assert_array_equal(
        loaded.variables[DataVariable("t1_signal")].z,
        [
            [1 + 2j, 2 + 3j, complex(np.nan, np.nan)],
            [4 + 5j, 6 + 7j, complex(np.nan, np.nan)],
            [complex(np.nan, np.nan)] * 3,
            [complex(np.nan, np.nan)] * 3,
        ],
    )
    np.testing.assert_array_equal(
        loaded.variables[DataVariable("t1_time")].z,
        [[0, 1e-6, 2e-6], [0, 3e-6, np.nan], [np.nan] * 3, [np.nan] * 3],
    )
    np.testing.assert_array_equal(
        loaded.variables[DataVariable("t1")].z, [20e-6, np.nan, np.nan, np.nan]
    )
    status = engine.status()
    assert (
        status is not None
        and status.lifecycle == "done"
        and engine.records(run_id) == canonical
    )
