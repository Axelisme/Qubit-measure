"""Saved workflow writers consume synthetic science payloads without experiments."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from pydantic import TypeAdapter
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import (
    Axis as ContainerAxis,
)
from zcu_tools.datafile import (
    Container,
    DataVariable,
    LabberMetadata,
    Variable,
    load,
    load_grouped_labber_data,
    save,
)
from zcu_tools.experiment import AxesSpec, Axis, ZSpec
from zcu_tools.experiment.metadata import CfgSnapshot, RunSnapshot
from zcu_tools.experiment.native_persistence import (
    PreparedRunData,
    metadata_from_container,
)
from zcu_tools.experiment.workflows import RunIdentity, RunPaths
from zcu_tools.experiment.workflows.artifacts import RunManifest
from zcu_tools.experiment.workflows.export import (
    load_trace,
    validate_destinations,
    write_outputs,
)

from tests._filesystem import symlink_or_skip
from tests._native_support import native_cfg, native_container, native_metadata


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
        cfg_type=ConfigBase,
        tag="test/native",
        data_variable=DataVariable("readout"),
    )
    path = tmp_path / "trace.h5"
    metadata = native_metadata().model_copy(
        update={"started_at": None, "completion": completion}
    )
    container = native_container(metadata=metadata)
    save(
        path,
        replace(
            container,
            docs={
                **container.docs,
                "cfg": {
                    "values": {},
                    "cfg_type": "ConfigBase",
                    "schema_version": "1.0",
                },
            },
        ),
    )
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


@dataclass
class SyntheticTraceResult:
    frequency: np.ndarray
    flux: np.ndarray
    signal: np.ndarray


@pytest.fixture
def trace_spec() -> AxesSpec[SyntheticTraceResult, ConfigBase]:
    return AxesSpec(
        axes=(Axis("frequency", "frequency", "Hz"), Axis("flux", "Flux", "V")),
        z=ZSpec("signal", "signal", "V"),
        result_type=SyntheticTraceResult,
        cfg_type=ConfigBase,
        tag="test/native",
        data_variable=DataVariable("readout"),
    )


@pytest.fixture
def stopped_trace() -> Container:
    container = native_container(
        metadata=native_metadata().model_copy(update={"completion": "stopped"})
    )
    return replace(
        container,
        vars={
            "readout": Variable(
                np.array([[1 + 2j, 3 - 4j, 5j], [6 + 7j, complex(np.nan, np.nan), 9j]]),
                (
                    ContainerAxis("Flux", "V", np.array([0.1, 0.2])),
                    ContainerAxis("frequency", "Hz", np.array([4e9, 5e9, 6e9])),
                ),
                "V",
            )
        },
        docs={
            **container.docs,
            "cfg": {"values": {}, "cfg_type": "ConfigBase", "schema_version": "1.0"},
        },
    )


def test_trace_rejects_stopped_by_default(
    tmp_path: Path,
    trace_spec: AxesSpec[SyntheticTraceResult, ConfigBase],
    stopped_trace: Container,
) -> None:
    path = tmp_path / "stopped.h5"
    save(path, stopped_trace)
    with pytest.raises(
        ValueError, match="Expected complete test/native acquisition"
    ) as error:
        load_trace(tmp_path, path.name, trace_spec)
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    "completion,allow_stopped",
    [("complete", False), ("complete", True), ("stopped", True)],
)
def test_trace_completion_policy_preserves_raw_values_and_inner_first_axes(
    tmp_path: Path,
    trace_spec: AxesSpec[SyntheticTraceResult, ConfigBase],
    stopped_trace: Container,
    completion: str,
    allow_stopped: bool,
) -> None:
    path = tmp_path / "trace.h5"
    metadata = native_metadata().model_copy(update={"completion": completion})
    save(
        path,
        replace(
            stopped_trace,
            docs={**stopped_trace.docs, "metadata": metadata.model_dump(mode="json")},
        ),
    )
    try:
        trace = load_trace(tmp_path, path.name, trace_spec, allow_stopped=allow_stopped)
    except (TypeError, ValueError) as error:
        pytest.fail(
            f"Explicit allow_stopped={allow_stopped} must load {completion} raw data: {error}"
        )
    np.testing.assert_array_equal(
        trace.z, [[1 + 2j, 3 - 4j, 5j], [6 + 7j, complex(np.nan, np.nan), 9j]]
    )
    assert trace.z.shape == (2, 3)
    assert trace.data.unit == "V"
    assert [(axis.name, axis.unit) for axis in trace.axes] == [
        ("frequency", "Hz"),
        ("Flux", "V"),
    ]
    np.testing.assert_array_equal(trace.axes[0].values, [4e9, 5e9, 6e9])
    np.testing.assert_array_equal(trace.axes[1].values, [0.1, 0.2])


@pytest.mark.parametrize("completion", ["partial", "unknown"])
@pytest.mark.parametrize("allow_stopped", [False, True])
def test_trace_never_allows_partial_or_unknown(
    tmp_path: Path,
    trace_spec: AxesSpec[SyntheticTraceResult, ConfigBase],
    stopped_trace: Container,
    completion: str,
    allow_stopped: bool,
) -> None:
    path = tmp_path / "incomplete.h5"
    metadata = native_metadata().model_copy(update={"completion": completion})
    save(
        path,
        replace(
            stopped_trace,
            docs={**stopped_trace.docs, "metadata": metadata.model_dump(mode="json")},
        ),
    )
    with pytest.raises(ValueError, match="Expected complete") as error:
        load_trace(tmp_path, path.name, trace_spec, allow_stopped=allow_stopped)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("allow_stopped", [False, True])
@pytest.mark.parametrize(
    "invalid,diagnosis",
    [
        ("tag", "test/native acquisition"),
        ("unit", "signal unit"),
        ("missing", "missing variables"),
        ("empty", "Empty acquisition"),
    ],
)
def test_trace_completion_policy_keeps_validation_and_source_diagnostics(
    tmp_path: Path,
    trace_spec: AxesSpec[SyntheticTraceResult, ConfigBase],
    stopped_trace: Container,
    allow_stopped: bool,
    invalid: str,
    diagnosis: str,
) -> None:
    path = tmp_path / "invalid.h5"
    metadata = native_metadata().model_copy(
        update={"completion": "stopped" if allow_stopped else "complete"}
    )
    if invalid == "tag":
        metadata = metadata.model_copy(update={"experiment": "test/other"})
    variable = stopped_trace.vars["readout"]
    if invalid == "unit":
        variable = replace(variable, unit="A")
    elif invalid == "empty":
        variable = replace(
            variable,
            values=np.empty((0, 3), dtype=complex),
            axes=(ContainerAxis("Flux", "V", np.array([])), variable.axes[1]),
        )
    container = replace(
        stopped_trace,
        vars={} if invalid == "missing" else {"readout": variable},
        docs={**stopped_trace.docs, "metadata": metadata.model_dump(mode="json")},
    )
    save(path, container)
    with pytest.raises(ValueError, match=diagnosis) as error:
        load_trace(tmp_path, path.name, trace_spec, allow_stopped=allow_stopped)
    assert str(path) in str(error.value)


def inputs(
    tmp_path: Path,
    lifecycle: Literal["done", "cancelled", "failed"] = "done",
    run_id: str = "0123456789abcdef0123456789abcdef",
) -> tuple[RunManifest, PreparedRunData, RunSnapshot]:
    metadata = native_metadata()
    now = datetime(2026, 10, 12, tzinfo=UTC)
    manifest = TypeAdapter(RunManifest).validate_python(
        {
            "format": "zcu-workflow-run",
            "format_version": 1,
            "identity": RunIdentity(run_id, metadata.provenance),
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
    prepared = PreparedRunData(
        native_container(), native_cfg(), LabberMetadata(), "independent"
    )
    return manifest, prepared, metadata.snapshot


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
        assert labber.path.name == f"synthetic_abcdef_{sequence}.hdf5"
        assert native.path is not None
        stored = load(native.path)
        assert metadata_from_container(stored).run_id == manifest.identity.run_id
        assert (
            metadata_from_container(stored).started_at
            == manifest.started_at.isoformat()
        )
        assert metadata_from_container(stored).finished_at == (
            None if manifest.ended_at is None else manifest.ended_at.isoformat()
        )
        assert metadata_from_container(stored).completion == completion
        assert metadata_from_container(stored).snapshot == snapshot
        assert metadata_from_container(stored).experiment == "synthetic"
        assert (
            metadata_from_container(stored).provenance == manifest.identity.provenance
        )
        assert (
            TypeAdapter(CfgSnapshot).validate_python(stored.docs["cfg"]) == prepared.cfg
        )
        restored = load_grouped_labber_data(str(labber.path))
        for name, variable in prepared.container.vars.items():
            np.testing.assert_array_equal(stored.vars[name].values, variable.values)
            np.testing.assert_array_equal(
                restored.variables[DataVariable(name)].z, variable.values
            )
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
    assert native.error and "run_id" in native.error
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
