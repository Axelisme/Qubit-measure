"""Run-owned metadata across the records and native Container seams."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import numpy as np
import pytest
import zcu_tools.experiment as experiment
from pydantic import TypeAdapter
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable, LabberMetadata, load, save


def historical_metadata(
    run_id: str = "20261006T010000Z-a1b2c3",
) -> experiment.RunMetadata:
    """Return historical facts for run_id, including Labber input metadata."""
    return experiment.RunMetadata(
        run_id=run_id,
        experiment="test/metadata",
        started_at=None,
        finished_at="2026-10-06T01:00:02Z",
        completion="unknown",
        snapshot=experiment.RunSnapshot(
            entry_id=UUID("12345678-1234-5678-1234-567812345678"),
            entry_name="historical entry",
            point="sweet",
            description=None,
            roles={"qubit": "Q1"},
        ),
        provenance=experiment.SoftwareProvenance(
            software_versions={"zcu-tools": "0.1.0"},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
        device_snapshot={"bias": {"value": 0.25, "unit": "A"}},
        labber=LabberMetadata(
            comment="handwritten note",
            tags=["offline"],
            project="project",
            user="operator",
            creation_time=1759276800.0,
        ),
    )


@dataclass
class Signal:
    frequency: np.ndarray
    values: np.ndarray


def declaration() -> experiment.AxesSpec[Signal, ConfigBase]:
    """Map a scaled frequency coordinate and complex voltage signal."""
    return experiment.AxesSpec(
        axes=(experiment.Axis("frequency", "Frequency", "Hz", scale=1e6),),
        z=experiment.ZSpec("values", "Signal", "V", scale=1e-3),
        result_type=Signal,
        cfg_type=ConfigBase,
        tag="test/metadata",
        data_variable=DataVariable("signal"),
    )


@pytest.mark.parametrize("entry", ["model", "validator"])
@pytest.mark.parametrize(
    "run_id",
    [
        "20261022T120000Z-ab12cd",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-3",
        "00000000000000000000000000000000-000000-1",
        "ffffffffffffffffffffffffffffffff-999999-1234567890",
    ],
)
def test_run_identity_accepts_exact_experiment_or_workflow_id(
    entry: str, run_id: str
) -> None:
    if entry == "model":
        metadata = historical_metadata(run_id)
        assert metadata.run_id == run_id
        assert metadata.model_dump(mode="json")["run_id"] == run_id
    else:
        assert experiment.validate_run_id(run_id) is None


@pytest.mark.parametrize("entry", ["model", "validator"])
@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "arbitrary-token",
        "20261022T120000Z-ab12c",
        "20261022T120000Z-ab12cde",
        "20261022T120000Z-AB12CD",
        "20261022T120000Z-ab12cd-000001-3",
        "21ed5156dc9b441a8b5cda3cf26633f7",
        "21ed5156dc9b441a8b5cda3cf26633f7-1-3",
        "21ed5156dc9b441a8b5cda3cf26633f7-0000001-3",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-0",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-03",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001--1",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-+3",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-3.0",
        "21ED5156DC9B441A8B5CDA3CF26633F7-000001-3",
        "g1ed5156dc9b441a8b5cda3cf26633f7-000001-3",
        "21ed5156dc9b441a8b5cda3cf26633f-000001-3",
        "021ed5156dc9b441a8b5cda3cf26633f7-000001-3",
        "21ed5156dc9b441a8b5cda3cf26633f7-０００００１-3",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-３",
        " 20261022T120000Z-ab12cd",
        "20261022T120000Z-ab12cd\n",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-3\n",
        "prefix-21ed5156dc9b441a8b5cda3cf26633f7-000001-3",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-3-extra",
    ],
)
def test_run_identity_rejects_other_strings(entry: str, run_id: str) -> None:
    validate = historical_metadata if entry == "model" else experiment.validate_run_id
    with pytest.raises(ValueError, match="run_id"):
        validate(run_id)


def test_readonly_snapshot_roles_serialize_without_warnings() -> None:
    snapshot = experiment.RunSnapshot(
        entry_id=UUID("12345678-1234-5678-1234-567812345678"),
        entry_name="historical entry",
        point="sweet",
        description=None,
        roles=MappingProxyType({"qubit": "Q1"}),
    )
    adapter = TypeAdapter(experiment.RunSnapshot)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        python_doc = adapter.dump_python(snapshot, mode="python")
        json_doc = adapter.dump_python(snapshot, mode="json")
        json_bytes = adapter.dump_json(snapshot)

    assert not caught, [str(warning.message) for warning in caught]
    expected_json = {
        "entry_id": "12345678-1234-5678-1234-567812345678",
        "entry_name": "historical entry",
        "point": "sweet",
        "description": None,
        "roles": {"qubit": "Q1"},
    }
    assert python_doc == {**expected_json, "entry_id": snapshot.entry_id}
    assert json_doc == expected_json
    assert json.loads(json_bytes) == expected_json
    python_doc["roles"]["qubit"] = "renamed in Python output"
    json_doc["roles"]["qubit"] = "renamed in JSON output"
    assert snapshot.roles == {"qubit": "Q1"}
    assert json.loads(adapter.dump_json(snapshot)) == expected_json


def test_record_detaches_cfg_and_metadata_but_shares_result() -> None:
    metadata = historical_metadata()
    cfg = ConfigBase()
    result = Signal(np.array([4000.0, 5000.0]), np.array([1j, 2j]))
    record = experiment.RunRecord(cfg, result, metadata=metadata)
    assert record.metadata == metadata
    assert record.metadata is not None
    assert record.metadata is not metadata
    assert record.cfg is not cfg
    assert record.result is result
    assert record.metadata.snapshot.roles == {"qubit": "Q1"}
    assert record.metadata.snapshot is not metadata.snapshot
    assert record.metadata.labber is not None
    assert metadata.labber is not None
    metadata.labber.tags.append("later")
    assert record.metadata.labber.tags == ["offline"]


@pytest.mark.parametrize(
    "run_id",
    [
        "20261022T120000Z-ab12cd",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-3",
    ],
)
def test_native_round_trip_keeps_record_metadata_and_container_docs(
    tmp_path: Path, run_id: str
) -> None:
    metadata = historical_metadata(run_id)
    record = experiment.RunRecord(
        ConfigBase(),
        Signal(np.array([4000.0, 5000.0]), np.array([1j, 2j])),
        metadata=metadata,
    )
    path = tmp_path / "run.h5"
    experiment.save_run(record, path, spec=declaration())
    container = load(path)
    assert container.docs["cfg"] == {
        "values": {},
        "cfg_type": "ConfigBase",
        "schema_version": "1.0",
    }
    assert container.docs["metadata"] == metadata.model_dump(mode="json")
    np.testing.assert_array_equal(container.vars["signal"].axes[0].values, [4e9, 5e9])
    np.testing.assert_allclose(container.vars["signal"].values, [0.001j, 0.002j])
    restored = experiment.load_run_with_metadata(path, spec=declaration())
    assert restored.metadata == metadata
    assert restored.cfg == record.cfg
    np.testing.assert_allclose(restored.result.values, record.result.values)
    assert (
        experiment.restore_run(container, spec=declaration()).record.metadata
        == metadata
    )


def test_native_save_requires_record_metadata(tmp_path: Path) -> None:
    record = experiment.RunRecord(
        ConfigBase(),
        Signal(np.array([4000.0]), np.array([1j])),
    )
    assert record.metadata is None
    with pytest.raises(ValueError, match="metadata"):
        experiment.save_run(record, tmp_path / "missing.h5", spec=declaration())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("defect", ["completion", "snapshot", "run_id", "started_at"])
def test_native_load_rejects_invalid_metadata_with_source_path(
    tmp_path: Path, defect: str
) -> None:
    metadata = historical_metadata()
    record = experiment.RunRecord(
        ConfigBase(),
        Signal(np.array([4000.0]), np.array([1j])),
        metadata=metadata,
    )
    path = tmp_path / "invalid-metadata.h5"
    experiment.save_run(record, path, spec=declaration())
    container = load(path)
    docs = dict(container.docs)
    docs["metadata"][defect] = (
        "20261022T120000Z-ab12cd-000001-3" if defect == "run_id" else "invalid"
    )
    save(path, replace(container, docs=docs), replace=True)
    with pytest.raises(ValueError, match="metadata") as caught:
        experiment.load_run_with_metadata(path, spec=declaration())
    assert str(path) in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completion", "invalid"),
        ("run_id", "20261022T120000Z-ab12cd-000001-3"),
    ],
)
def test_native_save_validates_unchecked_metadata_updates(
    tmp_path: Path, field: str, value: str
) -> None:
    metadata = historical_metadata().model_copy(update={field: value})
    record = experiment.RunRecord(
        ConfigBase(),
        Signal(np.array([4000.0]), np.array([1j])),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="metadata|completion"):
        experiment.save_run(record, tmp_path / "invalid.h5", spec=declaration())
    assert list(tmp_path.iterdir()) == []
