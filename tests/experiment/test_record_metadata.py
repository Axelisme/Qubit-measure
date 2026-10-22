"""Run-owned metadata across the records and native Container seams."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import zcu_tools.experiment as experiment
from zcu_tools.datafile import LabberMetadata, load, save


def historical_metadata() -> experiment.RunMetadata:
    """Return fixed historical facts, including optional Labber input metadata."""
    return experiment.RunMetadata(
        run_id="20261006T010000Z-a1b2c3",
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


def declaration() -> experiment.AxesSpec[Signal, experiment.ExpCfgModel]:
    """Map a scaled frequency coordinate and complex voltage signal."""
    return experiment.AxesSpec(
        axes=(experiment.Axis("frequency", "Frequency", "Hz", scale=1e6),),
        z=experiment.ZSpec("values", "Signal", "V", scale=1e-3),
        result_type=Signal,
        cfg_type=experiment.ExpCfgModel,
        tag="test/metadata",
    )


def test_record_detaches_cfg_and_metadata_but_shares_result() -> None:
    metadata = historical_metadata()
    cfg = experiment.ExpCfgModel()
    result = Signal(np.array([4000.0, 5000.0]), np.array([1j, 2j]))
    record = experiment.RunRecord(cfg, result, metadata=metadata)
    assert record.metadata == metadata
    assert record.metadata is not metadata
    assert record.cfg is not cfg
    assert record.result is result
    assert record.metadata.snapshot.roles == {"qubit": "Q1"}
    assert record.metadata.snapshot is not metadata.snapshot
    assert record.metadata.labber is not None
    assert metadata.labber is not None
    metadata.labber.tags.append("later")
    assert record.metadata.labber.tags == ["offline"]


def test_native_round_trip_keeps_record_metadata_and_container_docs(
    tmp_path: Path,
) -> None:
    metadata = historical_metadata()
    record = experiment.RunRecord(
        experiment.ExpCfgModel(),
        Signal(np.array([4000.0, 5000.0]), np.array([1j, 2j])),
        metadata=metadata,
    )
    path = tmp_path / "run.h5"
    experiment.save_run(record, path, spec=declaration())
    container = load(path)
    assert container.docs["cfg"] == {
        "values": {},
        "cfg_type": "ExpCfgModel",
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
        experiment.ExpCfgModel(),
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
        experiment.ExpCfgModel(),
        Signal(np.array([4000.0]), np.array([1j])),
        metadata=metadata,
    )
    path = tmp_path / "invalid-metadata.h5"
    experiment.save_run(record, path, spec=declaration())
    container = load(path)
    docs = dict(container.docs)
    docs["metadata"][defect] = "invalid"
    save(path, replace(container, docs=docs), replace=True)
    with pytest.raises(ValueError, match="metadata") as caught:
        experiment.load_run_with_metadata(path, spec=declaration())
    assert str(path) in str(caught.value)


def test_native_save_validates_unchecked_metadata_updates(tmp_path: Path) -> None:
    metadata = historical_metadata().model_copy(update={"completion": "invalid"})
    record = experiment.RunRecord(
        experiment.ExpCfgModel(),
        Signal(np.array([4000.0]), np.array([1j])),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="metadata|completion"):
        experiment.save_run(record, tmp_path / "invalid.h5", spec=declaration())
    assert list(tmp_path.iterdir()) == []
