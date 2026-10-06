"""Synthetic historical inputs shared by native datafile and experiment tests."""

from uuid import UUID

import h5py as h5
import numpy as np
from zcu_tools.datafile import (
    CfgSnapshot,
    CloneOrigin,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    ParameterSnapshot,
    ParameterSource,
    RunMetadata,
    RunSnapshot,
    SoftwareProvenance,
)


def hdf_dataset(file: h5.File, path: str) -> h5.Dataset:
    """Return the fixture's dataset, asserting its public HDF5 node type."""
    node = file[path]
    assert isinstance(node, h5.Dataset)
    return node


def hdf_group(file: h5.File, path: str) -> h5.Group:
    """Return the fixture's group, asserting its public HDF5 node type."""
    node = file[path]
    assert isinstance(node, h5.Group)
    return node


def hdf_json_text(file: h5.File, path: str) -> str:
    """Read a fixture's scalar UTF-8 JSON dataset without a type suppression."""
    value = hdf_dataset(file, path).asstr()[()]
    assert isinstance(value, str)
    return value


def native_metadata(tag: str = "test/native") -> RunMetadata:
    """Return historical metadata with every snapshot/provenance field populated."""
    return RunMetadata(
        run_id="20261006T010000Z-a1b2c3",
        experiment=tag,
        started_at="2026-10-06T01:00:00Z",
        finished_at="2026-10-06T01:00:02Z",
        completion="complete",
        labber_path="exports/previous.h5",
        snapshot=RunSnapshot(
            entry_id=UUID("12345678-1234-5678-1234-567812345678"),
            entry_name="test_entry",
            point="sweet",
            description="Historical snapshot",
            roles={"qubit": "Q1"},
            params={
                "Q1.freq": ParameterSnapshot(
                    value=5000.0,
                    unit="MHz",
                    source=ParameterSource(
                        source="event-1",
                        kind="calibration",
                        run_id="20261006T000000Z-d4e5f6",
                        at="2026-10-06T00:00:02Z",
                        stderr=0.1,
                        cloned_from=CloneOrigin(
                            entry_id=UUID("87654321-4321-8765-4321-876543218765"),
                            point="origin",
                        ),
                    ),
                )
            },
        ),
        provenance=SoftwareProvenance(
            software_versions={"zcu-tools": "0.1.0"},
            git_commit="a" * 40,
            git_dirty=False,
            qick_version="0.2.418",
            soc_fingerprint="synthetic-soc",
            hostname="synthetic-host",
        ),
    )


def native_cfg() -> CfgSnapshot:
    """Return a complete JSON cfg with nested forward-minor fields."""
    return CfgSnapshot(
        values={"frequency": 5000.0, "nested": {"unknown": [None, True, 1.5]}},
        cfg_type="SyntheticCfg",
        schema_version="1.3",
    )


def native_payload() -> ExperimentPayload:
    """Return a single SI complex signal and one absolute timestamp."""
    return ExperimentPayload(
        variables={
            DataVariable("readout"): LabberPayload(
                ("signal", "V", np.array([1.0 + 2.0j, 3.0 - 4.0j])),
                axes=[("frequency", "Hz", np.array([4.0e9, 5.0e9]))],
                timestamps=np.array([1760000000.0]),
            )
        },
        metadata=LabberMetadata(tags="test/native"),
        representation="single",
    )
