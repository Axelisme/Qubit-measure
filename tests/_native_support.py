"""Synthetic historical inputs shared by native datafile and experiment tests."""

from uuid import UUID

import h5py as h5
import numpy as np
from pydantic import TypeAdapter
from zcu_tools.datafile import Axis, Container, ContainerAttrs, Variable
from zcu_tools.experiment.metadata import (
    CfgSnapshot,
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
        snapshot=RunSnapshot(
            entry_id=UUID("12345678-1234-5678-1234-567812345678"),
            entry_name="test_entry",
            point="sweet",
            description="Historical snapshot",
            roles={"qubit": "Q1"},
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


def native_container(*, metadata: RunMetadata | None = None) -> Container:
    """Build one synthetic SI readout and its historical cfg/metadata documents."""
    metadata = native_metadata() if metadata is None else metadata
    return Container(
        vars={
            "readout": Variable(
                np.array([1 + 2j, 3 - 4j]),
                (Axis("frequency", "Hz", np.array([4e9, 5e9])),),
                "V",
            )
        },
        docs={
            "cfg": TypeAdapter(CfgSnapshot).dump_python(native_cfg(), mode="json"),
            "metadata": metadata.model_dump(mode="json"),
        },
        groups={},
        attrs=ContainerAttrs("3.0", "synthetic-tests", "experiment-run", {}),
    )
