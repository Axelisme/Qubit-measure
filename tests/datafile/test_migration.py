"""Migration-only Labber reading through the public datafile contract."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import (
    AxisSchema,
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    VariableSchema,
    load_legacy_labber_payload,
    write_labber,
)


@pytest.mark.parametrize("representation", ["single", "grouped"])
def test_explicit_variable_identity_and_reader_metadata_are_preserved(
    tmp_path: Path, representation: str
) -> None:
    source = tmp_path / "unrelated_filename.hdf5"
    payload = ExperimentPayload(
        variables={
            DataVariable("measured"): LabberPayload(
                ("voltage", "V", np.array([2.0, 3.0])),
                [("time", "s", np.array([0.0, 1.0]))],
            )
        },
        metadata=LabberMetadata(tags=["synthetic"], project="test_project"),
        representation="single" if representation == "single" else "grouped",
    )
    write_labber(
        source,
        payload,
        cfg=CfgSnapshot(values={}, cfg_type="Synthetic", schema_version="1.0"),
    )
    before = source.read_bytes()
    schema = (
        VariableSchema(
            variable=DataVariable("measured"),
            axes=(AxisSchema(name="time", unit="s", dtype=np.dtype("float64")),),
            signal_name="voltage",
            signal_unit="V",
            signal_dtype=np.dtype("float64"),
        ),
    )
    loaded = load_legacy_labber_payload(source, schema=schema)
    assert loaded.representation == representation
    assert loaded.metadata.tags == ["synthetic"]
    assert loaded.metadata.project == "test_project"
    assert tuple(loaded.variables) == (DataVariable("measured"),)
    values = loaded.variables[DataVariable("measured")].z
    np.testing.assert_array_equal(values, [2.0, 3.0])
    assert np.asarray(values).dtype == np.dtype("float64")
    assert source.read_bytes() == before


def test_unmarked_file_does_not_guess_multiple_variable_identities(
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.hdf5"
    payload = ExperimentPayload(
        variables={
            DataVariable("one"): LabberPayload(
                ("voltage", "V", np.array([1.0])), [("time", "s", np.array([0.0]))]
            )
        },
        metadata=LabberMetadata(),
        representation="single",
    )
    write_labber(
        source,
        payload,
        cfg=CfgSnapshot(values={}, cfg_type="Synthetic", schema_version="1.0"),
    )
    schema = tuple(
        VariableSchema(
            variable=DataVariable(name),
            axes=(AxisSchema(name="time", unit="s", dtype=np.dtype("float64")),),
            signal_name="voltage",
            signal_unit="V",
            signal_dtype=np.dtype("float64"),
        )
        for name in ("one", "two")
    )
    with pytest.raises(ValueError, match="cannot identify multiple"):
        load_legacy_labber_payload(source, schema=schema)


def test_declared_units_are_not_guessed_or_converted(tmp_path: Path) -> None:
    source = tmp_path / "signal.hdf5"
    payload = ExperimentPayload(
        variables={
            DataVariable("one"): LabberPayload(
                ("voltage", "mV", np.array([1.0])), [("time", "s", np.array([0.0]))]
            )
        },
        metadata=LabberMetadata(),
        representation="single",
    )
    write_labber(
        source,
        payload,
        cfg=CfgSnapshot(values={}, cfg_type="Synthetic", schema_version="1.0"),
    )
    schema = (
        VariableSchema(
            variable=DataVariable("one"),
            axes=(AxisSchema(name="time", unit="s", dtype=np.dtype("float64")),),
            signal_name="voltage",
            signal_unit="V",
            signal_dtype=np.dtype("float64"),
        ),
    )
    with pytest.raises(ValueError, match="units"):
        load_legacy_labber_payload(source, schema=schema)
