"""Migration-only Labber reading through the public offline contract."""

from pathlib import Path

import h5py as h5
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
    write_labber,
)

from migrations.legacy_labber import load_legacy_labber_payload


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


@pytest.fixture
def unmarked_logs(tmp_path: Path) -> tuple[Path, tuple[VariableSchema, ...]]:
    source = tmp_path / "old.hdf5"
    schemas: list[VariableSchema] = []
    for index, (name, axis, values, signal) in enumerate([
        ("first", "time", [0.0, 1.0], [1.0, 2.0]),
        ("second", "frequency", [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]),
    ]):
        path = source if index == 0 else tmp_path / "additional.hdf5"
        payload = ExperimentPayload(
            variables={DataVariable(name): LabberPayload(
                (name, "V", np.asarray(signal)), [(axis, "s", np.asarray(values))],
                timestamps=np.array([1759276800.0 + index]),
            )}, metadata=LabberMetadata(tags=["offline"], creation_time=1759276800.0),
            representation="single",
        )
        write_labber(path, payload, cfg=CfgSnapshot(values={}, cfg_type="Synthetic", schema_version="1.0"))
        if index:
            with h5.File(source, "r+") as root, h5.File(path, "r") as extra:
                group = root.create_group("Log_2")
                for key in ("Data", "Channels", "Log list"):
                    extra.copy(key, group)
        schemas.append(VariableSchema(
            variable=DataVariable(name),
            axes=(AxisSchema(name=axis, unit="s", dtype=np.dtype("float64")),),
            signal_name=name, signal_unit="V", signal_dtype=np.dtype("float64"),
        ))
    return source, tuple(schemas)


def test_unmarked_logs_follow_explicit_schema_order_with_independent_grids(
    unmarked_logs: tuple[Path, tuple[VariableSchema, ...]],
) -> None:
    source, schema = unmarked_logs
    original = source.read_bytes()
    try:
        loaded = load_legacy_labber_payload(source, schema=schema)
    except ValueError as error:
        pytest.fail(f"Explicit ordered logs must be convertible: {error}")
    assert loaded.representation == "grouped"
    assert loaded.metadata.tags == ["offline"]
    np.testing.assert_array_equal(loaded.variables[DataVariable("first")].z, [1.0, 2.0])
    np.testing.assert_array_equal(loaded.variables[DataVariable("second")].z, [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(loaded.variables[DataVariable("second")].x, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(loaded.variables[DataVariable("second")].timestamps, [1759276801.0])
    assert source.read_bytes() == original


@pytest.mark.parametrize("mismatch", ["count", "signal", "axis", "unit", "shape"])
def test_unmarked_log_schema_mismatch_is_located(
    unmarked_logs: tuple[Path, tuple[VariableSchema, ...]], mismatch: str
) -> None:
    source, schema = unmarked_logs
    if mismatch == "count":
        schema = schema[:1]
    else:
        with h5.File(source, "r+") as file:
            if mismatch == "shape":
                file["Log_2/Data"].attrs["Step dimensions"] = np.array([99])
            elif mismatch == "signal":
                rows = file["Log_2/Log list"][()]
                rows[0]["channel_name"] = "wrong"
                file["Log_2/Log list"][...] = rows
            else:
                rows = file["Log_2/Channels"][()]
                rows[0]["name" if mismatch == "axis" else "unitPhys"] = "wrong"
                file["Log_2/Channels"][...] = rows
    with pytest.raises(ValueError, match="log count|/second|Log_2"):
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
