"""Shared payload export at the public datafile seam.

Parity cases pin the existing Labber representation while old saver entrypoints
remain supported; remove the comparison baseline if those entrypoints retire.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.datafile import (
    Axis,
    CfgSnapshot,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    decode_labber_comment,
    load_grouped_labber_data,
    load_labber_data,
    load_run_data,
    save_grouped_labber_data,
    save_labber_data,
    save_run_data,
    write_labber,
)

from tests._native_support import native_metadata


def _cfg() -> CfgSnapshot:
    return CfgSnapshot(
        values={"frequency": 5.0e9, "nested": {"value": [None, True, 2.0]}},
        cfg_type="SyntheticCfg",
        schema_version="1.0",
    )


def _signal(complex_signal: bool = True, *, label: str = "signal") -> LabberPayload:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    if complex_signal:
        values = values + 1j * (values + 0.5)
    return LabberPayload(
        (label, "V", values),
        axes=[
            ("frequency", "Hz", np.array([4.0e9, 4.5e9, 5.0e9])),
            ("power", "dBm", np.array([-20.0, -10.0])),
        ],
        timestamps=np.array([1760000000.0, 1760000002.0]),
    )


def _equal_channel(actual: Axis, expected: Axis) -> None:
    assert actual.name == expected.name
    assert actual.unit == expected.unit
    np.testing.assert_array_equal(actual.values, expected.values)
    assert np.asarray(actual.values).dtype == np.asarray(expected.values).dtype


@pytest.mark.parametrize("complex_signal", [False, True])
def test_single_export_matches_existing_reader_at_exact_path(
    tmp_path: Path, complex_signal: bool
) -> None:
    signal = _signal(complex_signal)
    metadata = LabberMetadata(tags=["scan", "test"], project="project", user="user")
    payload = ExperimentPayload(
        variables={DataVariable("readout"): signal},
        metadata=metadata,
        representation="single",
    )
    baseline = save_labber_data(
        str(tmp_path / "baseline.hdf5"),
        z=signal.data,
        axes=signal.axes,
        tags=metadata.tags,
        project=metadata.project,
        user=metadata.user,
        timestamps=signal.timestamps,
    )
    exact = tmp_path / "opaque.export"
    write_labber(exact, payload, cfg=_cfg(), comment="synthetic")
    actual = load_labber_data(str(exact))
    expected = load_labber_data(baseline)
    _equal_channel(actual.data, expected.data)
    for actual_axis, expected_axis in zip(actual.axes, expected.axes, strict=True):
        _equal_channel(actual_axis, expected_axis)
    np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
    assert (actual.tags, actual.project, actual.user) == (
        expected.tags,
        expected.project,
        expected.user,
    )


def test_export_stores_complete_cfg_and_caller_comment(tmp_path: Path) -> None:
    cfg = _cfg()
    payload = ExperimentPayload(
        variables={DataVariable("readout"): _signal()},
        metadata=LabberMetadata(comment="old metadata text"),
        representation="single",
    )
    destination = tmp_path / "cfg.hdf5"
    write_labber(destination, payload, cfg=cfg, comment="caller text")
    raw = json.loads(load_labber_data(str(destination)).comment)
    assert raw["cfg"] == cfg.values
    assert raw["comment"] == "caller text"
    assert len(raw["timestamp"]) == 19
    decoded = decode_labber_comment(load_labber_data(str(destination)).comment)
    assert decoded.cfg == cfg.values
    assert decoded.comment == "caller text"
    assert decoded.timestamp == raw["timestamp"]


@pytest.mark.parametrize("variable_count", [1, 2])
def test_grouped_export_keeps_declared_identities_and_reader_parity(
    tmp_path: Path, variable_count: int
) -> None:
    variables = {
        DataVariable(f"var_{index}"): _signal(
            complex_signal=bool(index), label=f"signal_{index}"
        )
        for index in range(variable_count)
    }
    metadata = LabberMetadata(
        comment="grouped text",
        tags=["grouped", "test"],
        project="project",
        user="user",
        creation_time=1760000000.0,
    )
    payload = ExperimentPayload(
        variables=variables, metadata=metadata, representation="grouped"
    )
    baseline = save_grouped_labber_data(
        str(tmp_path / "baseline.hdf5"),
        {str(variable): signal for variable, signal in variables.items()},
        metadata=metadata,
    )
    destination = tmp_path / "grouped.export"
    write_labber(destination, payload, cfg=_cfg())
    actual = load_grouped_labber_data(
        str(destination), required_variables=tuple(variables)
    )
    expected = load_grouped_labber_data(baseline, required_variables=tuple(variables))
    assert tuple(actual.variables) == tuple(expected.variables) == tuple(variables)
    for variable in variables:
        loaded = actual.variables[variable]
        original = expected.variables[variable]
        _equal_channel(loaded.data, original.data)
        for loaded_axis, original_axis in zip(loaded.axes, original.axes, strict=True):
            _equal_channel(loaded_axis, original_axis)
        np.testing.assert_array_equal(loaded.timestamps, original.timestamps)
    assert (
        actual.metadata.tags,
        actual.metadata.project,
        actual.metadata.user,
        actual.metadata.creation_time,
    ) == (
        expected.metadata.tags,
        expected.metadata.project,
        expected.metadata.user,
        expected.metadata.creation_time,
    )
    decoded = decode_labber_comment(actual.metadata.comment)
    assert decoded.cfg == _cfg().values
    assert decoded.comment == "grouped text"


@pytest.mark.parametrize(
    "text", ["operator note", "2024", "[1,2]", '"quoted"', '{"operator":"note"}', ""]
)
def test_comment_decoder_preserves_non_envelope_text(text: str) -> None:
    decoded = decode_labber_comment(text)
    assert decoded.cfg is None
    assert decoded.timestamp is None
    assert decoded.comment == text


@pytest.mark.parametrize(
    "text", ['{"cfg":[]}', '{"cfg":5}', '{"comment":false}', '{"timestamp":42}']
)
def test_comment_decoder_rejects_malformed_envelope_fields(text: str) -> None:
    with pytest.raises(ValueError, match="cfg|comment|timestamp"):
        decode_labber_comment(text)


@pytest.mark.parametrize("representation", ["single", "grouped"])
def test_export_refuses_existing_exact_path_without_touching_it(
    tmp_path: Path, representation: Literal["single", "grouped"]
) -> None:
    signal = _signal()
    payload = ExperimentPayload(
        variables={DataVariable("readout"): signal},
        metadata=LabberMetadata(),
        representation=representation,
    )
    destination = tmp_path / "existing.export"
    destination.write_bytes(b"original destination")
    with pytest.raises(FileExistsError):
        write_labber(destination, payload, cfg=_cfg())
    assert destination.read_bytes() == b"original destination"


def test_incompatible_labber_grid_does_not_block_native_writer(tmp_path: Path) -> None:
    first = _signal(label="first")
    second = LabberPayload(
        ("second", "V", np.array([1.0, 2.0])),
        axes=[("delay", "s", np.array([1.0e-6, 2.0e-6]))],
        timestamps=np.array([1760000000.0]),
    )
    payload = ExperimentPayload(
        variables={DataVariable("first"): first, DataVariable("second"): second},
        metadata=LabberMetadata(),
        representation="grouped",
    )
    with pytest.raises(ValueError, match="common"):
        write_labber(tmp_path / "labber.export", payload, cfg=_cfg())
    destination = tmp_path / "native.h5"
    save_run_data(destination, payload, native_metadata(), cfg=_cfg())
    loaded = load_run_data(destination)
    assert tuple(loaded.payload.variables) == tuple(payload.variables)
    for variable in payload.variables:
        np.testing.assert_array_equal(
            loaded.payload.variables[variable].z, payload.variables[variable].z
        )


def test_native_metadata_failure_does_not_block_labber_writer(tmp_path: Path) -> None:
    payload = ExperimentPayload(
        variables={DataVariable("readout"): _signal()},
        metadata=LabberMetadata(),
        representation="single",
    )
    metadata = replace(native_metadata(), run_id="invalid")
    with pytest.raises(ValueError, match="run_id"):
        save_run_data(tmp_path / "native.h5", payload, metadata, cfg=_cfg())
    destination = tmp_path / "labber.export"
    write_labber(destination, payload, cfg=_cfg())
    np.testing.assert_array_equal(
        load_labber_data(str(destination)).z,
        payload.variables[DataVariable("readout")].z,
    )


def test_single_scalar_is_rejected_before_creating_destination(tmp_path: Path) -> None:
    destination = tmp_path / "scalar.export"
    payload = ExperimentPayload(
        variables={
            DataVariable("readout"): LabberPayload(
                ("signal", "V", np.array(1.0)), axes=[]
            )
        },
        metadata=LabberMetadata(),
        representation="single",
    )
    with pytest.raises(ValueError, match="step axis") as error:
        write_labber(destination, payload, cfg=_cfg())
    assert str(destination) in str(error.value)
    assert "readout" in str(error.value)
    assert "axis" in str(error.value)
    assert not destination.exists()


def test_export_rejects_complex_coordinates_instead_of_discarding_them(
    tmp_path: Path,
) -> None:
    payload = ExperimentPayload(
        variables={
            DataVariable("readout"): LabberPayload(
                ("signal", "V", np.array([1.0, 2.0])),
                axes=[("frequency", "Hz", np.array([4.0e9 + 1j, 5.0e9 + 2j]))],
            )
        },
        metadata=LabberMetadata(),
        representation="single",
    )
    with pytest.raises(ValueError, match="imaginary"):
        write_labber(tmp_path / "invalid.export", payload, cfg=_cfg())
