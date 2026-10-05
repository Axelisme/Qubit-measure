"""Native experiment-data contract tests through the public datafile facade."""

import json
import os
from dataclasses import replace
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
import xarray as xr
from zcu_tools.datafile import (
    AxisSchema,
    DataVariable,
    ExperimentPayload,
    LabberMetadata,
    LabberPayload,
    SoftwareProvenance,
    VariableSchema,
    load_run_data,
    save_labber_data,
    save_run_data,
    validate_experiment_payload,
)

from tests._native_support import (
    hdf_dataset,
    hdf_json_text,
    native_cfg,
    native_metadata,
    native_payload,
)


def test_single_run_round_trip_preserves_si_arrays_and_historical_inputs(
    tmp_path: Path,
) -> None:
    payload, metadata, cfg = native_payload(), native_metadata(), native_cfg()
    destination = tmp_path / "opaque.h5"
    save_run_data(destination, payload, metadata, cfg=cfg)
    stored = load_run_data(destination)

    assert stored.metadata == metadata
    assert stored.cfg == cfg
    assert stored.payload.metadata == payload.metadata
    assert stored.payload.representation == "single"
    assert tuple(stored.payload.variables) == (DataVariable("readout"),)
    assert stored.extensions is None
    recovered = stored.payload.variables[DataVariable("readout")]
    np.testing.assert_array_equal(
        recovered.z, payload.variables[DataVariable("readout")].z
    )
    np.testing.assert_array_equal(recovered.axes[0].values, np.array([4.0e9, 5.0e9]))
    np.testing.assert_array_equal(recovered.timestamps, np.array([1760000000.0]))

    with h5.File(destination, "r") as file:
        assert file.attrs["format"] == "zcu.experiment-data"
        assert file.attrs["format_version"] == "1.0"
        signal = hdf_dataset(file, "data/readout/signal")
        assert signal.attrs["units"] == "V"
        assert signal.id.get_type().get_member_name(0) == b"r"
        assert signal.id.get_type().get_member_name(1) == b"i"
        assert file["data/readout/frequency"].attrs["units"] == "Hz"
        assert file["data/readout/timestamps"].attrs["units"] == "s"
        assert signal.dims[0][0].name == "/data/readout/frequency"
        assert json.loads(hdf_json_text(file, "cfg")) == cfg.values
        assert json.loads(hdf_json_text(file, "context"))["entry_id"] == str(
            metadata.snapshot.entry_id
        )
        assert file["provenance"].attrs["git_dirty"] == "false"
        assert file["provenance"].attrs["hostname"] == "synthetic-host"

    with xr.open_dataset(
        destination, engine="h5netcdf", group="data/readout"
    ) as dataset:
        np.testing.assert_array_equal(dataset["signal"].values, recovered.z)
        np.testing.assert_array_equal(
            dataset.coords["frequency"].values, recovered.axes[0].values
        )
        assert dataset["signal"].attrs["units"] == "V"


@pytest.mark.parametrize("count", [1, 2])
def test_grouped_preserves_representation_and_heterogeneous_grids(
    tmp_path: Path, count: int
) -> None:
    variables = dict(native_payload().variables)
    if count == 2:
        variables[DataVariable("decay")] = LabberPayload(
            ("amplitude", "", np.arange(6, dtype=np.float32).reshape(2, 3)),
            axes=[
                ("delay", "s", np.array([1e-6, 2e-6, 3e-6])),
                ("state", "", np.array([0, 1])),
            ],
            timestamps=np.array([1760000001.0, np.nan]),
        )
    payload = ExperimentPayload(
        variables=variables, metadata=LabberMetadata(), representation="grouped"
    )
    path = tmp_path / "grouped.h5"
    save_run_data(path, payload, native_metadata(), cfg=native_cfg())
    loaded = load_run_data(path)
    assert loaded.payload.representation == "grouped"
    assert tuple(loaded.payload.variables) == tuple(variables)
    for name, expected in variables.items():
        actual = loaded.payload.variables[name]
        np.testing.assert_array_equal(actual.z, expected.z)
        assert actual.z.dtype == expected.z.dtype
        np.testing.assert_array_equal(actual.timestamps, expected.timestamps)
        for axis, expected_axis in zip(actual.axes, expected.axes, strict=True):
            assert (axis.name, axis.unit) == (expected_axis.name, expected_axis.unit)
            np.testing.assert_array_equal(axis.values, expected_axis.values)


@pytest.mark.parametrize("timestamps", [None, np.array([np.nan], dtype=np.float32)])
def test_optional_timestamps_preserve_absence_dtype_and_nan(
    tmp_path: Path, timestamps: np.ndarray | None
) -> None:
    payload = native_payload()
    variable = payload.variables[DataVariable("readout")]
    variable.timestamps = timestamps
    path = tmp_path / "timestamps.h5"
    save_run_data(path, payload, native_metadata(), cfg=native_cfg())
    actual = load_run_data(path).payload.variables[DataVariable("readout")].timestamps
    if timestamps is None:
        assert actual is None
    else:
        assert actual.dtype == timestamps.dtype
        np.testing.assert_array_equal(actual, timestamps)


@pytest.mark.parametrize(
    "dirty,encoded", [(None, "unknown"), (True, "true"), (False, "false")]
)
def test_unknown_evidence_has_explicit_null_encoding(
    tmp_path: Path, dirty: bool | None, encoded: str
) -> None:
    metadata = replace(
        native_metadata(),
        finished_at=None,
        labber_path=None,
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=dirty,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    path = tmp_path / "unknown.h5"
    save_run_data(path, native_payload(), metadata, cfg=native_cfg())
    assert load_run_data(path).metadata == metadata
    with h5.File(path, "r") as file:
        assert file.attrs["finished_at"] == ""
        assert file.attrs["labber_path"] == ""
        provenance = file["provenance"]
        assert provenance.attrs["git_dirty"] == encoded
        for name in ("git_commit", "qick_version", "soc_fingerprint", "hostname"):
            assert provenance.attrs[name] == ""


@pytest.mark.parametrize(
    "attribute",
    [
        "software_versions",
        "git_commit",
        "git_dirty",
        "qick_version",
        "soc_fingerprint",
        "hostname",
    ],
)
def test_missing_provenance_is_not_silently_unknown(
    tmp_path: Path, attribute: str
) -> None:
    path = tmp_path / "missing.h5"
    save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(path, "r+") as file:
        del file["provenance"].attrs[attribute]
    with pytest.raises(ValueError, match=attribute) as error:
        load_run_data(path)
    assert str(path) in str(error.value)
    assert "/provenance" in str(error.value)
    assert attribute in str(error.value)


@pytest.mark.parametrize(
    "location,attribute,value",
    [
        ("/", "format_version", "2.0"),
        ("/", "completion", "finished"),
        ("/", "started_at", "not-a-time"),
        ("/data/readout/signal", "units", "A"),
        ("/provenance", "git_dirty", "yes"),
        ("/cfg", "cfg_schema_version", "broken"),
    ],
)
def test_loader_reports_source_and_location_for_invalid_wire_values(
    tmp_path: Path, location: str, attribute: str, value: str
) -> None:
    path = tmp_path / "invalid.h5"
    save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(path, "r+") as file:
        if attribute == "units":
            # A generic loader accepts arbitrary declared units, but a missing unit is invalid.
            del file[location].attrs[attribute]
        else:
            file[location].attrs[attribute] = value
    with pytest.raises(ValueError, match=attribute) as error:
        load_run_data(path)
    assert str(path) in str(error.value)
    assert location in str(error.value)


def test_native_loader_rejects_labber_without_fallback(tmp_path: Path) -> None:
    path = Path(
        save_labber_data(
            str(tmp_path / "labber.hdf5"),
            ("Signal", "", np.array([1.0, 2.0])),
            axes=[("x", "", np.array([0, 1]))],
        )
    )
    with pytest.raises(ValueError, match="format") as error:
        load_run_data(path)
    assert str(path) in str(error.value)
    assert "/" in str(error.value)


@pytest.mark.parametrize(
    "timestamps",
    [np.zeros((1, 1)), np.zeros(2), np.array([1j]), np.array(["bad"], dtype=object)],
)
def test_invalid_timestamps_fail_before_publication(
    tmp_path: Path, timestamps: np.ndarray
) -> None:
    payload = native_payload()
    payload.variables[DataVariable("readout")].timestamps = timestamps
    path = tmp_path / "never-created.h5"
    with pytest.raises(ValueError, match="timestamps"):
        save_run_data(path, payload, native_metadata(), cfg=native_cfg())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["", "a/b", "frequency", "timestamps"])
def test_invalid_signal_names_fail_before_publication(
    tmp_path: Path, name: str
) -> None:
    payload = native_payload()
    variable = payload.variables[DataVariable("readout")]
    payload = replace(
        payload,
        variables={
            DataVariable("readout"): LabberPayload(
                (name, "V", variable.z), axes=variable.axes
            )
        },
    )
    with pytest.raises(ValueError, match="readout"):
        save_run_data(
            tmp_path / "invalid.h5", payload, native_metadata(), cfg=native_cfg()
        )
    assert list(tmp_path.iterdir()) == []


def test_public_schema_validation_checks_wire_labels_units_dtypes_and_shape() -> None:
    payload = native_payload()
    schema = VariableSchema(
        variable=DataVariable("readout"),
        axes=(AxisSchema(name="frequency", unit="Hz", dtype=np.dtype("float64")),),
        signal_name="signal",
        signal_unit="V",
        signal_dtype=np.dtype("complex128"),
    )
    validate_experiment_payload(payload, schema=(schema,))
    for wrong in (
        replace(schema, signal_unit="A"),
        replace(schema, signal_name="other"),
        replace(schema, signal_dtype=np.dtype("float64")),
        replace(schema, axes=()),
        replace(schema, variable=DataVariable("missing")),
    ):
        with pytest.raises(ValueError, match="readout|missing"):
            validate_experiment_payload(payload, schema=(wrong,))
    payload = replace(
        payload,
        variables={
            DataVariable("readout"): LabberPayload(
                ("signal", "V", np.zeros(3, dtype=complex)),
                axes=payload.variables[DataVariable("readout")].axes,
            )
        },
    )
    with pytest.raises(ValueError, match="shape"):
        validate_experiment_payload(payload, schema=(schema,))


@pytest.mark.parametrize(
    "defect", ["axis_shape", "signal_dtype", "signal_shape", "axis_dtype"]
)
def test_schema_rejects_malformed_wire_arrays(defect: str) -> None:
    payload = native_payload()
    variable = payload.variables[DataVariable("readout")]
    values = np.array([1 + 2j, 3 - 4j])
    axes = [("frequency", "Hz", np.array([4e9, 5e9]))]
    if defect == "axis_shape":
        axes = [("frequency", "Hz", np.ones((1, 2)))]
    elif defect == "axis_dtype":
        axes = [("frequency", "Hz", np.array(["one", "two"]))]
    elif defect == "signal_dtype":
        variable = LabberPayload(("signal", "V", np.array(["one", "two"])), axes=axes)
    elif defect == "signal_shape":
        values = np.zeros(3, dtype=complex)
    if defect != "signal_dtype":
        variable = LabberPayload(("signal", "V", values), axes=axes)
    payload = replace(payload, variables={DataVariable("readout"): variable})
    schema = VariableSchema(
        variable=DataVariable("readout"),
        axes=(AxisSchema(name="frequency", unit="Hz", dtype=np.dtype("float64")),),
        signal_name="signal",
        signal_unit="V",
        signal_dtype=np.dtype("complex128"),
    )
    with pytest.raises(ValueError, match="readout"):
        validate_experiment_payload(payload, schema=(schema,))


def test_failed_atomic_publication_preserves_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "original.h5"
    save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    original = path.read_bytes()
    failure = OSError("synthetic publication failure")

    def fail_replace(_source: object, _destination: object) -> None:
        raise failure

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic publication failure") as error:
        save_run_data(
            path, native_payload(), native_metadata(), cfg=native_cfg(), replace=True
        )
    assert error.value is failure
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "defect",
    ["missing_cfg", "missing_context", "timestamps_shape", "partial_variable_markers"],
)
def test_loader_rejects_missing_or_malformed_known_layout(
    tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "malformed.h5"
    save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(path, "r+") as file:
        if defect == "missing_cfg":
            location = "/cfg"
            del file["cfg"]
        elif defect == "missing_context":
            location = "/context"
            del file["context"]
        elif defect == "timestamps_shape":
            location = "/data/readout/timestamps"
            del file[location]
            file.create_dataset(location, data=[1.0, 2.0]).attrs["units"] = "s"
        else:
            location = "/data/readout"
            del file[location].attrs["signal"]
    with pytest.raises(ValueError, match=location) as error:
        load_run_data(path)
    assert str(path) in str(error.value)


def test_exact_destination_conflict_replace_and_validation_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exact.h5"
    save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_run_data(path, native_payload(), native_metadata(), cfg=native_cfg())
    assert path.read_bytes() == original
    changed = replace(native_metadata(), completion="stopped", finished_at=None)
    save_run_data(path, native_payload(), changed, cfg=native_cfg(), replace=True)
    assert load_run_data(path).metadata == changed
    updated = path.read_bytes()
    invalid = native_payload()
    invalid.variables[DataVariable("readout")].timestamps = np.zeros(3)
    with pytest.raises(ValueError, match="timestamps"):
        save_run_data(path, invalid, native_metadata(), cfg=native_cfg(), replace=True)
    assert path.read_bytes() == updated
    assert list(tmp_path.iterdir()) == [path]
