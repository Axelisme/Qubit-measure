"""Container to Labber export and independent-grid conversion contracts."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from zcu_tools import datafile


def _container(*, independent: bool = False) -> datafile.Container:
    first = datafile.Variable(
        np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]),
        (
            datafile.Axis("power", "dBm", np.array([-20.0, -10.0])),
            datafile.Axis("frequency", "Hz", np.array([4.0e9, 5.0e9])),
        ),
        "V",
    )
    second = (
        datafile.Variable(
            np.array([2.0, 3.0, 4.0]),
            (datafile.Axis("delay", "s", np.array([1.0e-6, 2.0e-6, 3.0e-6])),),
            "A",
        )
        if independent
        else first
    )
    return datafile.Container(
        {"first": first, "second": second},
        {"opaque": {"arbitrary": True}},
        {},
        datafile.ContainerAttrs("3.0", "synthetic", None, {}),
    )


def test_grouped_export_preserves_arrays_and_explicit_labber_metadata(
    tmp_path: Path,
) -> None:
    container = _container()
    metadata = datafile.LabberMetadata(
        comment="operator text",
        tags=["a", "b"],
        project="P",
        user="U",
        creation_time=1760000000.0,
    )
    path = tmp_path / "exact.export"
    datafile.write_labber(path, container, metadata=metadata)
    loaded = datafile.load_grouped_labber_data(str(path))
    np.testing.assert_array_equal(
        loaded.variables[datafile.DataVariable("first")].z,
        container.vars["first"].values,
    )
    assert loaded.metadata == metadata
    assert [
        axis.name for axis in loaded.variables[datafile.DataVariable("first")].axes
    ] == [
        "frequency",
        "power",
    ]


def test_single_export_and_read_labber_retain_metadata(tmp_path: Path) -> None:
    original = _container()
    container = datafile.Container(
        {"signal": original.vars["first"]}, {}, {}, original.attrs
    )
    metadata = datafile.LabberMetadata(
        comment="manual",
        tags=["signal"],
        project="P",
        user="U",
        creation_time=1760000000.0,
    )
    path = tmp_path / "single.hdf5"
    datafile.write_labber(path, container, metadata=metadata)
    actual, actual_metadata = datafile.read_labber(path)
    assert actual_metadata == metadata
    np.testing.assert_array_equal(
        actual.vars["signal"].values, container.vars["signal"].values
    )
    assert [axis.name for axis in actual.vars["signal"].axes] == ["power", "frequency"]


def test_independent_export_keeps_each_grid_and_rank(tmp_path: Path) -> None:
    original = _container(independent=True)
    path = tmp_path / "independent.hdf5"
    datafile.write_labber(path, original, grouped_layout="independent")
    loaded = datafile.load_grouped_labber_data(str(path))
    for name, variable in original.vars.items():
        np.testing.assert_array_equal(
            loaded.variables[datafile.DataVariable(name)].z, variable.values
        )
        assert [
            axis.name for axis in loaded.variables[datafile.DataVariable(name)].axes
        ] == [axis.name for axis in reversed(variable.axes)]


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_independent_rank_three_exports_first_axis_as_named_2d_logs(
    tmp_path: Path, dtype: type[np.float64] | type[np.complex128]
) -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4).astype(dtype)
    if dtype == np.complex128:
        values += 2j
    axes = (
        datafile.Axis("Sensor", "", np.array([7, 7])),
        datafile.Axis("bias", "V", np.array([-0.1, 0.0, 0.1])),
        datafile.Axis("frequency", "Hz", np.array([1e9, 2e9, 3e9, 4e9])),
    )
    original = _container(independent=True)
    container = datafile.Container(
        {"signal": datafile.Variable(values, axes, "V"), **original.vars},
        {},
        {},
        original.attrs,
    )
    metadata = datafile.LabberMetadata(comment="split", creation_time=1760000000.0)
    path = tmp_path / "slices.hdf5"
    datafile.write_labber(
        path, container, metadata=metadata, grouped_layout="independent"
    )

    loaded = datafile.load_grouped_labber_data(str(path))
    assert list(loaded.variables) == ["signal_0", "signal_1", "first", "second"]
    assert loaded.metadata == metadata
    projected, projected_metadata = datafile.read_labber(path)
    assert projected_metadata == metadata
    for index in range(2):
        payload = loaded.variables[f"signal_{index}"]
        np.testing.assert_array_equal(payload.z, values[index])
        assert payload.data.unit == "V"
        assert [(axis.name, axis.unit) for axis in payload.axes] == [
            ("frequency", "Hz"),
            ("bias", "V"),
        ]
        for actual, expected in zip(payload.axes, reversed(axes[1:]), strict=True):
            np.testing.assert_array_equal(actual.values, expected.values)
        np.testing.assert_array_equal(
            projected.vars[f"signal_{index}"].values, values[index]
        )
        assert [axis.name for axis in projected.vars[f"signal_{index}"].axes] == [
            "bias",
            "frequency",
        ]
    for name, variable in original.vars.items():
        np.testing.assert_array_equal(loaded.variables[name].z, variable.values)

    # These labels and groups are the Labber wire contract visible to other readers.
    with h5py.File(path, "r") as file:
        logs = [file, file["Log_2"], file["Log_3"], file["Log_4"]]
        assert [log.attrs["log_name"] for log in logs] == [
            "slices - signal Sensor=7",
            "slices - signal Sensor=7",
            "slices",
            "slices",
        ]


@pytest.mark.parametrize("collision_first", [False, True])
def test_independent_slice_identity_collision_raises_before_creation(
    tmp_path: Path, collision_first: bool
) -> None:
    original = _container()
    cube = datafile.Variable(
        np.ones((2, 2, 2)),
        (datafile.Axis("Sensor", "", np.array([4, 8])), *original.vars["first"].axes),
        "V",
    )
    items = [("signal", cube), ("signal_0", original.vars["first"])]
    if collision_first:
        items.reverse()
    container = datafile.Container(dict(items), {}, {}, original.attrs)
    path = tmp_path / "collision.hdf5"
    with pytest.raises(ValueError, match="duplicate.*signal_0|collision.*signal_0"):
        datafile.write_labber(path, container, grouped_layout="independent")
    assert not path.exists()


def test_independent_rank_four_raises_before_creation(tmp_path: Path) -> None:
    axes = tuple(datafile.Axis(name, "", np.arange(2)) for name in ("a", "b", "c", "d"))
    container = datafile.Container(
        {"signal": datafile.Variable(np.ones((2, 2, 2, 2)), axes, "V")},
        {},
        {},
        _container().attrs,
    )
    path = tmp_path / "rank_four.hdf5"
    with pytest.raises(ValueError, match="rank"):
        datafile.write_labber(path, container, grouped_layout="independent")
    assert not path.exists()


def test_common_grid_failure_does_not_affect_native_save(tmp_path: Path) -> None:
    container = _container(independent=True)
    with pytest.raises(ValueError, match="common"):
        datafile.write_labber(tmp_path / "invalid.hdf5", container)
    native = tmp_path / "data.h5"
    datafile.save(native, container)
    np.testing.assert_array_equal(
        datafile.load(native).vars["second"].values, [2, 3, 4]
    )


def test_export_refuses_existing_path_and_scalar_before_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.hdf5"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        datafile.write_labber(path, _container())
    assert path.read_bytes() == b"original"
    container = datafile.Container(
        {"scalar": datafile.Variable(np.array(1), (), "")}, {}, {}, _container().attrs
    )
    with pytest.raises(ValueError, match="axis|rank"):
        datafile.write_labber(tmp_path / "scalar.hdf5", container)
