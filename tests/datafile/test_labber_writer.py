"""Container to Labber export and independent-grid conversion contracts."""

from pathlib import Path

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
        loaded.variables["first"].z, container.vars["first"].values
    )
    assert loaded.metadata == metadata
    assert [axis.name for axis in loaded.variables["first"].axes] == [
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
        np.testing.assert_array_equal(loaded.variables[name].z, variable.values)
        assert [axis.name for axis in loaded.variables[name].axes] == [
            axis.name for axis in reversed(variable.axes)
        ]


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


@pytest.mark.parametrize(
    "text", ["operator note", "2024", "[1,2]", '"quoted"', '{"operator":"note"}', ""]
)
def test_comment_decoder_preserves_non_envelope_text(text: str) -> None:
    decoded = datafile.decode_labber_comment(text)
    assert decoded.cfg is None
    assert decoded.timestamp is None
    assert decoded.comment == text


@pytest.mark.parametrize(
    "text", ['{"cfg":[]}', '{"cfg":5}', '{"comment":false}', '{"timestamp":42}']
)
def test_comment_decoder_rejects_malformed_envelope_fields(text: str) -> None:
    with pytest.raises(ValueError, match="cfg|comment|timestamp"):
        datafile.decode_labber_comment(text)
