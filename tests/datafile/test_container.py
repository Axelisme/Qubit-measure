"""Generic container public API and HDF5 wire contracts."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from zcu_tools import datafile


def test_namespaces_roundtrip_and_detached_groups(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as file:
        group = file.create_group("custom")
        group.attrs["label"] = "opaque"
        group.create_dataset("nested/counts", data=[7, 9])
    path = tmp_path / "container.h5"
    axis = datafile.Axis("frequency", "Hz", np.array([2.0, 4.0]))
    datafile.save(
        path,
        vars={"signal": datafile.Variable(np.array([1 + 2j, 3 + 4j]), (axis,), "V")},
        docs={"custom": {"nested": [None, True, {"unknown": "保留"}]}},
        groups={"opaque": datafile.GroupSource(source, "/custom")},
        producer="synthetic",
        kind="example",
    )
    loaded = datafile.load(path)
    path.unlink()
    source.unlink()
    rewritten = tmp_path / "rewritten.h5"
    datafile.save(rewritten, vars=loaded.vars, docs=loaded.docs, groups=loaded.groups, producer="rewrite")
    actual = datafile.load(rewritten)
    np.testing.assert_array_equal(actual.vars["signal"].values, [1 + 2j, 3 + 4j])
    assert actual.vars["signal"].axes[0].name == "frequency"
    assert actual.docs == {"custom": {"nested": [None, True, {"unknown": "保留"}]}}
    with h5py.File(rewritten, "r") as file:
        assert file.attrs["format"] == "zcu.container"
        assert file.attrs["format_version"] == "3.0"
        assert file["groups/opaque"].attrs["label"] == "opaque"
        np.testing.assert_array_equal(file["groups/opaque/nested/counts"], [7, 9])
        signal = file["vars/signal/values"]
        assert signal.dims[0][0].name == "/vars/signal/axes/frequency"


@pytest.mark.parametrize("namespace", ["vars", "docs", "groups"])
def test_empty_namespaces_and_scalar(tmp_path: Path, namespace: str) -> None:
    path = tmp_path / "empty.h5"
    datafile.save(path, vars={}, docs={}, producer="test")
    loaded = datafile.load(path)
    assert getattr(loaded, namespace) == {}
    datafile.save(path, vars={"scalar": datafile.Variable(np.array(3), (), "")}, docs={}, producer="test", replace=True)
    assert datafile.load(path).vars["scalar"].values == 3


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\x00b"])
@pytest.mark.parametrize("namespace", ["vars", "docs", "groups"])
def test_invalid_names_fail_before_publication(tmp_path: Path, name: str, namespace: str) -> None:
    path = tmp_path / "invalid.h5"
    item = datafile.Variable(np.array(1), (), "")
    with pytest.raises(ValueError, match="name"):
        if namespace == "vars":
            datafile.save(path, vars={name: item}, docs={}, producer="test")
        elif namespace == "docs":
            datafile.save(path, vars={}, docs={name: {}}, producer="test")
        else:
            datafile.save(path, vars={}, docs={}, groups={name: datafile.GroupSource(path, "/custom")}, producer="test")
    assert not path.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_rejected_without_replacing(tmp_path: Path, value: float) -> None:
    path = tmp_path / "valid.h5"
    datafile.save(path, vars={}, docs={"doc": {"value": 2}}, producer="test")
    with pytest.raises(ValueError, match="JSON|finite"):
        datafile.save(path, vars={}, docs={"doc": {"nested": [value]}}, producer="test", replace=True)
    assert datafile.load(path).docs["doc"] == {"value": 2}


def test_shape_and_duplicate_axes_are_rejected(tmp_path: Path) -> None:
    axis = datafile.Axis("x", "", np.array([1, 2]))
    with pytest.raises(ValueError, match="shape"):
        datafile.save(tmp_path / "shape.h5", vars={"v": datafile.Variable(np.ones(3), (axis,), "")}, docs={}, producer="test")
    with pytest.raises(ValueError, match="duplicate"):
        datafile.save(tmp_path / "axes.h5", vars={"v": datafile.Variable(np.ones((2, 2)), (axis, axis), "")}, docs={}, producer="test")


def test_inspect_reports_schema_without_reading_values(tmp_path: Path) -> None:
    path = tmp_path / "schema.h5"
    axis = datafile.Axis("x", "s", np.array([1, 2]))
    datafile.save(path, vars={"v": datafile.Variable(np.array([1, 2], dtype=np.int16), (axis,), "A")}, docs={"note": {"x": 1}}, producer="test")
    info = datafile.inspect(path)
    assert info.vars["v"].shape == (2,)
    assert info.vars["v"].dtype == np.dtype("int16")
    assert info.vars["v"].unit == "A"
    assert info.docs == ("note",)
    assert info.attrs["producer"] == "test"


def test_append_rows_replace_whole_document_and_finalize(tmp_path: Path) -> None:
    path = tmp_path / "append.h5"
    axes = (datafile.Axis("outer", "", np.array([0, 1])), datafile.Axis("inner", "s", np.array([2, 3, 4])))
    schema = datafile.VariableSchema(shape=(2, 3), axes=axes, dtype=np.dtype("float64"), unit="V", fill_value=np.nan)
    with datafile.open_append(path, vars={"v": schema}, docs={"progress": {"old": True}}, producer="test") as writer:
        writer.write_row("v", 0, np.array([10.0, 11.0, 12.0]))
        writer.replace_doc("progress", {"row": 1})
    partial = datafile.load(path)
    np.testing.assert_array_equal(partial.vars["v"].values[0], [10, 11, 12])
    assert np.isnan(partial.vars["v"].values[1]).all()
    assert partial.docs == {"progress": {"row": 1}}
    assert not partial.attrs["finalized"]
    with datafile.open_append(tmp_path / "complete.h5", vars={"v": schema}, producer="test") as writer:
        with pytest.raises(ValueError, match="shape"):
            writer.write_row("v", 0, np.ones(2))
        writer.write_row("v", 0, np.ones(3))
        writer.write_row("v", 1, np.ones(3) * 2)
        writer.finalize()
        with pytest.raises(ValueError, match="finalized"):
            writer.write_row("v", 0, np.ones(3))
    assert datafile.load(tmp_path / "complete.h5").attrs["finalized"]


def test_unsupported_major_and_nonfinite_wire_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.h5"
    datafile.save(path, vars={}, docs={"note": {}}, producer="test")
    with h5py.File(path, "r+") as file:
        file.attrs["format_version"] = "4.0"
    with pytest.raises(datafile.FormatError, match="4.0"):
        datafile.load(path)
    with h5py.File(path, "r+") as file:
        file.attrs["format_version"] = "3.8"
        file["docs/note"][()] = '{"value": NaN}'
    with pytest.raises(datafile.FormatError, match="JSON|finite"):
        datafile.load(path)
