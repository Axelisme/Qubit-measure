"""Generic container public API and HDF5 wire contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from zcu_tools import datafile


def _dataset(file: h5py.File, name: str) -> h5py.Dataset:
    node = file[name]
    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"{name}: expected dataset")
    return node


def _container() -> datafile.Container:
    return datafile.Container(
        {}, {}, {}, datafile.ContainerAttrs("3.0", "synthetic", None, {})
    )


def test_namespaces_roundtrip_and_detached_groups(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as file:
        group = file.create_group("custom")
        group.attrs["label"] = "opaque"
        group.create_dataset("nested/counts", data=[7, 9])
    path = tmp_path / "container.h5"
    axis = datafile.Axis("frequency", "Hz", np.array([2.0, 4.0]))
    container = replace(
        _container(),
        vars={"signal": datafile.Variable(np.array([1 + 2j, 3 + 4j]), (axis,), "V")},
        docs={"custom": {"nested": [None, True, {"unknown": "保留"}]}},
        groups={"opaque": datafile.GroupSource(source, "/custom")},
    )
    datafile.save(path, container)
    loaded = datafile.load(path)
    path.unlink()
    source.unlink()
    rewritten = tmp_path / "rewritten.h5"
    datafile.save(rewritten, loaded)
    actual = datafile.load(rewritten)
    np.testing.assert_array_equal(actual.vars["signal"].values, [1 + 2j, 3 + 4j])
    assert actual.vars["signal"].axes[0].name == "frequency"
    assert actual.docs == {"custom": {"nested": [None, True, {"unknown": "保留"}]}}
    with h5py.File(rewritten, "r") as file:
        assert file.attrs["format"] == "zcu.container"
        assert file.attrs["format_version"] == "3.0"
        assert file["groups/opaque"].attrs["label"] == "opaque"
        np.testing.assert_array_equal(file["groups/opaque/nested/counts"], [7, 9])
        assert (
            _dataset(file, "vars/signal/values").dims[0][0].name
            == "/vars/signal/axes/frequency"
        )


@pytest.mark.parametrize("namespace", ["vars", "docs", "groups"])
def test_empty_namespaces_and_scalar(tmp_path: Path, namespace: str) -> None:
    path = tmp_path / "empty.h5"
    datafile.save(path, _container())
    assert getattr(datafile.load(path), namespace) == {}
    datafile.save(
        path,
        replace(_container(), vars={"scalar": datafile.Variable(np.array(3), (), "")}),
        replace=True,
    )
    assert datafile.load(path).vars["scalar"].values == 3


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\x00b"])
@pytest.mark.parametrize("namespace", ["vars", "docs", "groups"])
def test_invalid_names_fail_before_publication(
    tmp_path: Path, name: str, namespace: str
) -> None:
    path = tmp_path / "invalid.h5"
    container = _container()
    if namespace == "vars":
        container = replace(
            container, vars={name: datafile.Variable(np.array(1), (), "")}
        )
    elif namespace == "docs":
        container = replace(container, docs={name: {}})
    else:
        container = replace(
            container, groups={name: datafile.GroupSource(path, "/custom")}
        )
    with pytest.raises(ValueError, match="name"):
        datafile.save(path, container)
    assert not path.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_rejected_without_replacing(
    tmp_path: Path, value: float
) -> None:
    path = tmp_path / "valid.h5"
    datafile.save(path, replace(_container(), docs={"doc": {"value": 2}}))
    with pytest.raises(ValueError, match="JSON|finite"):
        datafile.save(
            path, replace(_container(), docs={"doc": {"nested": [value]}}), replace=True
        )
    assert datafile.load(path).docs["doc"] == {"value": 2}


def test_shape_and_duplicate_axes_are_rejected(tmp_path: Path) -> None:
    axis = datafile.Axis("x", "", np.array([1, 2]))
    with pytest.raises(ValueError, match="shape"):
        datafile.save(
            tmp_path / "shape.h5",
            replace(
                _container(), vars={"v": datafile.Variable(np.ones(3), (axis,), "")}
            ),
        )
    with pytest.raises(ValueError, match="duplicate"):
        datafile.save(
            tmp_path / "axes.h5",
            replace(
                _container(),
                vars={"v": datafile.Variable(np.ones((2, 2)), (axis, axis), "")},
            ),
        )


def test_inspect_reports_schema_without_reading_values(tmp_path: Path) -> None:
    path = tmp_path / "schema.h5"
    axis = datafile.Axis("x", "s", np.array([1, 2]))
    datafile.save(
        path,
        replace(
            _container(),
            vars={
                "v": datafile.Variable(np.array([1, 2], dtype=np.int16), (axis,), "A")
            },
            docs={"note": {"x": 1}},
        ),
    )
    info = datafile.inspect(path)
    assert info.vars["v"].shape == (2,)
    assert info.vars["v"].dtype == np.dtype("int16")
    assert info.vars["v"].unit == "A"
    assert info.docs == ("note",)
    assert info.attrs.producer == "synthetic"


def test_append_rows_replace_whole_document_and_finalize(tmp_path: Path) -> None:
    path = tmp_path / "append.h5"
    axes = (
        datafile.Axis("outer", "", np.array([0, 1])),
        datafile.Axis("inner", "s", np.array([2, 3, 4])),
    )
    schema = datafile.VariableSchema(
        shape=(2, 3), axes=axes, dtype=np.dtype("float64"), unit="V", fill_value=np.nan
    )
    with datafile.open_append(
        path, vars={"v": schema}, docs={"progress": {"old": True}}, producer="test"
    ) as writer:
        writer.write_row("v", 0, np.array([10.0, 11.0, 12.0]))
        writer.replace_doc("progress", {"row": 1})
    partial = datafile.load(path)
    np.testing.assert_array_equal(partial.vars["v"].values[0], [10, 11, 12])
    assert np.isnan(partial.vars["v"].values[1]).all()
    assert partial.docs == {"progress": {"row": 1}}
    assert not partial.attrs.extra["finalized"]
    with datafile.open_append(
        tmp_path / "complete.h5", vars={"v": schema}, producer="test"
    ) as writer:
        with pytest.raises(ValueError, match="shape"):
            writer.write_row("v", 0, np.ones(2))
        writer.write_row("v", 0, np.ones(3))
        writer.write_row("v", 1, np.ones(3) * 2)
        writer.finalize()
        with pytest.raises(ValueError, match="finalized"):
            writer.write_row("v", 0, np.ones(3))
    assert datafile.load(tmp_path / "complete.h5").attrs.extra["finalized"]


def test_forward_minor_attrs_survive_load_modify_save(tmp_path: Path) -> None:
    path = tmp_path / "future.h5"
    attrs = datafile.ContainerAttrs(
        "3.7", "original", "original-kind", {"unknown": "保留", "count": 2}
    )
    datafile.save(path, replace(_container(), attrs=attrs))
    loaded = datafile.load(path)
    assert loaded.attrs == attrs
    rewritten = tmp_path / "rewritten.h5"
    datafile.save(rewritten, replace(loaded, docs={"new": {"ok": True}}))
    assert datafile.load(rewritten).attrs == attrs
    for version in ("2.1", "4.0", "3.invalid"):
        with pytest.raises(datafile.FormatError, match="version|major"):
            datafile.save(
                tmp_path / "invalid-version.h5",
                replace(loaded, attrs=replace(attrs, format_version=version)),
            )


def test_legacy_native_converts_to_container_and_rewrites_major_three(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.h5"
    with h5py.File(source, "w") as file:
        file.attrs.update(
            format="zcu.experiment-data",
            format_version="2.9",
            run_id="historic",
            experiment="synthetic",
            started_at="",
            finished_at="",
            completion="unknown",
            labber_path="",
        )
        cfg = file.create_dataset("cfg", data='{"parameter": 7}')
        cfg.attrs.update(cfg_type="synthetic.Cfg", cfg_schema_version="1.2")
        file.create_dataset("context", data='{"point": "P", "future": [1, 2]}')
        file.create_dataset("device", data='{"device": {"value": 1.0}}')
        file.create_group("provenance").attrs.update(
            software_versions='{"python": "3"}', git_dirty="unknown", git_commit="abc"
        )
        file.create_dataset("future", data=[8, 9])
        data = file.create_group("data")
        data.attrs.update(representation="single", metadata='{"comment": "old"}')
        variable = data.create_group("signal")
        variable.attrs.update(signal="voltage", axes='["x"]')
        x = variable.create_dataset("x", data=[2, 4])
        x.attrs["units"] = "s"
        x.make_scale("x")
        values = variable.create_dataset("voltage", data=[10.0, 20.0])
        values.attrs["units"] = "V"
        values.dims[0].attach_scale(x)
    loaded = datafile.load(source)
    assert loaded.docs["cfg"] == {
        "values": {"parameter": 7},
        "cfg_type": "synthetic.Cfg",
        "schema_version": "1.2",
    }
    assert loaded.docs["metadata"]["snapshot"] == {"point": "P", "future": [1, 2]}
    assert loaded.docs["metadata"]["started_at"] is None
    np.testing.assert_array_equal(loaded.vars["signal"].values, [10, 20])
    destination = tmp_path / "converted.h5"
    datafile.save(destination, loaded)
    assert datafile.load(destination).attrs.format_version == "3.0"
    with h5py.File(destination, "r") as file:
        np.testing.assert_array_equal(file["groups/legacy/future"], [8, 9])


def test_unsupported_major_and_nonfinite_wire_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.h5"
    datafile.save(path, replace(_container(), docs={"note": {}}))
    with h5py.File(path, "r+") as file:
        file.attrs["format_version"] = "4.0"
    with pytest.raises(datafile.FormatError, match="4.0"):
        datafile.load(path)
    with h5py.File(path, "r+") as file:
        file.attrs["format_version"] = "3.8"
        _dataset(file, "docs/note")[()] = '{"value": NaN}'
    with pytest.raises(datafile.FormatError, match="JSON|finite"):
        datafile.load(path)
