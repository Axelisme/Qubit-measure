"""Forward-minor generic rewrites preserve unknown data through the public seam."""

import json
from dataclasses import replace
from pathlib import Path

import h5py as h5
import numpy as np
from zcu_tools.datafile import load_run_data, save_run_data

from tests._native_support import (
    hdf_dataset,
    hdf_group,
    hdf_json_text,
    native_cfg,
    native_metadata,
    native_payload,
)


def _write_future_native(source: Path) -> None:
    """Create synthetic forward-minor HDF5 extensions for the rewrite contract."""
    save_run_data(source, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(source, "r+") as file:
        file.attrs["format_version"] = "1.9"
        file.attrs["future"] = "root-value"
        variable = hdf_group(file, "data/readout")
        variable.attrs["future"] = 17
        signal = hdf_dataset(file, "data/readout/signal")
        signal.attrs["future"] = "dataset-value"
        compound = file.create_dataset(
            "future_compound",
            data=np.array([(7, 2.5)], dtype=[("count", "i4"), ("weight", "f8")]),
        )
        ragged = file.create_dataset(
            "future_ragged", (2,), dtype=h5.vlen_dtype(np.dtype("int32"))
        )
        ragged[0], ragged[1] = (
            np.array([1, 2], dtype="int32"),
            np.array([3], dtype="int32"),
        )
        file.create_dataset(
            "future_text", data="未知值", dtype=h5.string_dtype("utf-8")
        )
        file["hard_signal"] = signal
        file["soft_signal"] = h5.SoftLink("/data/readout/signal")
        file["external"] = h5.ExternalLink("not-present.h5", "/unknown")
        references = file.create_dataset("future_references", (2,), dtype=h5.ref_dtype)
        references[0], references[1] = signal.ref, compound.ref
        regions = file.create_dataset("future_regions", (1,), dtype=h5.regionref_dtype)
        regions[0] = signal.regionref[0:1]
        cfg = json.loads(hdf_json_text(file, "cfg"))
        cfg["future"] = {"array": [None, {"leaf": "kept"}]}
        hdf_dataset(file, "cfg")[()] = json.dumps(cfg)
        context = json.loads(hdf_json_text(file, "context"))
        context["future"] = {"leaf": [False, 9]}
        param = context["params"]["Q1.freq"]
        param["future"] = "parameter"
        param["source"]["future"] = "source"
        param["source"]["cloned_from"]["future"] = "clone"
        hdf_dataset(file, "context")[()] = json.dumps(context)
        # Unmarked data children are extensions, not malformed variables.
        hdf_group(file, "data").create_group("future_layout").create_dataset(
            "opaque", data=[11]
        )


def test_detached_image_preserves_minor_unknown_nodes_links_and_references(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    _write_future_native(source)
    with h5.File(source, "r") as file:
        expected_context = json.loads(hdf_json_text(file, "context"))
    ordinary = load_run_data(source)
    assert ordinary.extensions is None
    assert ordinary.cfg.values["future"] == {"array": [None, {"leaf": "kept"}]}
    assert ordinary.metadata.snapshot == native_metadata().snapshot
    stored = load_run_data(source, preserve_unknown=True)
    assert stored.extensions is not None
    assert isinstance(stored.extensions.file_image, bytes)
    source.unlink()
    destination = tmp_path / "rewritten.h5"
    save_run_data(
        destination,
        stored.payload,
        stored.metadata,
        cfg=stored.cfg,
        extensions=stored.extensions,
    )
    assert load_run_data(destination).metadata == stored.metadata
    with h5.File(destination, "r") as file:
        assert file.attrs["format_version"] == "1.9"
        assert file.attrs["future"] == "root-value"
        assert hdf_group(file, "data/readout").attrs["future"] == 17
        signal = hdf_dataset(file, "data/readout/signal")
        assert signal.attrs["future"] == "dataset-value"
        assert signal.id == file["hard_signal"].id
        assert signal.id == file["soft_signal"].id
        assert isinstance(file.get("soft_signal", getlink=True), h5.SoftLink)
        external = file.get("external", getlink=True)
        assert isinstance(external, h5.ExternalLink)
        assert (external.filename, external.path) == ("not-present.h5", "/unknown")
        np.testing.assert_array_equal(
            hdf_dataset(file, "future_compound")[()],
            np.array([(7, 2.5)], dtype=[("count", "i4"), ("weight", "f8")]),
        )
        np.testing.assert_array_equal(hdf_dataset(file, "future_ragged")[0], [1, 2])
        np.testing.assert_array_equal(hdf_dataset(file, "future_ragged")[1], [3])
        assert hdf_dataset(file, "future_text").asstr()[()] == "未知值"
        refs = hdf_dataset(file, "future_references")[()]
        assert file[refs[0]].id == signal.id
        assert file[refs[1]].id == hdf_dataset(file, "future_compound").id
        region = hdf_dataset(file, "future_regions")[0]
        assert file[region].id == signal.id
        np.testing.assert_array_equal(signal[region], np.array([1 + 2j]))
        assert signal.dims[0][0].id == file["data/readout/frequency"].id
        assert json.loads(hdf_json_text(file, "cfg")) == stored.cfg.values
        actual_context = json.loads(hdf_json_text(file, "context"))
        assert actual_context == expected_context
        np.testing.assert_array_equal(
            hdf_dataset(file, "data/future_layout/opaque")[()], [11]
        )


def test_image_rewrite_updates_known_values_without_merging_dynamic_maps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    save_run_data(source, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(source, "r+") as file:
        context = json.loads(hdf_json_text(file, "context"))
        context["future"] = "kept"
        context["params"]["Q1.freq"]["future"] = "param-kept"
        context["params"]["Q1.freq"]["source"]["future"] = "source-kept"
        hdf_dataset(file, "context")[()] = json.dumps(context)
        file["signal_alias"] = hdf_dataset(file, "data/readout/signal")
    stored = load_run_data(source, preserve_unknown=True)
    snapshot = stored.metadata.snapshot
    parameter = snapshot.params["Q1.freq"]
    updated_parameter = replace(
        parameter,
        value={"new": [True, None]},
        source=replace(parameter.source, cloned_from=None),
    )
    metadata = replace(
        stored.metadata,
        snapshot=replace(
            snapshot, roles={"resonator": "R2"}, params={"Q1.freq": updated_parameter}
        ),
    )
    payload = stored.payload
    payload.variables[next(iter(payload.variables))].z[:] = [5 + 6j, 7 + 8j]
    cfg = replace(stored.cfg, values={"only": "new"})
    destination = tmp_path / "updated.h5"
    save_run_data(destination, payload, metadata, cfg=cfg, extensions=stored.extensions)
    with h5.File(destination, "r") as file:
        actual = json.loads(hdf_json_text(file, "context"))
        assert actual["future"] == "kept"
        assert actual["roles"] == {"resonator": "R2"}
        assert set(actual["params"]) == {"Q1.freq"}
        param = actual["params"]["Q1.freq"]
        assert param["value"] == {"new": [True, None]}
        assert param["future"] == "param-kept"
        assert param["source"]["future"] == "source-kept"
        assert param["source"]["cloned_from"] is None
        assert json.loads(hdf_json_text(file, "cfg")) == {"only": "new"}
        assert (
            hdf_dataset(file, "signal_alias").id
            == hdf_dataset(file, "data/readout/signal").id
        )
        np.testing.assert_array_equal(
            hdf_dataset(file, "signal_alias")[()], [5 + 6j, 7 + 8j]
        )
