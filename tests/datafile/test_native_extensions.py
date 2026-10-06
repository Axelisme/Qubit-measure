"""Forward-minor generic rewrites preserve unknown data through the public seam."""

import json
from dataclasses import replace
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
from zcu_tools.datafile import load_run_data, save_run_data

from tests._native_support import (
    hdf_dataset,
    hdf_group,
    hdf_json_text,
    native_cfg,
    native_metadata,
    native_payload,
)


@pytest.mark.parametrize(
    "location",
    [
        "cfg",
        "context",
        "provenance",
        "data",
        "data/readout/signal",
        "data/readout/frequency",
        "data/readout/timestamps",
    ],
)
@pytest.mark.parametrize("route", ["direct", "soft_chain", "soft_parent"])
@pytest.mark.parametrize("preserve_unknown", [False, True])
def test_known_nodes_reject_external_targets_before_dereferencing(
    tmp_path: Path,
    location: str,
    route: str,
    preserve_unknown: bool,
) -> None:
    source, target = tmp_path / "source.h5", tmp_path / "external.h5"
    save_run_data(source, native_payload(), native_metadata(), cfg=native_cfg())
    target.write_bytes(source.read_bytes())
    before = target.read_bytes()
    with h5.File(source, "r+") as file:
        del file[location]
        if route == "direct":
            file[location] = h5.ExternalLink(str(target), "/" + location)
        elif route == "soft_chain":
            file["outside"] = h5.ExternalLink(str(target), "/" + location)
            file["hop"] = h5.SoftLink("/outside")
            file[location] = h5.SoftLink("/hop")
        else:
            file["outside"] = h5.ExternalLink(str(target), "/")
            file[location] = h5.SoftLink("/outside/" + location)
    with pytest.raises(ValueError) as caught:
        load_run_data(source, preserve_unknown=preserve_unknown)
    assert str(source) in str(caught.value)
    assert "/" + location in str(caught.value)
    assert "external" in str(caught.value).lower()
    assert target.read_bytes() == before
    # The same boundary must fail before trying to open an absent target.
    target.unlink()
    with pytest.raises(ValueError, match="external"):
        load_run_data(source, preserve_unknown=preserve_unknown)


def _write_fixed_json(source: Path, *, spare_bytes: int = 0) -> dict[str, str]:
    """Build legal fixed UTF-8 datasets with aliases and references."""
    save_run_data(source, native_payload(), native_metadata(), cfg=native_cfg())
    raw: dict[str, str] = {}
    with h5.File(source, "r+") as file:
        context = json.loads(hdf_json_text(file, "context"))
        context["future"] = {"unicode": "未知", "number": 1e-6}
        context["params"]["Q1.freq"]["source"]["future"] = "kept"
        raw["cfg"] = '{"a":1,"b":2,"future":"未知","number":1e-6}'
        raw["context"] = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        refs = file.create_dataset("json_refs", (2,), dtype=h5.ref_dtype)
        for index, name in enumerate(("cfg", "context")):
            old = hdf_dataset(file, name)
            attrs = dict(old.attrs)
            del file[name]
            dataset = file.create_dataset(
                name,
                data=raw[name].encode("utf-8"),
                dtype=h5.string_dtype(
                    "utf-8", length=len(raw[name].encode("utf-8")) + spare_bytes
                ),
            )
            dataset.attrs.update(attrs)
            file[name + "_alias"] = dataset
            refs[index] = dataset.ref
    return raw


def test_fixed_json_roundtrip_retains_values_raw_text_aliases_and_references(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5"
    raw = _write_fixed_json(source)
    stored = load_run_data(source, preserve_unknown=True)
    source.unlink()
    destination = tmp_path / "rewritten.h5"
    save_run_data(
        destination,
        stored.payload,
        stored.metadata,
        cfg=stored.cfg,
        extensions=stored.extensions,
    )
    actual = load_run_data(destination)
    assert actual.cfg == stored.cfg
    assert actual.metadata == stored.metadata
    with h5.File(destination, "r") as file:
        for index, name in enumerate(("cfg", "context")):
            dataset = hdf_dataset(file, name)
            assert hdf_json_text(file, name) == raw[name]
            assert dataset.id == file[name + "_alias"].id
            assert dataset.id == file[hdf_dataset(file, "json_refs")[index]].id


@pytest.mark.parametrize("name", ["cfg", "context"])
@pytest.mark.parametrize("fits", [True, False])
def test_fixed_json_edits_obey_utf8_capacity_and_atomic_failure(
    tmp_path: Path,
    name: str,
    fits: bool,
) -> None:
    source = tmp_path / "source.h5"
    _write_fixed_json(source, spare_bytes=256)
    stored = load_run_data(source, preserve_unknown=True)
    source.unlink()
    value = "新" * (3 if fits else 1000)
    cfg, metadata = stored.cfg, stored.metadata
    if name == "cfg":
        cfg = replace(cfg, values={"new": value})
    else:
        metadata = replace(
            metadata, snapshot=replace(metadata.snapshot, description=value)
        )
    destination = tmp_path / "destination.h5"
    destination.write_bytes(b"existing destination")
    if not fits:
        before = set(tmp_path.iterdir())
        with pytest.raises(ValueError) as caught:
            save_run_data(
                destination,
                stored.payload,
                metadata,
                cfg=cfg,
                extensions=stored.extensions,
                replace=True,
            )
        assert str(destination) in str(caught.value)
        assert "/" + name in str(caught.value)
        assert "capacity" in str(caught.value)
        assert destination.read_bytes() == b"existing destination"
        assert set(tmp_path.iterdir()) == before
    else:
        save_run_data(
            destination,
            stored.payload,
            metadata,
            cfg=cfg,
            extensions=stored.extensions,
            replace=True,
        )
        actual = load_run_data(destination)
        assert actual.cfg == cfg
        assert actual.metadata == metadata
        with h5.File(destination, "r") as file:
            dataset = hdf_dataset(file, name)
            index = 0 if name == "cfg" else 1
            assert dataset.id == file[name + "_alias"].id
            assert dataset.id == file[hdf_dataset(file, "json_refs")[index]].id


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
