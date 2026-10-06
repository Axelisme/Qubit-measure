"""Explicit native-major migration through the public datafile seam."""

import json
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
from zcu_tools.datafile import (
    DataVariable,
    JsonObject,
    load_run_data,
    migrate_native_run_v1,
    save_run_data,
)
from zcu_tools.format_version import FormatError

from tests._native_support import (
    hdf_dataset,
    hdf_json_text,
    native_cfg,
    native_metadata,
    native_payload,
)


def write_v1(source: Path) -> JsonObject:
    """Build synthetic native1 layout with forward-minor extensions."""
    save_run_data(source, native_payload(), native_metadata(), cfg=native_cfg())
    with h5.File(source, "r+") as file:
        file.attrs["format_version"] = "1.9"
        file.attrs["future_root"] = "kept"
        context: JsonObject = json.loads(hdf_json_text(file, "context"))
        context["future"] = {"unicode": "未知", "detail": [True, None, 7]}
        expected = dict(context)
        context["params"] = {
            "Q1.freq": {"value": 5000, "unit": "MHz", "source": {"future": "opaque"}}
        }
        hdf_dataset(file, "context")[()] = json.dumps(context)
        file["context_alias"] = hdf_dataset(file, "context")
        refs = file.create_dataset("future_refs", (2,), dtype=h5.ref_dtype)
        refs[0] = hdf_dataset(file, "context").ref
        signal = hdf_dataset(file, "data/readout/signal")
        refs[1] = signal.ref
        regions = file.create_dataset("future_regions", (1,), dtype=h5.regionref_dtype)
        regions[0] = signal.regionref[0:1]
        file["signal_alias"] = signal
        file["soft_signal"] = h5.SoftLink("/data/readout/signal")
        file["external"] = h5.ExternalLink("not-present.h5", "/future")
        file["provenance"].attrs["future"] = "unknown evidence"
        file.create_dataset("future_values", data=np.array([3, 9], dtype=np.int16))
    return expected


def test_migration_preserves_source_known_data_and_unknown_references(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "old.h5", tmp_path / "new.h5"
    expected_context = write_v1(source)
    before = source.read_bytes()
    with pytest.raises(FormatError, match="format_version"):
        load_run_data(source)

    assert migrate_native_run_v1(source, destination) == destination

    assert source.read_bytes() == before
    stored = load_run_data(destination)
    assert stored.metadata == native_metadata()
    assert stored.cfg == native_cfg()
    assert stored.payload.metadata == native_payload().metadata
    np.testing.assert_array_equal(
        stored.payload.variables[DataVariable("readout")].z,
        native_payload().variables[DataVariable("readout")].z,
    )
    with h5.File(destination, "r") as file:
        assert file.attrs["format_version"] == "2.0"
        assert file.attrs["future_root"] == "kept"
        assert json.loads(hdf_json_text(file, "context")) == expected_context
        assert file["provenance"].attrs["future"] == "unknown evidence"
        context = hdf_dataset(file, "context")
        signal = hdf_dataset(file, "data/readout/signal")
        refs = hdf_dataset(file, "future_refs")
        assert file[refs[0]].id == context.id == file["context_alias"].id
        assert file[refs[1]].id == signal.id == file["signal_alias"].id
        assert signal.id == file["soft_signal"].id
        assert isinstance(file.get("soft_signal", getlink=True), h5.SoftLink)
        external = file.get("external", getlink=True)
        assert isinstance(external, h5.ExternalLink)
        assert (external.filename, external.path) == ("not-present.h5", "/future")
        region = hdf_dataset(file, "future_regions")[0]
        assert file[region].id == signal.id
        np.testing.assert_array_equal(signal[region], [1 + 2j])
        np.testing.assert_array_equal(hdf_dataset(file, "future_values")[()], [3, 9])
        assert signal.dims[0][0].id == file["data/readout/frequency"].id


@pytest.mark.parametrize("link_kind", ["hard", "soft"])
def test_shared_cfg_context_conflict_never_publishes_changed_cfg(
    tmp_path: Path, link_kind: str
) -> None:
    source, destination = tmp_path / "shared.h5", tmp_path / "new.h5"
    write_v1(source)
    with h5.File(source, "r+") as file:
        context = json.loads(hdf_json_text(file, "context"))
        context["params"] = {}
        hdf_dataset(file, "context")[()] = json.dumps(context)
        cfg_attrs = dict(hdf_dataset(file, "cfg").attrs)
        del file["cfg"]
        if link_kind == "hard":
            file["cfg"] = hdf_dataset(file, "context")
        else:
            file["cfg"] = h5.SoftLink("/context")
        hdf_dataset(file, "context").attrs.update(cfg_attrs)
        # The unchanged known layout is valid, including cfg's complete JSON.
        file.attrs["format_version"] = "2.0"
    assert load_run_data(source).cfg.values == context
    with h5.File(source, "r+") as file:
        file.attrs["format_version"] = "1.0"
    before = source.read_bytes()

    with pytest.raises(ValueError, match=r"/context.*shared.*?/cfg") as caught:
        migrate_native_run_v1(source, destination)

    assert str(source) in str(caught.value)
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("version", ["2.0", "3.0", "0.9", "1.bad"])
def test_migration_rejects_wrong_major_or_invalid_version_without_publication(
    tmp_path: Path, version: str
) -> None:
    source, destination = tmp_path / "old.h5", tmp_path / "new.h5"
    write_v1(source)
    with h5.File(source, "r+") as file:
        file.attrs["format_version"] = version
    before = source.read_bytes()
    with pytest.raises(FormatError, match="format_version") as caught:
        migrate_native_run_v1(source, destination)
    assert str(source) in str(caught.value)
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("defect", ["root", "context", "params", "payload", "external"])
def test_invalid_source_reports_location_and_cleans_temporary_copy(
    tmp_path: Path, defect: str
) -> None:
    source, destination = tmp_path / "old.h5", tmp_path / "new.h5"
    write_v1(source)
    with h5.File(source, "r+") as file:
        if defect == "root":
            del file.attrs["started_at"]
        elif defect == "context":
            context = json.loads(hdf_json_text(file, "context"))
            del context["entry_id"]
            hdf_dataset(file, "context")[()] = json.dumps(context)
        elif defect == "params":
            context = json.loads(hdf_json_text(file, "context"))
            context["params"] = None
            hdf_dataset(file, "context")[()] = json.dumps(context)
        elif defect == "payload":
            del file["data/readout/signal"].attrs["units"]
        else:
            del file["cfg"]
            file["cfg"] = h5.ExternalLink("not-present.h5", "/cfg")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="/") as caught:
        migrate_native_run_v1(source, destination)
    assert str(source) in str(caught.value)
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {source}


def test_migration_never_overwrites_destination_or_rewrites_source(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "old.h5", tmp_path / "new.h5"
    write_v1(source)
    before = source.read_bytes()
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        migrate_native_run_v1(source, destination)
    with pytest.raises(ValueError, match="different"):
        migrate_native_run_v1(source, source)
    alias = tmp_path / "alias.h5"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="different"):
        migrate_native_run_v1(source, alias)
    assert destination.read_bytes() == b"existing"
    assert source.read_bytes() == before
