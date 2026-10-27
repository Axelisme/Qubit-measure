from __future__ import annotations

import h5py
import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.io import dump_spectrums, load_spectrums
from zcu_tools.analysis.fluxdep.models import SpectrumResult


@pytest.mark.parametrize("include_type", [False, True])
def test_spectrum_collection_roundtrips_values_shapes_and_dtypes(
    tmp_path, include_type
):
    first = SpectrumResult(
        offset=0.25,
        period=1.0,
        device="bias",
        unit="A",
        device_snapshot={"coil": {"value": 0.002, "unit": "A"}},
        spectrum={
            "dev_values": np.array([0.0, 1.0]),
            "fluxs": np.array([0.25, 0.75]),
            "freqs": np.array([4.0, 5.0, 6.0]),
            "signals": np.array([[1 + 2j, 3 - 4j, 5j], [6j, 7 + 8j, 9 - 1j]]),
        },
        points={
            "dev_values": np.array([], dtype=np.float64),
            "fluxs": np.array([], dtype=np.float64),
            "freqs": np.array([], dtype=np.float64),
        },
    )
    second = SpectrumResult(
        offset=0.75,
        period=-2.0,
        device="unnamed_device",
        unit="native",
        spectrum=first["spectrum"],
        points={
            "dev_values": np.array([0.0, 1.0]),
            "fluxs": np.array([0.25, 0.75]),
            "freqs": np.array([4.0, 6.0]),
        },
    )
    if include_type:
        first["type"] = "OneTone"
        second["type"] = "TwoTone"
    expected = {"first": first, "second": second}
    path = str(tmp_path / "spectrums.hdf5")
    dump_spectrums(path, expected)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["format"] == "zcu.fluxdep-spectrums"
        assert handle.attrs["version"] == 2
        assert handle["first"].attrs["offset"] == 0.25
        snapshot = handle["first/device_snapshot/coil"]
        assert isinstance(snapshot, h5py.Dataset)
        assert snapshot[()] == 0.002
        assert snapshot.attrs["unit"] == "A"
    actual = load_spectrums(path)
    assert actual.keys() == expected.keys()
    for name, source in expected.items():
        result = actual[name]
        assert result["offset"] == source["offset"]
        assert result["period"] == source["period"]
        assert result["device"] == source["device"]
        assert result["unit"] == source["unit"]
        assert result.get("device_snapshot") == source.get("device_snapshot")
        assert result.get("type") == source.get("type")
        pairs = (
            (result["spectrum"]["dev_values"], source["spectrum"]["dev_values"]),
            (result["spectrum"]["fluxs"], source["spectrum"]["fluxs"]),
            (result["spectrum"]["freqs"], source["spectrum"]["freqs"]),
            (result["spectrum"]["signals"], source["spectrum"]["signals"]),
            (result["points"]["dev_values"], source["points"]["dev_values"]),
            (result["points"]["fluxs"], source["points"]["fluxs"]),
            (result["points"]["freqs"], source["points"]["freqs"]),
        )
        for restored, array in pairs:
            np.testing.assert_array_equal(restored, array)
            assert restored.shape == array.shape
            assert restored.dtype == array.dtype


@pytest.mark.parametrize("version", [None, 1, 3])
def test_unsupported_collection_requires_explicit_upgrade(tmp_path, version):
    path = tmp_path / "old.hdf5"
    with h5py.File(path, "w") as handle:
        if version is not None:
            handle.attrs.update(format="zcu.fluxdep-spectrums", version=version)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="scripts/upgrade_spectrums.py"):
        load_spectrums(str(path))
    assert path.read_bytes() == original


def test_load_spectrums_preserves_non_group_rejection(tmp_path):
    path = tmp_path / "raw.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(format="zcu.fluxdep-spectrums", version=2)
        handle.create_dataset("raw", data=np.array([1.0]))
    with pytest.raises(TypeError, match="raw"):
        load_spectrums(str(path))


def test_load_spectrums_preserves_missing_group_error(tmp_path):
    path = tmp_path / "incomplete.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(format="zcu.fluxdep-spectrums", version=2)
        handle.create_group("entry")
    with pytest.raises(KeyError):
        load_spectrums(str(path))


@pytest.mark.parametrize(
    "field",
    [
        "spectrum/dev_values",
        "spectrum/fluxs",
        "spectrum/freqs",
        "spectrum/signals",
        "points/dev_values",
        "points/fluxs",
        "points/freqs",
    ],
)
@pytest.mark.parametrize("kind", ["group", "datatype"])
def test_load_spectrums_rejects_non_dataset_fields_with_their_path(
    tmp_path, field, kind
):
    path = tmp_path / "wrong-kind.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs.update(format="zcu.fluxdep-spectrums", version=2)
        entry = handle.create_group("entry")
        entry.attrs.update(offset=0.25, period=1.0, device="bias", unit="A")
        for group_name, fields in (
            ("spectrum", ("dev_values", "fluxs", "freqs", "signals")),
            ("points", ("dev_values", "fluxs", "freqs")),
        ):
            group = entry.create_group(group_name)
            for name in fields:
                group.create_dataset(name, data=np.array([1.0]))
        del entry[field]
        if kind == "group":
            entry.create_group(field)
        else:
            entry[field] = np.dtype("float64")

    with pytest.raises(TypeError, match=f"/entry/{field}"):
        load_spectrums(str(path))
